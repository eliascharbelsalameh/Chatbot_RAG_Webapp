"""Async web crawler using curl_cffi (Chrome TLS impersonation).

Why not Scrapy here: stellantis.com is fronted by a WAF (Akamai/Cloudflare-style)
that blocks any client whose TLS handshake doesn't look like a real browser. curl_cffi
impersonates Chrome's fingerprint and gets through.

Public API:
    crawl(start_urls=..., allowed_domains=..., max_pages=..., output=Path, ...) -> Path

CLI:
    python crawler.py
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urldefrag

from bs4 import BeautifulSoup
from curl_cffi import requests as crequests  # type: ignore

import config

SKIP_EXT = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp", ".ico",
    ".mp4", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".ogg",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".dmg", ".exe", ".msi",
)

TOPIC_PATTERNS: dict[str, re.Pattern] = {
    "partners_labs_companies": re.compile(r"partner|collaborat|joint venture|consortium|alliance|supplier|stellab", re.I),
    "goals_short_mid_long_term": re.compile(r"dare forward|2030|strateg|target|ambition|roadmap|net[- ]?zero|carbon neutral", re.I),
    "teams": re.compile(r"leadership|executive|board of directors|management|top executive", re.I),
    "departments": re.compile(r"engineering|manufacturing|design|purchasing|software|sustainability|finance|human resources", re.I),
    "programs": re.compile(r"stla|free2move|mobilisights|circular economy|software[- ]?defined vehicle|sdv|battery hub", re.I),
    "certifications": re.compile(r"iso\s?\d|sbti|certif|standard|rating|audited|ecovadis|cdp|msci", re.I),
    "agenda": re.compile(r"investor day|capital markets|earnings|results|agenda|calendar|event", re.I),
    "contributions": re.compile(r"foundation|donation|philanthrop|open[- ]?source|grant|scholarship|community", re.I),
    "research_papers": re.compile(r"research paper|publication|journal|conference|ieee|sae|proceedings", re.I),
    "stocks": re.compile(r"share|stock|ticker|dividend|buyback|nyse|euronext|borsa", re.I),
    "press_releases": re.compile(r"press release|news|announce|statement", re.I),
}


def _is_allowed(url: str, allowed_domains: list[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


def _is_html_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return not path.endswith(SKIP_EXT)


def _clean_html(html: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    links = []
    for a in soup.find_all("a", href=True):
        links.append(a["href"])
    return title, text, links


def _tag_topics(text: str, url: str) -> list[str]:
    hay = f"{url} {text[:5000]}"
    return [t for t, pat in TOPIC_PATTERNS.items() if pat.search(hay)]


def _normalize(base: str, href: str) -> str | None:
    if not href:
        return None
    href = href.strip()
    if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
        return None
    try:
        from urllib.parse import urljoin
        absolute = urljoin(base, href)
        absolute, _ = urldefrag(absolute)
        if urlparse(absolute).scheme not in ("http", "https"):
            return None
        return absolute
    except Exception:
        return None


async def _fetch(session, url: str, timeout: int = 25):
    return await session.get(url, impersonate="chrome", timeout=timeout, allow_redirects=True)


async def crawl_async(
    start_urls: list[str],
    allowed_domains: list[str],
    output: Path,
    max_pages: int,
    concurrency: int,
    delay: float,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fh = output.open("w", encoding="utf-8")

    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    for u in start_urls:
        if u not in seen:
            seen.add(u)
            queue.append((u, 0))

    sem = asyncio.Semaphore(concurrency)
    pages_written = 0
    start_time = time.monotonic()
    lock = asyncio.Lock()

    async with crequests.AsyncSession() as session:

        async def worker(url: str, depth: int):
            nonlocal pages_written
            async with sem:
                if delay:
                    await asyncio.sleep(delay)
                try:
                    r = await _fetch(session, url)
                except Exception as e:
                    return [], None
                ct = (r.headers.get("content-type") or "").lower()
                if r.status_code != 200 or "html" not in ct:
                    return [], None
                title, text, raw_links = _clean_html(r.text)
                record = None
                if len(text) >= 100:
                    record = {
                        "url": url,
                        "title": title,
                        "text": text,
                        "depth": depth,
                        "topics": _tag_topics(text, url),
                    }
                new_links: list[tuple[str, int]] = []
                for href in raw_links:
                    nxt = _normalize(url, href)
                    if not nxt or nxt in seen:
                        continue
                    if not _is_allowed(nxt, allowed_domains):
                        continue
                    if not _is_html_url(nxt):
                        continue
                    seen.add(nxt)
                    new_links.append((nxt, depth + 1))
                return new_links, record

        while queue and pages_written < max_pages:
            batch = []
            while queue and len(batch) < concurrency and pages_written + len(batch) < max_pages:
                batch.append(queue.popleft())
            if not batch:
                break
            results = await asyncio.gather(*(worker(u, d) for u, d in batch), return_exceptions=False)
            async with lock:
                for new_links, record in results:
                    if record and pages_written < max_pages:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pages_written += 1
                    for link in new_links:
                        if link[0] not in (q[0] for q in queue):
                            queue.append(link)
                fh.flush()
            elapsed = time.monotonic() - start_time
            rate = pages_written / max(elapsed, 0.1)
            print(f"[crawler] pages={pages_written}/{max_pages} queue={len(queue)} rate={rate:.1f}/s elapsed={elapsed:.0f}s")

    fh.close()
    print(f"[crawler] done. {pages_written} pages -> {output}")
    return output


def crawl(
    start_urls: Iterable[str] | None = None,
    allowed_domains: Iterable[str] | None = None,
    output: Path | None = None,
    max_pages: int | None = None,
    concurrency: int | None = None,
    delay: float | None = None,
) -> Path:
    return asyncio.run(crawl_async(
        start_urls=list(start_urls or config.CRAWL_START_URLS),
        allowed_domains=list(allowed_domains or config.CRAWL_ALLOWED_DOMAINS),
        output=Path(output or config.CRAWL_OUTPUT),
        max_pages=max_pages or config.CRAWL_MAX_PAGES,
        concurrency=concurrency or config.CRAWL_CONCURRENCY,
        delay=delay if delay is not None else config.CRAWL_DOWNLOAD_DELAY,
    ))


if __name__ == "__main__":
    print(f"Stellantis crawl -> {config.CRAWL_OUTPUT}")
    print(f"  start: {config.CRAWL_START_URLS}")
    print(f"  max pages: {config.CRAWL_MAX_PAGES}")
    print(f"  concurrency: {config.CRAWL_CONCURRENCY}  delay: {config.CRAWL_DOWNLOAD_DELAY}s")
    crawl()
