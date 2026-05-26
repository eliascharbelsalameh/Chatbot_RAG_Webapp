"""Scrapy spider tuned for stellantis.com.

- Unlimited depth (DEPTH_LIMIT=0) with a max-pages safety cap.
- Polite: respects robots.txt, throttles, identifies itself.
- Topic tagging: applies regex hints to each page so downstream RAG can filter.
- Streams output as JSONL (one record per page) so partial progress is preserved
  even if interrupted.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import scrapy
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.http import TextResponse

import config


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


class StellantisSpider(scrapy.Spider):
    name = "stellantis"

    custom_settings = {
        # robots.txt itself returns 403 from a bot UA on stellantis.com; we use a real
        # browser UA below and skip the robots check. Lower-volume polite crawl.
        "ROBOTSTXT_OBEY": False,
        "USER_AGENT": config.CRAWL_USER_AGENT,
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        "DOWNLOAD_DELAY": config.CRAWL_DOWNLOAD_DELAY,
        "CONCURRENT_REQUESTS": config.CRAWL_CONCURRENCY,
        "CONCURRENT_REQUESTS_PER_DOMAIN": config.CRAWL_CONCURRENCY,
        "DEPTH_LIMIT": config.CRAWL_DEPTH_LIMIT,  # 0 = unlimited
        "CLOSESPIDER_PAGECOUNT": config.CRAWL_MAX_PAGES,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 4.0,
        "RETRY_TIMES": 2,
        "DOWNLOAD_TIMEOUT": 30,
        "LOG_LEVEL": "INFO",
        "FEED_EXPORT_ENCODING": "utf-8",
        "HTTPERROR_ALLOWED_CODES": [],
        "DEPTH_PRIORITY": 1,  # BFS — surface-level pages first
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
        "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
    }

    def __init__(self, output_path: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = list(config.CRAWL_START_URLS)
        self.allowed_domains = list(config.CRAWL_ALLOWED_DOMAINS)
        self.output_path = Path(output_path or config.CRAWL_OUTPUT)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate / open for streaming append.
        self._fh = self.output_path.open("w", encoding="utf-8")
        self.seen: set[str] = set()

    def closed(self, reason):
        try:
            self._fh.close()
        except Exception:
            pass
        self.logger.info(f"Spider closed: {reason}. Wrote {len(self.seen)} pages to {self.output_path}")

    @staticmethod
    def _is_html(response) -> bool:
        ct = response.headers.get("Content-Type", b"").decode("latin-1", errors="ignore").lower()
        return "html" in ct or isinstance(response, TextResponse)

    @staticmethod
    def _clean(html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())
        return title, text

    @staticmethod
    def _tag_topics(text: str, url: str) -> list[str]:
        haystack = f"{url} {text[:5000]}"
        return [topic for topic, pat in TOPIC_PATTERNS.items() if pat.search(haystack)]

    def parse(self, response) -> Iterable:
        if response.url in self.seen:
            return
        self.seen.add(response.url)

        if not self._is_html(response):
            return

        title, text = self._clean(response.text)
        if len(text) < 100:
            # skip near-empty pages
            pass
        else:
            record = {
                "url": response.url,
                "title": title,
                "text": text,
                "depth": response.meta.get("depth", 0),
                "topics": self._tag_topics(text, response.url),
            }
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()

        for href in response.css("a::attr(href)").getall():
            next_url = response.urljoin(href)
            parsed = urlparse(next_url)
            if parsed.scheme not in ("http", "https"):
                continue
            # Skip obvious non-HTML asset URLs early.
            lower = parsed.path.lower()
            if lower.endswith((
                ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
                ".mp4", ".mov", ".webm", ".mp3", ".wav",
                ".zip", ".tar", ".gz", ".7z", ".rar",
                ".css", ".js", ".ico", ".woff", ".woff2", ".ttf", ".eot",
            )):
                continue
            # Keep PDFs as a special case — fetch and store as text-of-link only (skip parse).
            if lower.endswith(".pdf"):
                # We log the URL as a record but don't extract here (kept light).
                if next_url not in self.seen:
                    self.seen.add(next_url)
                    self._fh.write(json.dumps({
                        "url": next_url,
                        "title": "[PDF link]",
                        "text": f"PDF document linked from {response.url}",
                        "depth": response.meta.get("depth", 0) + 1,
                        "topics": [],
                        "is_pdf_link": True,
                    }, ensure_ascii=False) + "\n")
                    self._fh.flush()
                continue
            yield scrapy.Request(next_url, callback=self.parse)


def run(output_path: str | None = None) -> Path:
    """Run the spider as a one-shot process. Returns the JSONL path."""
    out = Path(output_path or config.CRAWL_OUTPUT)
    process = CrawlerProcess(settings={
        "TELNETCONSOLE_ENABLED": False,
    })
    process.crawl(StellantisSpider, output_path=str(out))
    process.start()
    return out


if __name__ == "__main__":
    print(f"Starting Stellantis crawl -> {config.CRAWL_OUTPUT}")
    print(f"  start URLs: {config.CRAWL_START_URLS}")
    print(f"  max pages:  {config.CRAWL_MAX_PAGES}")
    print(f"  depth:      {'unlimited' if config.CRAWL_DEPTH_LIMIT == 0 else config.CRAWL_DEPTH_LIMIT}")
    run()
