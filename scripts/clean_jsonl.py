"""Strip CDATA wrappers and redundant whitespace from an existing crawl JSONL.

Run once after a crawl to remove `<![CDATA[ ... ]]>` artifacts and the like that
leaked through from CMS-rendered pages.

Usage:
    python scripts/clean_jsonl.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CDATA_RE = re.compile(r"<!\[CDATA\[|\]\]>")
WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    text = CDATA_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def main(path: Path | None = None):
    src = Path(path or config.CRAWL_OUTPUT)
    if not src.exists():
        raise SystemExit(f"no crawl file at {src}")
    dst = src.with_suffix(".cleaned.jsonl")
    n_in = n_out = n_changed = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            original = rec.get("text", "")
            cleaned = clean(original)
            if cleaned != original:
                n_changed += 1
            rec["text"] = cleaned
            if len(cleaned) >= 50:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    src.replace(src.with_suffix(".pre-clean.jsonl"))
    dst.replace(src)
    print(f"[clean_jsonl] in={n_in} out={n_out} changed={n_changed}  ->  {src}")
    print(f"[clean_jsonl] backup saved as {src.with_suffix('.pre-clean.jsonl').name}")


if __name__ == "__main__":
    main()
