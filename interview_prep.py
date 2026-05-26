"""Run all 11 interview-prep questions against the RAG and write a markdown brief.

Usage:
    python interview_prep.py
Output:
    data/briefs/stellantis_interview_brief.md
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

# Force unbuffered stdout so per-question progress is visible in real time.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

import config
from indexing import load_index
from rag_chain import answer, get_llm


def main():
    print("[brief] loading FAISS index ...")
    vs = load_index()
    print("[brief] loading LLM ...")
    llm = get_llm()

    out_md = config.BRIEFS_DIR / "stellantis_interview_brief.md"
    out_json = config.BRIEFS_DIR / "stellantis_interview_brief.json"
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    results: list[dict] = []
    md_lines = [
        "# Stellantis Interview Prep Brief",
        "",
        f"_Generated: {timestamp}_",
        "",
        f"_Model: `{Path(config.LLM_PATH).name}` · Embeddings: `{config.EMBED_MODEL}` · Reranker: `{config.RERANK_MODEL if config.USE_RERANKER else 'off'}`_",
        "",
        "---",
        "",
    ]

    for i, (topic, question) in enumerate(config.INTERVIEW_TOPICS.items(), 1):
        t0 = dt.datetime.now()
        print(f"[brief] {i}/{len(config.INTERVIEW_TOPICS)} {topic} ...", flush=True)
        try:
            res = answer(question, vs, llm=llm)
        except Exception as e:
            res = {"answer": f"ERROR: {e}", "sources": []}
        dt_s = (dt.datetime.now() - t0).total_seconds()
        print(f"[brief]   -> {len(res['answer'])} chars, {len(res['sources'])} sources, {dt_s:.1f}s", flush=True)
        results.append({"topic": topic, "question": question, **res})
        # Incremental write so partial progress survives a crash.
        (config.BRIEFS_DIR / "stellantis_interview_brief.partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_lines.append(f"## {i}. {topic.replace('_', ' ').title()}")
        md_lines.append("")
        md_lines.append(f"**Question:** {question}")
        md_lines.append("")
        md_lines.append(res["answer"])
        md_lines.append("")
        if res["sources"]:
            md_lines.append("**Sources:**")
            for s in res["sources"]:
                title = s.get("title") or s["url"]
                md_lines.append(f"- [{title}]({s['url']})")
            md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[brief] wrote {out_md}")
    print(f"[brief] wrote {out_json}")


if __name__ == "__main__":
    main()
