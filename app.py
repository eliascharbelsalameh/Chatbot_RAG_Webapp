"""Streamlit UI for the upgraded Stellantis RAG.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

import config
from indexing import load_index
from rag_chain import answer, get_llm, stream_answer, lmstudio_ping

st.set_page_config(page_title="Stellantis Interview Prep RAG", layout="wide")

PAGES = ["Chat", "Mock Interview", "Run Crawler", "Build Index", "Generate Brief", "Settings"]


@st.cache_resource(show_spinner="Loading vector store...")
def _vs():
    return load_index()


@st.cache_resource(show_spinner="Loading LLM...")
def _llm():
    return get_llm(streaming=True)


def _index_ready() -> bool:
    return (config.VECTORSTORE_DIR / "index.faiss").exists()


st.sidebar.title("Stellantis RAG")
page = st.sidebar.radio("Page", PAGES)
st.sidebar.markdown("---")
_lm_ok, _lm_msg = lmstudio_ping()
st.sidebar.caption(f"LM Studio: {'✅' if _lm_ok else '❌'}  `{config.LMSTUDIO_BASE_URL}`")
if not _lm_ok:
    st.sidebar.warning(_lm_msg)
st.sidebar.caption(f"Model GGUF: `{Path(config.LLM_PATH).name}`")
st.sidebar.caption(f"Embed: `{config.EMBED_MODEL}`")
def _reranker_status() -> str:
    if not config.USE_RERANKER:
        return "off (disabled in config)"
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        return "on"
    except ImportError:
        return "off (pip install sentence-transformers)"


st.sidebar.caption(f"Reranker: `{_reranker_status()}`")
st.sidebar.caption(f"Index: {'✅' if _index_ready() else '❌'}  ({config.VECTORSTORE_DIR})")
st.sidebar.caption(f"Crawl JSONL: {'✅' if Path(config.CRAWL_OUTPUT).exists() else '❌'}")


if page == "Chat":
    st.title("Chat — Stellantis RAG")
    if not _index_ready():
        st.warning("No FAISS index yet. Go to **Build Index** after running the crawler.")
        st.stop()
    vs, llm = _vs(), _llm()

    if "history" not in st.session_state:
        st.session_state.history = []
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for s in turn["sources"]:
                        st.markdown(f"- [{s.get('title') or s['url']}]({s['url']})")

    q = st.chat_input("Ask about Stellantis...")
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Retrieving..."):
                token_iter = stream_answer(q, vs, llm=llm)
            buf = ""
            sources: list = []
            last_render = 0.0
            import time as _t
            for chunk, srcs in token_iter:
                if srcs is not None:
                    sources = srcs
                    continue
                buf += chunk
                # Throttle re-renders to ~12 fps so the browser keeps up with the GPU.
                now = _t.monotonic()
                if now - last_render > 0.08:
                    placeholder.markdown(buf + "▌")
                    last_render = now
            placeholder.markdown(buf)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- [{s.get('title') or s['url']}]({s['url']})")
        st.session_state.history.append({
            "role": "assistant",
            "content": buf,
            "sources": sources,
        })

elif page == "Mock Interview":
    st.title("Mock Interview — Stellantis HR Drill")
    st.caption("The LLM asks a Stellantis-grounded question. You answer. It grades you against the source context and gives a model answer + follow-up.")
    if not _index_ready():
        st.warning("Build the FAISS index first.")
        st.stop()
    vs, llm = _vs(), _llm()

    from mock_interview import drill_question, grade_answer

    if "mi_round" not in st.session_state:
        st.session_state.mi_round = None
        st.session_state.mi_history = []  # list of {q, a, grade}

    topic_choice = st.selectbox(
        "Topic (or Random)",
        ["Random"] + list(config.INTERVIEW_TOPICS.keys()),
        index=0,
    )

    cols = st.columns([1, 1, 3])
    if cols[0].button("Pick a new question", type="primary"):
        with st.spinner("Drafting question..."):
            t = None if topic_choice == "Random" else topic_choice
            st.session_state.mi_round = drill_question(vs, llm=llm, topic=t)
            st.session_state.mi_answer = ""
    if cols[1].button("Reset session"):
        st.session_state.mi_round = None
        st.session_state.mi_history = []
        st.session_state.mi_answer = ""

    rnd = st.session_state.mi_round
    if rnd:
        st.markdown(f"### Topic: `{rnd['topic']}`")
        st.markdown(f"**HR:** {rnd['question']}")
        user_answer = st.text_area(
            "Your answer",
            value=st.session_state.get("mi_answer", ""),
            height=200,
            placeholder="Speak the way you would in the actual interview...",
        )
        st.session_state.mi_answer = user_answer

        if st.button("Submit & Grade"):
            with st.spinner("Grading..."):
                grade = grade_answer(
                    rnd["question"], user_answer, rnd["ground_truth_context"], llm=llm
                )
            st.session_state.mi_history.append({
                "topic": rnd["topic"],
                "question": rnd["question"],
                "answer": user_answer,
                "grade": grade,
            })

            st.markdown(f"## Score: **{grade.get('score', 0)} / 10**")
            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### What you got right")
                st.markdown(grade.get("what_you_got_right") or "_–_")
            with colB:
                st.markdown("#### What you missed")
                st.markdown(grade.get("what_you_missed") or "_–_")
            st.markdown("#### Model answer")
            st.markdown(grade.get("model_answer") or "_–_")
            st.markdown("#### Possible follow-up")
            st.info(grade.get("follow_up") or "_–_")
            with st.expander("Source pages used"):
                for s in rnd["sources"]:
                    st.markdown(f"- [{s.get('title') or s['url']}]({s['url']})")

    if st.session_state.mi_history:
        with st.expander(f"Session history ({len(st.session_state.mi_history)} answers)"):
            scores = [h["grade"].get("score", 0) for h in st.session_state.mi_history]
            avg = sum(scores) / max(len(scores), 1)
            st.metric("Average score", f"{avg:.1f} / 10")
            for h in st.session_state.mi_history:
                st.markdown(f"- **{h['topic']}** — {h['grade'].get('score', 0)}/10 — {h['question']}")

elif page == "Run Crawler":
    st.title("Run Stellantis Crawler")
    st.write("Crawl `stellantis.com` (depth=unlimited) up to a page cap. Streams JSONL to disk.")
    cols = st.columns(3)
    with cols[0]:
        max_pages = st.number_input("Max pages", 50, 50000, config.CRAWL_MAX_PAGES, step=50)
    with cols[1]:
        delay = st.number_input("Download delay (s)", 0.0, 5.0, config.CRAWL_DOWNLOAD_DELAY, step=0.1)
    with cols[2]:
        concurrency = st.number_input("Concurrency", 1, 32, config.CRAWL_CONCURRENCY, step=1)
    start_urls = st.text_input("Start URLs (comma-sep)", ",".join(config.CRAWL_START_URLS))

    if st.button("Start crawl", type="primary"):
        env = os.environ.copy()
        env["RAG_CRAWL_MAX_PAGES"] = str(max_pages)
        env["RAG_CRAWL_DOWNLOAD_DELAY"] = str(delay)
        env["RAG_CRAWL_CONCURRENCY"] = str(concurrency)
        env["RAG_CRAWL_START_URLS"] = start_urls
        with st.spinner("Crawling — this can take a while. Watch terminal for live logs."):
            proc = subprocess.run(
                [sys.executable, "crawler.py"],
                cwd=str(Path(__file__).parent),
                env=env,
            )
        st.success(f"Crawler exited with code {proc.returncode}. Output: `{config.CRAWL_OUTPUT}`")

elif page == "Build Index":
    st.title("Build / Refresh FAISS Index")
    st.write(f"Reads `{config.CRAWL_OUTPUT}` (and any files in `{config.DOCS_DIR}`).")
    if st.button("Build index now", type="primary"):
        with st.spinner("Embedding + indexing..."):
            from indexing import build_index
            vs, chunks = build_index()
        st.success(f"Indexed {len(chunks)} chunks. Saved to `{config.VECTORSTORE_DIR}`.")
        st.cache_resource.clear()

elif page == "Generate Brief":
    st.title("Generate Interview Brief")
    st.write("Runs all 11 target queries and writes a markdown brief to `data/briefs/`.")
    if not _index_ready():
        st.warning("Build the index first.")
        st.stop()
    if st.button("Generate", type="primary"):
        from interview_prep import main as run_brief
        with st.spinner("Asking 11 questions..."):
            run_brief()
        brief = config.BRIEFS_DIR / "stellantis_interview_brief.md"
        if brief.exists():
            st.success(f"Wrote {brief}")
            st.download_button("Download brief.md", brief.read_bytes(), file_name=brief.name)
            st.markdown(brief.read_text(encoding="utf-8"))

elif page == "Settings":
    st.title("Settings (read-only)")
    st.write("Edit `config.py` or set environment variables / `.env` to change these.")
    st.json({
        "LLM_PATH": config.LLM_PATH,
        "LLM_N_CTX": config.LLM_N_CTX,
        "LLM_N_GPU_LAYERS": config.LLM_N_GPU_LAYERS,
        "LLM_TEMPERATURE": config.LLM_TEMPERATURE,
        "EMBED_MODEL": config.EMBED_MODEL,
        "USE_RERANKER": config.USE_RERANKER,
        "RERANK_MODEL": config.RERANK_MODEL,
        "RETRIEVER_K": config.RETRIEVER_K,
        "CHUNK_SIZE": config.CHUNK_SIZE,
        "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
        "CRAWL_START_URLS": config.CRAWL_START_URLS,
        "CRAWL_MAX_PAGES": config.CRAWL_MAX_PAGES,
        "CRAWL_DEPTH_LIMIT": config.CRAWL_DEPTH_LIMIT,
    })
