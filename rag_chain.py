"""Upgraded RAG chain.

- LlamaCpp w/ Llama-3.1-8B-Instruct (GGUF), temperature 0, full GPU offload.
- MMR retrieval to diversify hits across the crawl.
- Optional BGE reranker (bge-reranker-v2-m3) for precision.
- Source citations included in every answer.

Public API:
    get_llm()        -> LlamaCpp
    get_retriever(vs)-> retriever (with reranker if enabled)
    answer(query, vs, llm=None) -> {"answer": str, "sources": [dict]}
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

import config


def _to_text(result) -> str:
    """Coerce LangChain LLM/ChatModel output to plain str (handles AIMessage)."""
    return getattr(result, "content", result) if not isinstance(result, str) else result

SYSTEM_PROMPT = """You are an analyst preparing a candidate for an interview at Stellantis.
Use ONLY the provided context excerpts. If something is not in the context, say
"Not stated in the retrieved sources" — do NOT invent facts, names, or numbers.

Style rules:
- Keep the whole answer under ~10 short bullet points or 3 short paragraphs.
- Quote exact figures, dates, and names from the context.
- End with a complete sentence — do NOT trail off mid-sentence.
- Do NOT repeat the source URLs at the end; the UI shows them separately.

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["context", "question"])


STOP_SEQUENCES = [
    "\nQuestion:",
    "\n\nQuestion:",
    "\n[CDATA[",
    "<![CDATA[",
    "]]>",
    "\nNote:",
]


@lru_cache(maxsize=2)
def get_llm(streaming: bool = False) -> ChatOpenAI:
    """LM Studio's OpenAI-compatible endpoint. Returns a langchain ChatOpenAI."""
    return ChatOpenAI(
        base_url=config.LMSTUDIO_BASE_URL,
        api_key=config.LMSTUDIO_API_KEY,
        model=config.LMSTUDIO_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        streaming=streaming,
        stop=STOP_SEQUENCES,
        timeout=config.LLM_TIMEOUT,
    )


def lmstudio_ping() -> tuple[bool, str]:
    """Quick health check. Returns (ok, message)."""
    import urllib.request, urllib.error, json
    url = config.LMSTUDIO_BASE_URL.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
            ids = [m.get("id", "?") for m in data.get("data", [])]
            return True, f"LM Studio OK — loaded: {', '.join(ids) or '(none)'}"
    except urllib.error.URLError as e:
        return False, f"Cannot reach LM Studio at {url}: {e.reason}. In LM Studio, open the Developer / Local Server tab and click Start Server."
    except Exception as e:
        return False, f"Unexpected error pinging LM Studio: {e}"


def stream_answer(query: str, vs, llm=None):
    """Yield (token_chunk, sources_or_None). The final yield has sources_or_None set."""
    llm = llm or get_llm(streaming=True)
    docs = retrieve(query, vs)
    prompt = PROMPT.format(context=_format_context(docs), question=query)
    for chunk in llm.stream(prompt):
        yield _to_text(chunk), None
    seen = set()
    sources = []
    for d in docs:
        url = d.metadata.get("source", "unknown")
        if url in seen:
            continue
        seen.add(url)
        sources.append({
            "url": url,
            "title": d.metadata.get("title", ""),
            "topics": d.metadata.get("topics", ""),
        })
    yield "", sources


@lru_cache(maxsize=1)
def _reranker():
    """BGE reranker via sentence-transformers' CrossEncoder.

    We use CrossEncoder rather than FlagEmbedding.FlagReranker because the latter
    pulls in a heavy/broken dep chain (`warc3-wet-clueweb09`). The model weights
    are the same; behaviour is identical for our use case.
    """
    if not config.USE_RERANKER:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder(config.RERANK_MODEL, device=config.RERANK_DEVICE)
    except Exception as e:
        print(f"[rag_chain] reranker disabled ({e})")
        return None


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        title = d.metadata.get("title", "")
        head = f"[{i}] {title} — {src}" if title else f"[{i}] {src}"
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def retrieve(query: str, vs, k: int | None = None) -> list[Document]:
    fetch_k = config.RETRIEVER_FETCH_K
    k = k or config.RETRIEVER_K
    # MMR for diversity.
    docs = vs.max_marginal_relevance_search(query, k=fetch_k, fetch_k=fetch_k * 2, lambda_mult=0.5)
    rr = _reranker()
    if rr is None:
        return docs[:k]
    pairs = [(query, d.page_content) for d in docs]
    scores = rr.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[: config.RERANK_TOP_N]]


def answer(query: str, vs, llm: ChatOpenAI | None = None) -> dict[str, Any]:
    llm = llm or get_llm()
    docs = retrieve(query, vs)
    prompt = PROMPT.format(context=_format_context(docs), question=query)
    text = _to_text(llm.invoke(prompt)).strip()
    sources = []
    seen = set()
    for d in docs:
        url = d.metadata.get("source", "unknown")
        if url in seen:
            continue
        seen.add(url)
        sources.append({
            "url": url,
            "title": d.metadata.get("title", ""),
            "topics": d.metadata.get("topics", ""),
        })
    return {"answer": text, "sources": sources}
