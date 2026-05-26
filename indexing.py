"""Build / load the FAISS index from crawled JSONL.

Sources supported:
  - JSONL produced by stellantis_spider (`{url, title, text, topics, ...}`)
  - Local files under DOCS_DIR (PDF, DOCX, XLSX, TXT, JSON) via document_processing.read_files_from_directory

Embeddings: BAAI/bge-m3 on CUDA by default (configurable in config.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config


def _embeddings(device: str | None = None) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": device or config.EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_crawl_jsonl(path: Path) -> Iterable[Document]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "") or ""
            if len(text) < 50:
                continue
            yield Document(
                page_content=text,
                metadata={
                    "source": obj.get("url", "unknown"),
                    "title": obj.get("title", ""),
                    "topics": ",".join(obj.get("topics", []) or []),
                    "depth": obj.get("depth", -1),
                    "kind": "web",
                },
            )


def load_local_docs() -> list[Document]:
    """Optional: pick up PDFs/DOCX/etc. from data/docs."""
    if not config.DOCS_DIR.exists() or not any(config.DOCS_DIR.iterdir()):
        return []
    from document_processing import read_files_from_directory  # local existing module
    return read_files_from_directory(str(config.DOCS_DIR))


def build_index(
    crawl_path: Path | None = None,
    include_local_docs: bool = True,
    save: bool = True,
) -> tuple[FAISS, list[Document]]:
    crawl_path = Path(crawl_path or config.CRAWL_OUTPUT)
    docs: list[Document] = []
    if crawl_path.exists():
        docs.extend(load_crawl_jsonl(crawl_path))
        print(f"[indexing] loaded {len(docs)} web pages from {crawl_path}")
    else:
        print(f"[indexing] WARNING: no crawl file at {crawl_path}")
    if include_local_docs:
        local = load_local_docs()
        if local:
            print(f"[indexing] loaded {len(local)} local docs from {config.DOCS_DIR}")
            docs.extend(local)
    if not docs:
        raise ValueError("No documents to index. Run the crawler first or drop files in data/docs/.")

    splitter = _splitter()
    chunks = splitter.split_documents(docs)
    print(f"[indexing] split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    embeddings = _embeddings()
    print(f"[indexing] embedding with {config.EMBED_MODEL} on {config.EMBED_DEVICE} ...")
    vs = FAISS.from_documents(chunks, embeddings)

    if save:
        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(config.VECTORSTORE_DIR))
        print(f"[indexing] FAISS saved to {config.VECTORSTORE_DIR}")
    return vs, chunks


def load_index() -> FAISS:
    if not (config.VECTORSTORE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index at {config.VECTORSTORE_DIR}. Run `python build_index.py` first."
        )
    # At query time use the lighter device — leaves VRAM for the LLM.
    return FAISS.load_local(
        str(config.VECTORSTORE_DIR),
        _embeddings(device=config.EMBED_DEVICE_QUERY),
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":
    build_index()
