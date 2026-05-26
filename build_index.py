"""CLI: build the FAISS index from the latest crawl + any local docs."""
from indexing import build_index

if __name__ == "__main__":
    build_index()
