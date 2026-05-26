"""Single source of truth for paths, models, and crawl settings.

Override anything via environment variables or a .env file (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
CRAWL_DIR = Path(os.getenv("RAG_CRAWL_DIR", DATA_DIR / "crawl"))
VECTORSTORE_DIR = Path(os.getenv("RAG_VECTORSTORE_DIR", ROOT / "vectorstore" / "db_faiss"))
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", DATA_DIR / "docs"))
BRIEFS_DIR = Path(os.getenv("RAG_BRIEFS_DIR", DATA_DIR / "briefs"))

for d in (DATA_DIR, CRAWL_DIR, VECTORSTORE_DIR.parent, DOCS_DIR, BRIEFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----- LLM (LM Studio OpenAI-compatible server) -----
# Why LM Studio instead of llama-cpp-python? Because no llama-cpp-python wheel
# with both (a) CUDA support and (b) Llama-3.1 GGUF compatibility exists for
# Windows + Python 3.11 on the abetlen index. LM Studio ships its own CUDA
# runtime and serves the SAME GGUF file you already have cached. Start it via:
#     LM Studio -> select the model -> Developer/Local Server tab -> Start Server
LMSTUDIO_BASE_URL = os.getenv("RAG_LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.getenv("RAG_LMSTUDIO_API_KEY", "lm-studio")  # any non-empty string works
# Use the model identifier as shown in LM Studio's server tab. "auto" picks whatever is loaded.
LMSTUDIO_MODEL = os.getenv("RAG_LMSTUDIO_MODEL", "auto")
# Path to the GGUF (kept for documentation / legacy interop; no longer loaded directly).
LLM_PATH = os.getenv(
    "RAG_LLM_PATH",
    r"C:\Users\elias\.cache\lm-studio\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
)
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "1536"))
LLM_TIMEOUT = int(os.getenv("RAG_LLM_TIMEOUT", "120"))  # seconds

# ----- Embeddings -----
# BGE-M3 is top-tier multilingual (EN/FR/IT — all relevant for Stellantis). ~2.3GB.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
# Indexing batches 18k chunks — GPU is essential.
EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "cuda")
# At query time we embed ONE short string per question — CPU is fine and frees ~2.3GB
# VRAM for the LLM. Set to "cuda" only if you have headroom.
EMBED_DEVICE_QUERY = os.getenv("RAG_EMBED_DEVICE_QUERY", "cpu")

# ----- Retrieval -----
RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "8"))
RETRIEVER_FETCH_K = int(os.getenv("RAG_RETRIEVER_FETCH_K", "16"))
USE_RERANKER = os.getenv("RAG_USE_RERANKER", "true").lower() == "true"
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "5"))
# Reranker is small (~600MB) but every MB of VRAM matters on 8GB. CPU is ~200ms for 16 pairs.
RERANK_DEVICE = os.getenv("RAG_RERANK_DEVICE", "cpu")

# ----- Chunking -----
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

# ----- Crawl -----
CRAWL_START_URLS = [
    u.strip() for u in os.getenv(
        "RAG_CRAWL_START_URLS",
        "https://www.stellantis.com/en",
    ).split(",") if u.strip()
]
CRAWL_ALLOWED_DOMAINS = [
    d.strip() for d in os.getenv(
        "RAG_CRAWL_ALLOWED_DOMAINS",
        "stellantis.com",
    ).split(",") if d.strip()
]
CRAWL_MAX_PAGES = int(os.getenv("RAG_CRAWL_MAX_PAGES", "2000"))
CRAWL_DEPTH_LIMIT = int(os.getenv("RAG_CRAWL_DEPTH_LIMIT", "0"))  # 0 = unlimited in Scrapy
CRAWL_DOWNLOAD_DELAY = float(os.getenv("RAG_CRAWL_DOWNLOAD_DELAY", "0.5"))
CRAWL_CONCURRENCY = int(os.getenv("RAG_CRAWL_CONCURRENCY", "8"))
CRAWL_USER_AGENT = os.getenv(
    "RAG_CRAWL_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
)
CRAWL_OUTPUT = CRAWL_DIR / os.getenv("RAG_CRAWL_OUTPUT", "stellantis.jsonl")

# ----- Audio (optional, local) -----
WHISPER_MODEL = os.getenv("RAG_WHISPER_MODEL", "base")  # tiny|base|small|medium|large-v3
WHISPER_DEVICE = os.getenv("RAG_WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("RAG_WHISPER_COMPUTE", "int8_float16")

# ----- Interview topics (drives interview_prep.py) -----
INTERVIEW_TOPICS: dict[str, str] = {
    "partners_labs_companies": "List all of Stellantis' partners, including research labs, universities, suppliers, joint ventures, and corporate partners. For each, state the nature of the partnership and any cited goals or projects.",
    "goals_short_mid_long_term": "What are Stellantis' stated short-term, mid-term, and long-term strategic goals? Include the Dare Forward 2030 plan, electrification targets, carbon-neutrality targets, and financial targets. Use exact dates and numbers where given.",
    "teams": "What are the named teams, leadership groups, and executive committees at Stellantis? Include the Top Executive Team, the board, and any notable cross-functional teams.",
    "departments": "What are the main departments, business units, and functions inside Stellantis (e.g., engineering, software, manufacturing, design, purchasing, sustainability)? Briefly describe each.",
    "programs": "Which named programs, initiatives, and platforms does Stellantis run (e.g., STLA platforms, Dare Forward 2030, Circular Economy Hub, Software-Defined Vehicle program, Free2move)? Describe each and its objective.",
    "certifications": "Which certifications, standards, ratings, or audited credentials does Stellantis hold or pursue (ISO, SBTi, ESG ratings, safety, cybersecurity)? List each with the year if available.",
    "agenda": "What is the upcoming agenda for Stellantis: investor days, capital markets days, earnings releases, product launches, major events? Include dates.",
    "contributions": "What are Stellantis' notable contributions to research, open-source software, industry standards, philanthropy, and community programs?",
    "research_papers": "How many research papers has Stellantis published or co-authored, and what are their titles? Include conference/journal venues and years if mentioned.",
    "stocks": "Stock information for Stellantis: ticker symbols, listing venues, share structure, recent price commentary, dividend policy, share buyback programs.",
    "press_releases": "Summarize the most recent press releases from Stellantis, with date, headline, and a 1-2 sentence summary for each.",
}
