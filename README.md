# Stellantis Interview-Prep RAG

An open-source, fully-local RAG system that crawls `stellantis.com`, indexes it,
and answers focused interview-prep questions with cited sources.

Built on top of the original Amanda chatbot, fully rewired:

| Layer        | Before                                     | After                                                         |
|--------------|--------------------------------------------|---------------------------------------------------------------|
| LLM serving  | Llama-3.2-1B via llama-cpp-python (CPU)    | Llama-3.1-8B-Instruct Q4_K_M served by **LM Studio** (CUDA), temperature 0 |
| Embeddings   | `paraphrase-multilingual-mpnet-base-v2`    | **BAAI/bge-m3** (top multilingual, EN/FR/IT)                  |
| Reranker     | none                                       | **BAAI/bge-reranker-v2-m3** (optional, on by default)         |
| Retrieval    | top-k similarity                           | MMR (diversity) + cross-encoder rerank                        |
| Crawler      | depth ≤ 3, 100 pages, raw HTML kept        | depth = unlimited, polite, topic-tagged, JSONL stream         |
| Audio        | Deepgram (paid)                            | **faster-whisper** (local, free)                              |
| Config       | hardcoded Windows paths                    | `config.py` + `.env`                                          |
| Output       | none                                       | Auto-generated markdown interview brief                       |

No paid APIs.

---

## Hardware assumptions

The defaults target the machine this was built on: **Windows 11 + RTX 4060 8 GB**.
Override anything via `.env` (see `.env.example`).

| Need                 | Default                                                                  |
|----------------------|--------------------------------------------------------------------------|
| Local GGUF model     | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.6 GB, all layers on GPU)    |
| Embedding model      | `BAAI/bge-m3` (~2.3 GB, GPU)                                             |
| Reranker             | `BAAI/bge-reranker-v2-m3` (~600 MB, GPU)                                 |
| Whisper              | `base` int8_float16 on CUDA                                              |

CPU-only is supported (set `RAG_EMBED_DEVICE=cpu`, `RAG_LLM_N_GPU_LAYERS=0`,
`RAG_WHISPER_DEVICE=cpu`) but the brief generation will be slow.

---

## Quickstart

```powershell
# 1. install
pip install -r requirements.txt

# 2. (optional) copy and edit env
copy .env.example .env

# 3. START LM STUDIO SERVER (one-time per session)
#    - Open LM Studio
#    - Pick `lmstudio-community / Meta-Llama-3.1-8B-Instruct-GGUF / Q4_K_M`
#    - Click the `Developer` (or `Local Server`) tab on the left
#    - Click `Start Server`. Default endpoint: http://localhost:1234/v1
#    The Streamlit sidebar shows ✅/❌ for this connection.

# 4. crawl stellantis.com (depth=unlimited, capped at 2000 pages by default)
python crawler.py

# 4. build the FAISS index from the crawl
python build_index.py

# 5. generate the interview brief (one markdown file with all 11 topics)
python interview_prep.py

# 6. open the chat UI (also has Mock Interview mode)
streamlit run app.py
```

The brief lands in `data/briefs/stellantis_interview_brief.md` with cited URLs.

### Mock Interview mode

In the Streamlit UI, pick **Mock Interview** in the sidebar:

1. Choose a topic (or *Random*) and click **Pick a new question**.
2. The LLM drafts a realistic HR-style question grounded in retrieved Stellantis context.
3. You type your answer.
4. Click **Submit & Grade**: the LLM scores you 0–10, lists what you got right, what you missed, gives a model answer and a tougher follow-up question.
5. Your session-average score is tracked.

---

## What it answers (the 11 interview topics)

Defined in `config.INTERVIEW_TOPICS`:

1. Partners (labs and companies)
2. Goals (short / mid / long term)
3. Teams
4. Departments
5. Programs
6. Certifications
7. Agenda
8. Contributions
9. Research papers (count + names)
10. Stocks
11. Press releases

Edit the dict in [config.py](config.py) to add/remove topics — `interview_prep.py`
picks them up automatically.

---

## Project layout

```
config.py              # single source of truth (paths, models, crawl, retrieval)
crawler.py             # primary: async curl_cffi crawler (Chrome TLS impersonation)
stellantis_spider.py   # alternative: Scrapy spider (blocked by stellantis.com WAF)
indexing.py            # build / load FAISS from JSONL + local docs
build_index.py         # thin CLI wrapper
rag_chain.py           # MMR + reranker + LlamaCpp + cited answers
whisper_audio.py       # local faster-whisper transcription
interview_prep.py      # runs all 11 questions, writes markdown brief
mock_interview.py      # HR-style question generator + answer grader
app.py                 # new Streamlit UI (chat / mock interview / crawl / index / brief)
main.py                # original Amanda UI, kept as-is for reference
data/                  # crawl/, docs/, briefs/ (gitignored)
vectorstore/db_faiss/  # FAISS index files
```

---

## Reusability

Everything tunable is one of:

1. An env var (see `.env.example`).
2. A key in `config.py`.
3. The `INTERVIEW_TOPICS` dict (add your own question set for a different employer).

To point the system at any other site:

```powershell
$env:RAG_CRAWL_START_URLS = "https://www.example.com/"
$env:RAG_CRAWL_ALLOWED_DOMAINS = "example.com"
$env:RAG_CRAWL_OUTPUT = "example.jsonl"
python stellantis_spider.py
python build_index.py
streamlit run app.py
```

---

## Suggested upgrades (future)

- **Hybrid retrieval** — add BM25 alongside the dense index (langchain `EnsembleRetriever`) for keyword-heavy queries (ticker symbols, ISO codes).
- **Per-topic sub-indexes** — slice the JSONL by the `topics` tag and route each question to its own retriever for cleaner answers.
- **PDF extraction in the spider** — currently PDF *links* are logged but not parsed. Add `pdfplumber` to fetch + extract financial reports.
- **Sitemap-first crawling** — fetch `stellantis.com/sitemap.xml` to seed instead of pure link-following; faster and more complete.
- **Eval set** — write 20 ground-truth Q&A pairs and use Ragas / TruLens to score retrieval+answer quality before/after changes.
- **Switch to Qwen2.5-7B-Instruct or DeepSeek-V3-Distill-Qwen-7B** — competitive with Llama-3.1-8B and often stronger on structured extraction.
- **Stream tokens to the UI** — `LlamaCpp(streaming=True)` + `st.write_stream` for a more responsive chat.
- **Citations with span highlighting** — return char offsets per source so the UI can highlight the supporting sentence.

---

## License

MIT (inherits from the original repo).
