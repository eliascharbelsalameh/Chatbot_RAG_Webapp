# legacy/

Archived files from the original USEK *Amanda* chatbot project. Nothing in the
active system imports anything from here — kept for reference only.

| File | Notes |
|---|---|
| `main.py` | Original Streamlit app. Uses Deepgram (paid), downloads BART summarizer, depends on the siblings below. Replaced by `../app.py`. |
| `faiss_utils.py` | Old FAISS layer; the hardcoded USEK path was repointed to `../data/docs/`. Replaced by `../indexing.py`. |
| `audio_processing.py`, `audio_utils.py` | Deepgram transcription. Replaced by `../whisper_audio.py` (local faster-whisper). |
| `session_utils.py` | Streamlit session helpers used only by `main.py`. |
| `data_processing.py` | Old chunker. Cleaned (the duplicate `split_into_chunks` was removed). |
| `scrapy_spider.py`, `web_crawl.py`, `stellantis_spider.py` | Earlier crawler attempts. Replaced by `../crawler.py` (curl_cffi, Chrome TLS impersonation — required because `stellantis.com` 403s any non-browser TLS client). |
| 4 PDFs | Academic submission artifacts from the original semester project. |

Safe to delete this folder entirely if you no longer want the original project preserved.
