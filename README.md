# README: AI Assistant Chatbot for USEK Students

This project is an AI-powered assistant chatbot designed for USEK students. The chatbot leverages Retrieval-Augmented Generation (RAG) to provide context-aware, reliable responses by integrating large language models with a document vector store. The assistant can also transcribe audio messages, crawl the web for relevant information, and maintain a dynamic interaction history.

---

## Features

1. **RAG-Enhanced Chatbot**:
   - Combines LLM (Large Language Model) with a vector store for accurate retrieval and response.
   - Supports context-based queries with source document tracing.

2. **Audio Integration**:
   - Transcribes user audio messages to text for seamless interaction.

3. **Web Crawler**:
   - Gathers content from specified URLs to expand the chatbot's knowledge base.

4. **User-Friendly Interface**:
   - Built using Streamlit for real-time interactions.

5. **Feedback Mechanism**:
   - Users can provide feedback on chatbot responses for iterative improvement.

6. **Document Management**:
   - Processes large documents into retrievable chunks stored in a FAISS vector store.

---

## Installation

### Prerequisites

1. Python 3.8+
2. NVIDIA GPU with CUDA (optional for GPU support).
3. Anaconda/Miniconda (recommended).

---

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/ai-assistant-chatbot
   cd ai-assistant-chatbot
   ```

2. Create a virtual environment:
   ```bash
   conda create --name chatbot_env python=3.9
   conda activate chatbot_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration for GPU Support (Optional)

1. Install `llama-cpp-python` with GPU acceleration:
   ```bash
   set LLAMA_CUBLAS=1
   pip install -e .
   ```

2. Ensure CUDA is properly installed:
   - Verify with:
     ```bash
     nvcc --version
     ```
   - Install necessary drivers and libraries from NVIDIA if needed.

---

## Usage

1. **Run the Chatbot**:
   Start the chatbot using:
   ```bash
   streamlit run main.py
   ```

2. **Interacting with the Chatbot**:
   - Visit `http://localhost:8501` in your browser.
   - Use the interface to ask questions, upload documents, or provide feedback.

3. **Add Documents**:
   - Place documents in the `data` folder for processing.
   - Refresh the vector store using the interface.

4. **Web Crawler**:
   - Use the "Web Crawl" page to gather content from external websites.
   - Specify the URL, crawl depth, and maximum pages to crawl.

---

## File Structure

```plaintext
.
├── main.py                  # Main Streamlit app
├── audio_processing.py      # Audio transcription module
├── audio_utils.py           # Helper functions for audio processing
├── data_processing.py       # Document processing and chunking
├── document_processing.py   # Handles document ingestion
├── faiss_utils.py           # FAISS vector store management
├── rag_chain.py             # RAG chain creation
├── session_utils.py         # Session state management for Streamlit
├── scrapy_spider.py         # Scrapy spider for crawling
├── web_crawl.py             # Web crawler utility
├── requirements.txt         # Required Python dependencies
└── README.md                # This file
```

---

## Dependencies

- **Streamlit**: Real-time interface.
- **LangChain**: Chain-based reasoning for LLMs.
- **FAISS**: Fast Approximate Nearest Neighbor Search.
- **Transformers**: Hugging Face library for LLMs.
- **Deepgram SDK**: Audio transcription.
- **BeautifulSoup**: Web scraping.
- **Scrapy**: Advanced web crawling.

---

## Known Issues

1. **Compatibility with Libraries**:
   Ensure compatible versions of `pydantic` and `langchain` are installed.

2. **CUDA Errors**:
   Verify GPU drivers and CUDA toolkit installation if running in GPU mode.

---

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes and create a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
