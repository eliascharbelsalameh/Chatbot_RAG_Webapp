# **Amanda (RAG Enhanced) - README**

### **Overview**

This application leverages **Retrieval-Augmented Generation (RAG)** with **OpenAI's GPT-4-turbo** and **FAISS** for document retrieval, providing an AI-powered assistant, Amanda, who can answer questions based on the content of documents. The application supports **PDF**, **Word (DOCX)**, and **Excel (XLSX)** files. Additionally, it offers token usage and cost calculations for each response, enhancing transparency of API usage.

### **Prerequisites**

1. Python 3.8 or higher
2. OpenAI API Key

### **Packages Required**

```bash
streamlit==1.22.0
openai==1.51.0
langchain==0.0.183
sentence-transformers==2.2.2
faiss-cpu==1.7.4
pyperclip==1.8.2
tiktoken==0.4.0
docx==0.0.1
pdfplumber==0.5.28
pandas==1.5.3
gtts==2.2.3
pygame==2.1.2
langdetect==1.0.9
```

You can install these dependencies via **pip**:

```bash
pip install streamlit openai langchain sentence-transformers faiss-cpu pyperclip tiktoken docx pdfplumber pandas gtts pygame langdetect
```

### **OpenAI Setup**

1. Sign up for OpenAI and get your **API Key** [here](https://beta.openai.com/signup/).
2. Set the API Key as an environment variable:

**Linux/macOS:**

```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Windows (CMD):**

```bash
set OPENAI_API_KEY=your-api-key-here
```

### **Directory Structure**

1. Create a folder for your documents (PDFs, Word, and Excel files). Example:

```
project_folder/
│
├── app.py
├── vectorstore/
│   ├── db_faiss/  (Generated after first run)
└── Files_dir_RAG/
```

2. Place your documents in the `Files_dir_RAG/` directory.

### **Running the Application**

1. Navigate to the directory where the `app.py` file is located.
2. Run the application using Streamlit:

```bash
streamlit run app.py
```

3. The app will start in your browser where you can chat with Amanda and ask questions based on the documents.

### **Features**

- **Retrieval-Augmented Generation (RAG)**: Combines **GPT-4-turbo** with document retrieval for more accurate answers based on the content.
- **FAISS Vector Store**: Uses FAISS to index the documents, enabling efficient similarity search for relevant information.
- **Multi-format Document Support**: The app supports **PDF**, **Word (.docx)**, and **Excel (.xlsx)** files, providing rich document-based responses.
- **Multilingual Support**: Amanda can detect and respond in multiple languages, making it ideal for multilingual documents and diverse user bases.
- **Token Usage and Cost Display**: The app calculates and displays token usage and the associated cost for each response, providing insight into the API usage.
- **Text-to-Speech Functionality**: Amanda’s replies can be converted to speech using **gTTS**, with playback handled by **Pygame** for a more interactive experience.
- **Feedback Mechanism**: Users can like or dislike responses, providing a simple feedback mechanism to improve interactions.
- **Re-generate and Copy to Clipboard**: Allows users to regenerate responses or copy them to the clipboard for convenience.
- **Document Chunking**: Automatically splits large documents into chunks for better retrieval and response accuracy, with configurable chunk size and overlap.
- **Debugging Logs**: Logs activities like document processing and vector store creation, aiding in debugging.

### **New Updates**

- **Text-to-Speech**: Amanda now supports text-to-speech using **gTTS**, allowing users to listen to responses.
- **Chunking Configurations**: You can configure chunk size and overlap for better control over document splitting and retrieval.
- **Improved Multilingual Detection**: Amanda can now detect and respond in different languages, based on the input text.
- **Caching Enhancements**: Improved caching of the vector store and document chunks to speed up performance.
- **UI Improvements**: Better button placement and enhanced look for interaction buttons (like, dislike, re-generate, etc.).

### **Token Calculation and Cost Transparency**

The app provides detailed insights into token usage for each interaction. Based on the model used, token costs are calculated as follows:

- **GPT-4**: $0.03 per 1000 tokens
- **GPT-4-turbo**: $0.012 per 1000 tokens
- **GPT-3.5-turbo**: $0.002 per 1000 tokens

This feature helps in tracking API usage costs for each session.

### **Troubleshooting**

- **Error: "expected string or buffer"**: Ensure that the messages passed to the tokenizer are valid strings. The code has been updated with type checks to handle such issues.
- If the FAISS index needs to be rebuilt (e.g., if documents have changed), delete the `db_faiss` folder and restart the app to recreate the index.
- **Excel File Issues**: Ensure that Excel files are not open in another program and are in a supported format.

### **Environment Configuration (Optional)**

If you prefer to use a `.env` file to store your environment variables:

1. Install `python-dotenv`:

```bash
pip install python-dotenv
```

2. Create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-api-key-here
```

3. Update `app.py` to load environment variables from `.env`:

```python
from dotenv import load_dotenv
load_dotenv()
```

This allows you to manage API keys more securely.
