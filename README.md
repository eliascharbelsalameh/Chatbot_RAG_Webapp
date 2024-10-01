# **Amanda (RAG Enhanced) - README**

### **Overview**

This application uses **Retrieval-Augmented Generation (RAG)** with **OpenAI's GPT-3.5** and **FAISS** for document retrieval. It allows you to query a set of documents (PDFs, Word documents, and Excel files) to get relevant answers based on the content. Additionally, the application provides cost calculations and token usage details for each response, improving transparency of API usage.

### **Prerequisites**

1. Python 3.8 or higher
2. OpenAI API Key

### **Packages Required**

```bash
streamlit==1.22.0
openai==0.28.0
langchain==0.0.183
sentence-transformers==2.2.2
faiss-cpu==1.7.4
pyperclip==1.8.2
tiktoken==0.4.0
docx==0.0.1
pdfplumber==0.5.28
pandas==1.5.3
```

You can install these dependencies via **pip**:

```bash
pip install streamlit openai langchain sentence-transformers faiss-cpu pyperclip tiktoken docx pdfplumber pandas
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

- **Retrieval-Augmented Generation (RAG)**: Combines GPT-3.5 with document retrieval for more accurate answers based on the provided documents.
- **FAISS Vector Store**: Uses FAISS to index the documents, enabling efficient similarity search for relevant information.
- **Multi-format Document Support**: The app now supports **PDF**, **Word (.docx)**, and **Excel (.xlsx)** files for richer document-based responses.
- **Multilingual Support**: Amanda can understand and respond in multiple languages, enhancing its versatility for diverse users and multilingual documents.
- **Token Usage and Cost Display**: The app calculates and displays the number of tokens used and the associated cost for each response, providing insights into the API usage.
- **Feedback Mechanism**: Users can like or dislike responses, providing a simple feedback mechanism.
- **Re-generate and Copy to Clipboard**: Allows re-generating responses and copying them to the clipboard for convenience.

### **New Updates**

- **New Document Formats Supported**: You can now upload Word and Excel documents in addition to PDFs.
- **Multilingual Feature**: Amanda is now multilingual, allowing it to answer questions and retrieve information in various languages.
- **UI Enhancements**: The "like," "dislike," "re-generate," and "copy" buttons are now smaller and have an outlined appearance, enhancing the interface for a cleaner look.
- **Improved Logging and Debugging**: Added logs for reading documents, processing files, and creating/loading FAISS vector stores for easier debugging.
- **Caching Enhancements**: Cached the loading and creation of the vector store to improve application performance.

### **Note**

- The FAISS index is created on the first run and stored in the `vectorstore/db_faiss/` directory.
- For future runs, the app will load the pre-existing FAISS index, speeding up startup time.
- The **tiktoken** library is used to count tokens and calculate costs, ensuring accurate billing information.

### **Troubleshooting**

- **Error: "expected string or buffer"**: Ensure that the messages passed to the tokenizer are valid strings. The code has been updated with type checks to handle such issues.
- If the FAISS index needs to be rebuilt (e.g., if documents have changed), you can delete the `db_faiss` folder and restart the app to recreate the index.
- **Excel File Issues**: If you encounter an error reading Excel files, ensure that the file is not open in another program and is in a supported format.

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

With these steps, you can manage your API keys more securely and conveniently.
