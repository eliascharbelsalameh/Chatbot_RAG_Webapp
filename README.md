---

# **Amanda (RAG Enhanced) - README**

### **Overview**

This application uses **Retrieval-Augmented Generation (RAG)** with **OpenAI's GPT-3.5** and **FAISS** for document retrieval. It allows you to query a set of documents (PDFs) to get relevant answers based on the content.

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
```

You can install these dependencies via **pip**:

```bash
pip install streamlit openai langchain sentence-transformers faiss-cpu pyperclip
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

1. Create a folder for your PDFs. Example:

```
project_folder/
│
├── app.py
├── vectorstore/
│   ├── db_faiss/  (Generated after first run)
└── PDFs/ 
```

2. Place your PDFs in the `PDFs/` directory.

### **Running the Application**

1. Navigate to the directory where the `app.py` file is located.
2. Run the application using Streamlit:

```bash
streamlit run app.py
```

3. The app will start in your browser where you can chat with Amanda and ask questions based on the documents.

### **Note**

- The FAISS index is created on the first run and stored in the `vectorstore/db_faiss/` directory.
- For future runs, the app will load the pre-existing FAISS index, speeding up startup time.
