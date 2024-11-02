import os
from langchain.vectorstores import FAISS # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings # type: ignore
from document_processing import read_files_from_directory

vectorstore_dir = 'vectorstore/db_faiss'
input_directory = r"C:\Users\elias\OneDrive\Bureau\USEK-Electrical and Electronics Engineer\Semesters\Term-9_Fall-202510\GIN515-Deep Learning-non_repository\Files_dir_RAG"

def load_vector_store():
    if faiss_index_exists():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            return FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return recreate_vector_store()
    return recreate_vector_store()

def faiss_index_exists():
    index_file = os.path.join(vectorstore_dir, 'index.faiss')
    metadata_file = os.path.join(vectorstore_dir, 'docstore.pkl')
    return os.path.exists(index_file) and os.path.exists(metadata_file)

def recreate_vector_store():
    documents = read_files_from_directory(input_directory)
    if not documents:
        raise ValueError("No documents found. Cannot create FAISS vector store.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(vectorstore_dir)
    return vectorstore
