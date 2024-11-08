# faiss_utils.py

import os
from langchain.vectorstores import FAISS  # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
from document_processing import read_files_from_directory
from typing import List
from langchain.docstore.document import Document

# Get the absolute path of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directories using absolute paths
VECTORSTORE_DIR = os.path.join(CURRENT_DIR, 'vectorstore', 'db_faiss')
INPUT_DIRECTORY = r"C:\Users\elias\OneDrive\Bureau\USEK\Semesters\Term-9_Fall-202510\GIN515-Deep Learning-non_repository\Files_dir_RAG"
FAISS_INDEX_FILE = os.path.join(VECTORSTORE_DIR, 'index.faiss')
FAISS_METADATA_FILE = os.path.join(VECTORSTORE_DIR, 'docstore.pkl')


def load_vector_store(vector_store_dir: str = VECTORSTORE_DIR) -> FAISS:
    """
    Loads the FAISS vector store from the specified directory.
    Raises FileNotFoundError if it does not exist.
    """
    if faiss_index_exists(vector_store_dir):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            print(f"[FAISS Utils] Loading FAISS vector store from {vector_store_dir}...")
            vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
            print("[FAISS Utils] FAISS vector store loaded successfully.")
            return vector_store
        except Exception as e:
            print(f"[FAISS Utils] Failed to load FAISS vector store: {e}.")
            raise e
    else:
        print("[FAISS Utils] FAISS vector store does not exist.")
        raise FileNotFoundError("FAISS vector store does not exist. Please create it using 'Refresh Vector Store'.")


def faiss_index_exists(vector_store_dir: str = VECTORSTORE_DIR) -> bool:
    """
    Checks if the FAISS index and metadata files exist.
    """
    index_exists = os.path.exists(FAISS_INDEX_FILE)
    metadata_exists = os.path.exists(FAISS_METADATA_FILE)
    print(f"[FAISS Utils] FAISS Index Exists: {index_exists}, Metadata Exists: {metadata_exists}")
    return index_exists and metadata_exists


def create_vector_store(vector_store_dir: str = VECTORSTORE_DIR, input_dir: str = INPUT_DIRECTORY) -> FAISS:
    """
    Creates a new FAISS vector store from documents in the input directory.
    """
    print(f"[FAISS Utils] Reading documents from {input_dir}...")
    documents: List[Document] = read_files_from_directory(input_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {input_dir}. Cannot create FAISS vector store.")

    print(f"[FAISS Utils] Embedding documents using HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print(f"[FAISS Utils] Creating FAISS vector store from {input_dir}...")
    vector_store = FAISS.from_documents(documents, embeddings)

    # Ensure the vectorstore directory exists
    os.makedirs(vector_store_dir, exist_ok=True)

    print(f"[FAISS Utils] Saving FAISS vector store to {vector_store_dir}...")
    vector_store.save_local(vector_store_dir)
    print("[FAISS Utils] FAISS vector store created and saved successfully.")
    return vector_store


def refresh_vector_store(vector_store_dir: str = VECTORSTORE_DIR, input_dir: str = INPUT_DIRECTORY) -> FAISS:
    """
    Refreshes the FAISS vector store by recreating it from scratch.
    """
    try:
        print("[FAISS Utils] Refreshing FAISS vector store...")
        vector_store = create_vector_store(vector_store_dir, input_dir)
        print("[FAISS Utils] FAISS vector store refreshed successfully.")
        return vector_store
    except Exception as e:
        print(f"[FAISS Utils] Failed to refresh FAISS vector store: {e}.")
        raise e
