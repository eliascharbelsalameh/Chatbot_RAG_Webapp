# data_processing.py

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore

def load_crawled_data(json_file_path="crawled_data.json"):
    """
    Loads crawled data from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing crawled data.
        
    Returns:
        list of dict: List containing source_url and content.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            crawled_data = json.load(f)
        return crawled_data
    except FileNotFoundError:
        raise FileNotFoundError(f"File {json_file_path} not found.")
    except json.JSONDecodeError:
        raise ValueError(f"File {json_file_path} is not a valid JSON.")

def split_into_chunks(crawled_data, chunk_size=500, chunk_overlap=50):
    """
    Splits the crawled content into text chunks.
    
    Args:
        crawled_data (list of dict): List containing source_url and content.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of overlapping characters between chunks.
        
    Returns:
        list of dict: List containing chunk content and metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for entry in crawled_data:
        source_url = entry.get("source_url", "Unknown")
        content = entry.get("content", "")
        split_texts = text_splitter.split_text(content)
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "source_url": source_url,
                "chunk_id": f"{source_url}_chunk_{i}",
                "content": chunk
            })
    return chunks
