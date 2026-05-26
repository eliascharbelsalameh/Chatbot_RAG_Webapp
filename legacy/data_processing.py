# data_processing.py

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from typing import List

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

def split_into_chunks(text: str, max_length: int = 500) -> List[str]:
    """
    Splits the input text into chunks of maximum `max_length` words.
    
    Args:
        text (str): The text to be split.
        max_length (int): Maximum number of words per chunk.
        
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks