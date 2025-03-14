# session_utils.py

import streamlit as st
import queue
import multiprocessing

def initialize_session_state():
    if 'all_chats' not in st.session_state:
        st.session_state['all_chats'] = [[]]  # List of chats, each chat is a list of messages
    if 'active_chat' not in st.session_state:
        st.session_state['active_chat'] = 0  # Index of the active chat
    if 'chat_titles' not in st.session_state:
        st.session_state['chat_titles'] = ["Chat 1"]  # Titles of each chat
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = []  # Feedback for each message
    if 'debug_log' not in st.session_state:
        st.session_state['debug_log'] = []  # Debugging logs
    if 'transcription' not in st.session_state:
        st.session_state['transcription'] = ""  # Transcribed audio
    if 'vector_store_loaded' not in st.session_state:
        st.session_state['vector_store_loaded'] = False  # Indicates if vector store is loaded
    if 'recording' not in st.session_state:
        st.session_state['recording'] = False  # Indicates if recording is in progress
    if 'qa_chain' not in st.session_state:
        st.session_state['qa_chain'] = None  # Retrieval-Augmented Generation chain
    
    # Initialize "chunks" to prevent KeyError
    if 'chunks' not in st.session_state:
        st.session_state['chunks'] = []  # List to store document chunks
    # Ensure 'feedback' and 'debug_log' exist and match the number of chats

def initialize_logging():
    if 'log_queue' not in st.session_state:
        st.session_state['log_queue'] = multiprocessing.Queue()

def log_debug(message):
    if 'log_queue' not in st.session_state:
        initialize_logging()
    st.session_state['log_queue'].put(message)
