# main.py

import streamlit as st
import openai
import os
import pyperclip  # type: ignore
from deepgram import Deepgram  # type: ignore
import tempfile
import sounddevice as sd  # type: ignore
from transformers import pipeline  # type: ignore
from groq import Groq  # type: ignore
import json
import threading  # To handle threading for non-blocking operations
import tiktoken # type: ignore

from audio_utils import clear_audio_files, play_audio, record_audio, transcribe_audio_v3
from document_processing import read_files_from_directory
from faiss_utils import load_vector_store, refresh_vector_store
from rag_chain import create_rag_chain
from session_utils import initialize_session_state, log_debug
from audio_processing import get_recorder
from web_crawl import WebCrawler
from data_processing import load_crawled_data, split_into_chunks

# TODO: OLAMA download LLAMA 3.2 8b 
# TODO: apply chunking to the 
# TODO: fix whisper
# TODO: load files online
# TODO: add chat history
# TODO: enhance tables
# TODO: load file online
# TODO: embedding text 3

# Initialize Streamlit app
st.set_page_config(page_title="Amanda Chatbot", layout="wide")

# Initialize session state
initialize_session_state()

# Initialize the recorder
recorder = get_recorder()

# Define supported model and temperature
used_model = "gpt-4-turbo"
temperature = 0.01

# main.py (add this function below the imports and initializations)

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the given text using the initialized tokenizer.
    
    Args:
        text (str): The text to count tokens for.
        
    Returns:
        int: The number of tokens.
    """
    return len(encoding.encode(text))

# Initialize the tokenizer
try:
    encoding = tiktoken.encoding_for_model(used_model)
    log_debug(f"[Main] Tokenizer initialized for model: {used_model}")
except KeyError:
    # Fallback if the model is not recognized
    encoding = tiktoken.get_encoding("cl100k_base")
    log_debug("[Main] Fallback tokenizer initialized.")

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Deepgram API client
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Ensure this environment variable is set
if DEEPGRAM_API_KEY:
    deepgram_client = Deepgram(DEEPGRAM_API_KEY)
    log_debug("[Main] Deepgram client initialized.")
else:
    deepgram_client = None
    log_debug("[Main] Deepgram API key not found. Audio transcription will not work.")

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Clear any existing audio files
clear_audio_files()

# Define pages for the Streamlit application
PAGES = ["Chat with Amanda", "Debugging Logs", "User Feedback", "All Chunks", "Web Crawl"]

# Set up Streamlit sidebar
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", PAGES)

# Add "Refresh Vector Store" button in the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Vector Store"):
    with st.spinner("Refreshing the vector store..."):
        try:
            vector_store, chunks = refresh_vector_store()
            qa_chain = create_rag_chain(vector_store, used_model)
            st.session_state['qa_chain'] = qa_chain  # Store qa_chain in session state
            st.session_state['vector_store_loaded'] = True  # Indicate that vector store is loaded
            st.session_state['chunks'] = chunks  # Store chunks in session state
            st.success("Vector store refreshed successfully.")
            log_debug("Vector store refreshed successfully.")
        except Exception as e:
            st.error(f"Failed to refresh vector store: {e}")
            log_debug(f"Failed to refresh vector store: {e}")

# Add "Chats" section with "New Chat" button
st.sidebar.markdown("---")
st.sidebar.markdown("## Chats")
if st.sidebar.button("➕ New Chat"):
    st.session_state["all_chats"].append([])  # Create a new chat session
    st.session_state["active_chat"] = len(st.session_state["all_chats"]) - 1  # Set this as the active chat
    st.session_state["chat_titles"] = st.session_state.get("chat_titles", [])
    st.session_state["chat_titles"].append(f"Chat {len(st.session_state['all_chats'])}")

# Display chat list as a fixed list in the sidebar with an outline for the active chat
for i, title in enumerate(st.session_state.get("chat_titles", [])):
    if i == st.session_state["active_chat"]:
        st.sidebar.markdown(
            f"<div style='border: 2px solid #4CAF50; padding: 5px;'>{title}</div>",
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(title, key=f"chat_{i}"):
            st.session_state["active_chat"] = i  # Set the selected chat as active

# Add "Audio Recorder" section in the sidebar
st.sidebar.markdown("---")
st.sidebar.title("Audio Recorder")
st.sidebar.write("Record and transcribe audio using Deepgram to talk to Amanda.")

rec_col1, rec_col2 = st.sidebar.columns([1, 1])

with rec_col1:
    if st.sidebar.button("Start Recording"):
        if not st.session_state.get("recording", False):
            recorder.start_recording()
            st.session_state["recording"] = True
            st.success("Recording started.")
            log_debug("Recording started.")
        else:
            st.warning("Recording is already in progress.")

with rec_col2:
    if st.sidebar.button("Stop Recording"):
        if st.session_state.get("recording", False):
            mp3_file = recorder.stop_recording()
            st.session_state["recording"] = False
            if mp3_file:
                # Transcribe the recorded MP3
                try:
                    log_debug("Transcribing audio...")
                    transcript_response = transcribe_audio_v3(mp3_file)
                    # Assuming transcribe_audio_v3 returns a dict with 'text' key
                    transcript_text = transcript_response.get('text', 'Transcription failed.')
                    if not transcript_text:
                        st.error("Transcription failed: Empty response.")
                        log_debug("Audio transcription failed.")
                    else:
                        st.session_state["transcription"] = transcript_text
                        st.success(f"Transcription: {transcript_text}")
                        log_debug("Audio transcription successful.")
                except Exception as e:
                    st.error(f"Transcription exception: {str(e)}")
                    log_debug(f"Transcription exception: {str(e)}")
        else:
            st.warning("No recording in progress to stop.")

# Handle different pages
if selection == "Chat with Amanda":
    # Title and introductory text
    st.title("🤖 Chat with Amanda (RAG Enhanced)")
    st.write("Amanda is now enhanced with Retrieval-Augmented Generation (RAG) for better, more accurate responses!")

    # Check if vector store is loaded
    if not st.session_state.get('vector_store_loaded', False):
        # Attempt to load the vector store
        try:
            log_debug("Loading vector store...")
            vectorstore, chunks = load_vector_store()
            log_debug("Vector store loaded successfully.")
            qa_chain = create_rag_chain(vectorstore, used_model)
            st.session_state['qa_chain'] = qa_chain  # Store qa_chain in session state
            st.session_state['vector_store_loaded'] = True
            st.session_state['chunks'] = chunks  # Store chunks in session state
        except FileNotFoundError as e:
            st.error(str(e))
            log_debug(str(e))
            st.warning("Please click the '🔄 Refresh Vector Store' button in the sidebar to create the vector store.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            log_debug(f"Error loading vector store: {e}")
            st.stop()

    # Update chat title after the first user input
    if st.session_state["all_chats"][st.session_state["active_chat"]]:
        first_message = st.session_state["all_chats"][st.session_state["active_chat"]][0].get("content", "")
        if first_message and st.session_state["chat_titles"][st.session_state["active_chat"]].startswith("Chat"):
            try:
                new_chat_title = summarizer(first_message, max_length=6, min_length=2, do_sample=False)[0]['summary_text']
                st.session_state["chat_titles"][st.session_state["active_chat"]] = new_chat_title
                log_debug(f"Chat title updated to: {new_chat_title}")
            except Exception as e:
                log_debug(f"Summarization failed: {e}")

    # Set up the chat messages for the active chat
    active_messages = st.session_state["all_chats"][st.session_state["active_chat"]]

    # Use the transcription as user input
    user_input = st.session_state.get("transcription", "") or st.chat_input("Type your message here...")

    # Process user input and query RAG chain
    if user_input:
        active_messages.append({"role": "user", "content": user_input})
        st.session_state["all_chats"][st.session_state["active_chat"]] = active_messages
        log_debug(f"User input: {user_input}")

        with st.spinner("Amanda is thinking..."):
            try:
                qa_chain = st.session_state.get('qa_chain', None)
                if not qa_chain:
                    st.error("RAG Chain is not available. Please refresh the vector store.")
                    log_debug("RAG Chain is not available.")
                else:
                    log_debug("Querying RAG chain...")
                    result = qa_chain({"query": user_input})
                    if "result" in result and result["result"]:
                        amanda_message = result["result"].strip()
                        log_debug(f"Amanda response: {amanda_message}")

                        # Extract source information from the result
                        source_documents = result.get("source_documents", [])
                        if source_documents:
                            source = source_documents[0].metadata.get("source", "Unknown")
                            page = source_documents[0].metadata.get("page", "N/A")
                            source_info = f"Source: {source}, Page: {page}"
                            log_debug(f"Source information: {source_info}")
                        else:
                            source_info = "Source: Not available"
                            log_debug("No source information available.")

                        # Calculate cost based on token usage
                        num_tokens = count_tokens(user_input) + count_tokens(amanda_message)
                        if used_model == "gpt-4-turbo":
                            cost_per_1k_tokens = 0.012
                        elif used_model == "gpt-4":
                            cost_per_1k_tokens = 0.03
                        elif used_model == "gpt-3.5-turbo":
                            cost_per_1k_tokens = 0.002
                        else:
                            raise ValueError(f"Unsupported model: {used_model}")
                        cost = (num_tokens / 1000) * cost_per_1k_tokens
                        log_debug(f"Cost of operation: ${cost:.6f}")

                        # Append the response along with cost and source info
                        active_messages.append({
                            "role": "assistant",
                            "content": amanda_message,
                            "tokens": num_tokens,
                            "cost": cost,
                            "source_info": source_info
                        })
                        st.session_state["all_chats"][st.session_state["active_chat"]] = active_messages
            except Exception as e:
                st.error(f"Error: {str(e)}")
                log_debug(f"Error during RAG chain query: {str(e)}")

    # Display conversation history for the active chat
    st.markdown("---")
    for idx, message in enumerate(active_messages):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Amanda 🤖:** {message['content']}")

            # Display the cost of tokens
            if "cost" in message:
                token_count = message.get("tokens", 0)
                cost = message.get("cost", 0.0)
                st.markdown(f"<p style='color: grey;'>Tokens used: <i>{token_count}</i>, Cost: <i>${cost:.6f}</i></p>", unsafe_allow_html=True)

            # Display the source information
            if "source_info" in message:
                st.markdown(f"<p style='color: grey;'>{message['source_info']}</p>", unsafe_allow_html=True)

            # Align Play, Like, Dislike, Re-generate, and Copy buttons
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

            with col1:
                play_button = st.button("🔊", key=f"play_{idx}")
                if play_button:
                    play_audio(message['content'], file_name=f"amanda_reply_{idx}")

            with col2:
                like_button = st.button("👍", key=f"like_{idx}")
                if like_button:
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Liked")
                    else:
                        st.session_state["feedback"][idx] = "Liked"
                    st.success(f"Feedback for message {idx} set to: Liked")

            with col3:
                dislike_button = st.button("👎", key=f"dislike_{idx}")
                if dislike_button:
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Disliked")
                    else:
                        st.session_state["feedback"][idx] = "Disliked"
                    st.success(f"Feedback for message {idx} set to: Disliked")

            with col4:
                regenerate_button = st.button("🔄", key=f"regenerate_{idx}")
                if regenerate_button:
                    try:
                        log_debug("Regenerating response...")
                        # To regenerate, get the last user message
                        user_messages = [msg for msg in active_messages if msg["role"] == "user"]
                        if user_messages:
                            user_query = user_messages[-1]["content"]
                            result = st.session_state['qa_chain']({"query": user_query})
                            amanda_message = result["result"].strip()
                            # Replace the last assistant message
                            if len(active_messages) >= 1 and active_messages[-1]["role"] == "assistant":
                                active_messages[-1]["content"] = amanda_message
                                active_messages[-1]["tokens"] = count_tokens(amanda_message)
                                # Optionally update cost and source_info
                            else:
                                active_messages.append({
                                    "role": "assistant",
                                    "content": amanda_message
                                })
                            # Recalculate cost
                            num_tokens = count_tokens(user_query) + count_tokens(amanda_message)
                            if used_model == "gpt-4-turbo":
                                cost_per_1k_tokens = 0.012
                            elif used_model == "gpt-4":
                                cost_per_1k_tokens = 0.03
                            elif used_model == "gpt-3.5-turbo":
                                cost_per_1k_tokens = 0.002
                            else:
                                raise ValueError(f"Unsupported model: {used_model}")
                            cost = (num_tokens / 1000) * cost_per_1k_tokens
                            active_messages[-1]["cost"] = cost
                            active_messages[-1]["tokens"] = num_tokens
                            st.session_state["all_chats"][st.session_state["active_chat"]] = active_messages
                            log_debug("Regeneration successful.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        log_debug(f"Error during regeneration: {str(e)}")

            with col5:
                copy_button = st.button("📋", key=f"copy_{idx}")
                if copy_button:
                    pyperclip.copy(message['content'])
                    st.success("Copied to clipboard!")

elif selection == "Debugging Logs":
    st.title("Debugging Logs")
    if st.session_state.get("debug_log"):
        for log_message in st.session_state["debug_log"]:
            st.text(log_message)
    else:
        st.write("No debug logs available.")

elif selection == "User Feedback":
    st.title("User Feedback Summary")
    for i, feedback in enumerate(st.session_state["feedback"]):
        if feedback:
            st.markdown(f"Message {i}: {feedback}")

elif selection == "All Chunks":
    st.title("Document Chunks")
    if st.session_state.get("chunks"):
        log_debug(f"Total number of chunks: {len(st.session_state['chunks'])}")
        for idx, chunk in enumerate(st.session_state["chunks"]):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "N/A")
            chunk_type = chunk.metadata.get("type", "text")
            st.markdown(f"**Chunk {idx + 1} from {source}, Page: {page}, Type: {chunk_type}:**")
            st.text(chunk.page_content)
    else:
        st.write("No chunks available. Please load documents.")
        log_debug("No chunks available.")

elif selection == "Web Crawl":
    st.title("🌐 Web Crawler")
    st.write("Enter a URL and specify the depth to crawl the website.")

    # Input fields for URL and Depth
    with st.form(key='web_crawl_form'):
        start_url = st.text_input("Starting URL", value="https://www.example.com")
        max_depth = st.number_input("Crawl Depth", min_value=1, max_value=5, value=2, step=1)
        submit_button = st.form_submit_button(label='Start Crawling')

    if submit_button:
        if not start_url:
            st.error("Please enter a valid URL.")
        elif not WebCrawler(None, 1).is_valid_url(start_url):
            st.error("Please enter a well-formed URL (e.g., https://www.example.com).")
        else:
            # Initialize progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Function to run the crawl and save data
            def run_crawl():
                try:
                    crawler = WebCrawler(start_url=start_url, max_depth=int(max_depth), delay=1)
                    crawler.crawl()
                    st.success(f"Web crawling completed up to depth {max_depth}.")
                    log_debug(f"Web crawling completed for {start_url} up to depth {max_depth}.")
                except Exception as e:
                    st.error(f"An error occurred during crawling: {str(e)}")
                    log_debug(f"Web crawling error: {str(e)}")

            # Run the crawler in a separate thread to prevent blocking
            thread = threading.Thread(target=run_crawl)
            thread.start()

    # Display Crawled Data
    st.markdown("---")
    st.header("Crawled Data")

    if os.path.exists("crawled_data.json"):
        with open("crawled_data.json", "r", encoding='utf-8') as f:
            crawled_data = json.load(f)

        if crawled_data:
            for entry in crawled_data:
                st.subheader(entry["source_url"])
                st.write(entry["content"][:500] + '...')  # Display first 500 characters
                with st.expander("View Full Content"):
                    st.write(entry["content"])
        else:
            st.info("No data available. Start crawling to collect data.")
    else:
        st.info("No data available. Start crawling to collect data.")

    # Provide Option to Download the Crawled Data
    if os.path.exists("crawled_data.json"):
        with open("crawled_data.json", "r", encoding='utf-8') as f:
            crawled_data = f.read()
        st.download_button(
            label="Download Crawled Data as JSON",
            data=crawled_data,
            file_name='crawled_data.json',
            mime='application/json'
        )

# Footer
st.markdown("---")
