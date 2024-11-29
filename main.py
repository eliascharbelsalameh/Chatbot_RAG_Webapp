# main.py

import streamlit as st
import os
import pyperclip  # type: ignore
from deepgram import Deepgram  # type: ignore
import json
import multiprocessing  # To handle multiprocessing for Scrapy
import scrapy  # type: ignore
from urllib.parse import urlparse
import queue  # Added import for thread-safe queues
import time  # For sleep in progress updates
import io

from typing import Optional, List, Mapping, Any

from scrapy.crawler import CrawlerProcess  # type: ignore
from scrapy.utils.project import get_project_settings  # type: ignore
from document_processing import read_files_from_directory
from faiss_utils import load_vector_store, refresh_vector_store
from session_utils import initialize_session_state, log_debug, initialize_logging
from data_processing import load_crawled_data, split_into_chunks

# Import st_audiorec for audio recording
from st_audiorec import st_audiorec  # type: ignore

from langchain.llms import LlamaCpp  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.chains import RetrievalQA  # type: ignore

# Import for offline text-to-speech
import pyttsx3  # type: ignore
import tempfile

# Import BeautifulSoup for HTML parsing
from bs4 import BeautifulSoup  # type: ignore

# Path to your GGUF model
model_path = "C:/Users/elias/.cache/lm-studio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf"

# Initialize the LlamaCpp LLM
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=128000,    # Adjust to the model's maximum context size
    n_threads=8    # Adjust based on your CPU
)

# Initialize Streamlit app
st.set_page_config(page_title="Amanda Chatbot", layout="wide")

# Initialize session state
initialize_session_state()
initialize_logging()  # Initialize logging queue

# Ensure 'feedback' and 'debug_log' exist and match the number of chats
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = []

if 'debug_log' not in st.session_state:
    st.session_state['debug_log'] = []

# Synchronize the length of 'feedback' with 'all_chats'
while len(st.session_state['feedback']) < len(st.session_state['all_chats']):
    st.session_state['feedback'].append(None)
while len(st.session_state['feedback']) > len(st.session_state['all_chats']):
    st.session_state['feedback'].pop()

# Initialize the summarization pipeline with a specified model to avoid warnings
from transformers import pipeline   # type: ignore
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer_kwargs={"clean_up_tokenization_spaces": True}  # Set explicitly
)

# Define pages for the Streamlit application
PAGES = ["Chat with Amanda", "Debugging Logs", "User Feedback", "All Chunks", "Web Crawl"]

# Define the Scrapy Spider within main.py
class AmandaSpider(scrapy.Spider):
    name = "amanda_spider"

    custom_settings = {
        'FEED_EXPORT_ENCODING': 'utf-8',
        'LOG_LEVEL': 'ERROR',  # Reduce Scrapy logs
        'DEPTH_LIMIT': 3,  # Control the depth of crawling
        'CLOSESPIDER_PAGECOUNT': 100,  # Stop after crawling 100 pages
    }

    def __init__(self, start_url, max_depth=2, max_pages=100, log_queue=None, *args, **kwargs):
        super(AmandaSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]
        self.max_pages = max_pages
        self.page_count = 0
        self.crawled_data = []
        self.log_queue = log_queue

    def parse(self, response):
        if self.page_count >= self.max_pages:
            return

        self.page_count += 1

        # Only proceed if response is TextResponse (i.e., text content)
        if not isinstance(response, scrapy.http.TextResponse):
            self.log_debug(f"Skipped non-text content: {response.url}")
            return

        # Use BeautifulSoup to parse the response body
        soup = BeautifulSoup(response.text, 'lxml')

        # Remove script and style elements
        for script in soup(['script', 'style', 'header', 'footer', 'nav', 'noscript']):
            script.extract()  # Remove these elements

        # Get text
        text_content = soup.get_text(separator=' ')

        # Collapse whitespace
        text_content = ' '.join(text_content.split())

        if text_content.strip():
            self.crawled_data.append({
                "source_url": response.url,
                "content": text_content
            })
            self.log_debug(f"Crawled {self.page_count}: {response.url}")
        else:
            self.log_debug(f"No content extracted from: {response.url}")

        if self.page_count >= self.max_pages:
            return

        # Extract links and follow them
        for href in response.css('a::attr(href)').getall():
            next_url = response.urljoin(href)
            if self._is_valid_url(next_url) and self._is_allowed_domain(next_url):
                yield scrapy.Request(next_url, callback=self.parse)

    @staticmethod
    def _is_valid_url(url):
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https')

    def _is_allowed_domain(self, url):
        domain = urlparse(url).netloc
        return domain == self.allowed_domains[0]

    def closed(self, reason):
        # Save the crawled data
        with open("crawled_data.json", "w", encoding='utf-8') as f:
            json.dump(self.crawled_data, f, ensure_ascii=False, indent=4)
        self.log_debug(f"Spider closed: {reason}")

    def log_debug(self, message):
        if self.log_queue:
            self.log_queue.put(message)

# Function to transcribe audio using Deepgram
def transcribe_audio(audio_bytes):
    """
    Transcribe an audio bytes object using Deepgram API.

    Args:
        audio_bytes: Bytes of the recorded audio.

    Returns:
        dict: A dictionary containing the transcription result.
    """
    import asyncio

    async def _transcribe(audio_data):
        source = {'buffer': audio_data, 'mimetype': 'audio/wav'}
        response = await deepgram_client.transcription.prerecorded(source, {'punctuate': True})
        return response

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(_transcribe(audio_bytes))
    return response

# Function to run Scrapy spider in a separate process
def run_spider_process(start_url, max_depth, max_pages, log_queue):
    try:
        process = CrawlerProcess(get_project_settings())
        process.crawl(
            AmandaSpider,
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            log_queue=log_queue
        )
        process.start()
    except Exception as e:
        if log_queue:
            log_queue.put(f"Scrapy crawling error: {e}")

# Function to create RAG chain with custom prompt template
def create_rag_chain_with_prompt(vectorstore, llm):
    # Define the prompt template
    prompt_template = """You are Amanda, an AI assistant.

Answer the following question to the best of your ability using the provided context.
If you don't know the answer, just say "I don't know" without making anything up.
Keep your answer concise.

Context:
{context}

Question:
{question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain with the custom prompt and return source documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Set up Streamlit sidebar
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", PAGES)

# Add "Refresh Vector Store" button in the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Vector Store"):
    with st.spinner("Refreshing the vector store..."):
        try:
            vector_store, chunks = refresh_vector_store()
            qa_chain = create_rag_chain_with_prompt(vector_store, llm)
            st.session_state['qa_chain'] = qa_chain  # Store qa_chain in session state
            st.session_state['vector_store_loaded'] = True  # Indicate that vector store is loaded
            st.session_state['chunks'] = chunks  # Store chunks in session state
            st.success("Vector store refreshed successfully.")
            log_debug("Vector store refreshed successfully.")
        except Exception as e:
            st.error(f"Failed to refresh vector store: {e}")
            log_debug(f"Failed to refresh vector store: {e}")

# Add "Chats" section with "New Chat" and "Delete Chat" buttons
st.sidebar.markdown("---")
st.sidebar.markdown("## Chats")

# Button to add a new chat
if st.sidebar.button("➕ New Chat"):
    st.session_state["all_chats"].append([])  # Create a new chat session
    st.session_state["active_chat"] = len(st.session_state["all_chats"]) - 1  # Set this as the active chat
    st.session_state["chat_titles"] = st.session_state.get("chat_titles", [])
    st.session_state["chat_titles"].append(f"Chat {len(st.session_state['all_chats'])}")
    st.session_state["feedback"].append(None)  # Initialize feedback for the new chat
    log_debug(f"New chat created: Chat {len(st.session_state['all_chats'])}")

# Display chat list with Delete buttons
for i, title in enumerate(st.session_state.get("chat_titles", [])):
    # Create two columns: one for the title and one for the delete button
    chat_col1, chat_col2 = st.sidebar.columns([4, 1])  # Adjusted ratios for better alignment
    with chat_col1:
        if i == st.session_state["active_chat"]:
            st.markdown(
                f"<div style='border: 2px solid #4CAF50; padding: 5px; border-radius:5px;'>{title}</div>",
                unsafe_allow_html=True
            )
        else:
            if st.sidebar.button(title, key=f"select_chat_{i}"):
                st.session_state["active_chat"] = i  # Set the selected chat as active
                log_debug(f"Switched to chat: {title}")
    with chat_col2:
        if len(st.session_state["all_chats"]) > 1:
            if st.sidebar.button("🗑️", key=f"delete_chat_{i}"):
                # Delete the chat
                del st.session_state["all_chats"][i]
                del st.session_state["chat_titles"][i]
                del st.session_state["feedback"][i]
                log_debug(f"Deleted chat index: {i}")
                # Adjust active_chat index
                if st.session_state["active_chat"] >= len(st.session_state["all_chats"]):
                    st.session_state["active_chat"] = len(st.session_state["all_chats"]) - 1
                st.success(f"Chat {i + 1} deleted.")

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
            qa_chain = create_rag_chain_with_prompt(vectorstore, llm)
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
        user_messages = [msg['content'] for msg in st.session_state["all_chats"][st.session_state["active_chat"]] if msg['role'] == 'user']
        if user_messages and st.session_state["chat_titles"][st.session_state["active_chat"]].startswith("Chat"):
            first_messages_text = ' '.join(user_messages[:3])  # Combine first 3 user messages
            try:
                new_chat_title = summarizer(first_messages_text, max_length=15, min_length=5, do_sample=False)[0]['summary_text']
                st.session_state["chat_titles"][st.session_state["active_chat"]] = new_chat_title.strip()
                log_debug(f"Chat title updated to: {new_chat_title}")
            except Exception as e:
                log_debug(f"Summarization failed: {e}")

    # Set up the chat messages for the active chat
    active_messages = st.session_state["all_chats"][st.session_state["active_chat"]]

    # Add audio recorder using st_audiorec
    st.markdown("## Record a message to chat with Amanda")
    audio_bytes = st_audiorec()

    if audio_bytes is not None:
        DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        if DEEPGRAM_API_KEY:
            deepgram_client = Deepgram(DEEPGRAM_API_KEY)
            # Transcribe the recorded audio
            try:
                log_debug("Transcribing recorded audio...")
                transcript_response = transcribe_audio(audio_bytes)
                transcript_text = transcript_response['results']['channels'][0]['alternatives'][0]['transcript']
                if not transcript_text:
                    st.error("Transcription failed: Empty response.")
                    log_debug("Audio transcription failed.")
                else:
                    st.session_state["transcription"] = transcript_text
                    st.success(f"Transcription: {transcript_text}")
                    log_debug("Audio transcription successful.")
                    st.experimental_rerun()  # Rerun to process the transcription
            except Exception as e:
                st.error(f"Transcription exception: {str(e)}")
                log_debug(f"Transcription exception: {str(e)}")
        else:
            st.error("Deepgram API key not found. Audio transcription will not work.")

    # Always display the chat input box
    user_input = st.chat_input("Type your message here...")

    # If there is a transcription, use it as the user input
    if st.session_state.get("transcription", ""):
        user_input = st.session_state.get("transcription", "")
        # Clear the transcription after use
        st.session_state["transcription"] = ""

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
                    # Send the user's question to the chain
                    result = qa_chain({"query": user_input})

                    if result and "result" in result and result["result"]:
                        amanda_message = result["result"].strip()
                        log_debug(f"Amanda response: {amanda_message}")

                        # Append the response
                        active_messages.append({
                            "role": "assistant",
                            "content": amanda_message,
                            "source_documents": result.get("source_documents", [])
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

            # Display source documents if available
            if "source_documents" in message:
                source_lines = []
                for doc in message["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    source_lines.append(f"Source: {source}, Page: {page}")
                source_text = "  \n".join(source_lines)  # Markdown line breaks
                st.caption(source_text)  # Display in light gray font

            # Align Like, Dislike, Re-generate, Copy, and Play Audio buttons
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

            with col1:
                like_button = st.button("👍", key=f"like_{idx}")
                if like_button:
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Liked")
                    else:
                        st.session_state["feedback"][idx] = "Liked"
                    st.success(f"Feedback for message {idx} set to: Liked")

            with col2:
                dislike_button = st.button("👎", key=f"dislike_{idx}")
                if dislike_button:
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Disliked")
                    else:
                        st.session_state["feedback"][idx] = "Disliked"
                    st.success(f"Feedback for message {idx} set to: Disliked")

            with col3:
                regenerate_button = st.button("🔄", key=f"regenerate_{idx}")
                if regenerate_button:
                    try:
                        log_debug("Regenerating response...")
                        # Re-query the chain with the same user input
                        # Find the corresponding user message
                        user_message = None
                        for i in range(idx - 1, -1, -1):
                            if active_messages[i]["role"] == "user":
                                user_message = active_messages[i]["content"]
                                break
                        if user_message:
                            result = qa_chain({"query": user_message})
                            if "result" in result and result["result"]:
                                amanda_message = result["result"].strip()
                                log_debug(f"Regenerated Amanda response: {amanda_message}")

                                # Update the assistant message
                                active_messages[idx]["content"] = amanda_message
                                active_messages[idx]["source_documents"] = result.get("source_documents", [])

                                st.session_state["all_chats"][st.session_state["active_chat"]] = active_messages
                                log_debug("Regeneration successful.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        log_debug(f"Error during regeneration: {str(e)}")

            with col4:
                copy_button = st.button("📋", key=f"copy_{idx}")
                if copy_button:
                    pyperclip.copy(message['content'])
                    st.success("Copied to clipboard!")

            with col5:
                play_audio_button = st.button("🔊", key=f"play_audio_{idx}")
                if play_audio_button:
                    try:
                        # Initialize pyttsx3 engine
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)  # Adjust speech rate if needed

                        # Save audio to a temporary file
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        tmp_file_name = tmp_file.name
                        tmp_file.close()

                        engine.save_to_file(message['content'], tmp_file_name)
                        engine.runAndWait()

                        # Read the audio data
                        with open(tmp_file_name, 'rb') as f:
                            audio_data = f.read()
                        os.unlink(tmp_file_name)

                        # Play audio in Streamlit
                        st.audio(audio_data, format='audio/wav')
                        log_debug(f"Played audio for message {idx}")
                    except Exception as e:
                        st.error(f"Error playing audio: {str(e)}")
                        log_debug(f"Error playing audio: {str(e)}")

elif selection == "Debugging Logs":
    st.title("Debugging Logs")
    log_container = st.empty()

    # Continuously retrieve logs from the queue and display them
    while not st.session_state['log_queue'].empty():
        try:
            log_message = st.session_state['log_queue'].get_nowait()
            st.session_state['debug_log'].append(log_message)
        except queue.Empty:
            break

    if st.session_state.get("debug_log"):
        for log_message in st.session_state["debug_log"]:
            st.text(log_message)
    else:
        st.write("No debug logs available.")

elif selection == "User Feedback":
    st.title("User Feedback Summary")
    for i, feedback in enumerate(st.session_state.get("feedback", [])):
        if feedback:
            st.markdown(f"**Message {i + 1}:** {feedback}")

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
    st.write("Enter a URL, specify the depth, and maximum number of pages to crawl the website.")

    # Input fields for URL, Depth, and Max Pages
    with st.form(key='web_crawl_form'):
        start_url = st.text_input("Starting URL", value="https://www.example.com")
        max_depth = st.number_input("Crawl Depth", min_value=1, max_value=5, value=2, step=1)
        max_pages = st.number_input("Max Pages", min_value=10, max_value=1000, value=100, step=10)
        submit_button = st.form_submit_button(label='Start Crawling')

    if submit_button:
        if not start_url:
            st.error("Please enter a valid URL.")
        elif not AmandaSpider._is_valid_url(start_url):
            st.error("Please enter a well-formed URL (e.g., https://www.example.com).")
        else:
            # Initialize progress indicators
            progress_bar = st.progress(0.0)  # Initialize with 0.0
            with st.spinner("Crawling in progress..."):
                # Create a multiprocessing.Queue for logs
                crawl_log_queue = multiprocessing.Queue()

                # Start the crawler process
                crawler_process = multiprocessing.Process(
                    target=run_spider_process,
                    args=(start_url, int(max_depth), int(max_pages), crawl_log_queue)
                )
                crawler_process.start()

                # Monitor the crawler process and update logs and progress
                while crawler_process.is_alive():
                    try:
                        # Retrieve log messages
                        while not crawl_log_queue.empty():
                            log_message = crawl_log_queue.get_nowait()
                            st.session_state['debug_log'].append(log_message)
                            # Update progress if relevant
                            if "Crawled" in log_message:
                                parts = log_message.split(":")
                                if len(parts) >= 2 and parts[0].startswith("Crawled"):
                                    try:
                                        crawled_num = int(parts[0].split()[1])
                                        progress_fraction = crawled_num / max_pages
                                        progress_fraction = min(progress_fraction, 1.0)  # Ensure it doesn't exceed 1.0
                                        progress_bar.progress(progress_fraction)
                                    except ValueError:
                                        # Handle cases where the crawled number isn't an integer
                                        pass
                    except Exception as e:
                        st.error(f"Error retrieving logs: {e}")
                        log_debug(f"Error retrieving logs: {e}")
                    time.sleep(0.5)  # Adjust the sleep time as needed

                crawler_process.join()

                # Final log message
                st.success(f"Web crawling completed up to depth {max_depth} and {max_pages} pages.")
                log_debug(f"Web crawling completed for {start_url} up to depth {max_depth} and {max_pages} pages.")

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
st.write("© 2024 Amanda Chatbot. All rights reserved.")