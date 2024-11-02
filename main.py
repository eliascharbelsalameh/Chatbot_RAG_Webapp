import streamlit as st
import openai
import os
import pyperclip # type: ignore
from deepgram import Deepgram # type: ignore
import tempfile
import sounddevice as sd # type: ignore
from transformers import pipeline # type: ignore

from audio_utils import clear_audio_files, play_audio, record_audio, transcribe_audio_v3
from document_processing import read_files_from_directory
from faiss_utils import load_vector_store, recreate_vector_store, faiss_index_exists
from rag_chain import create_rag_chain
from session_utils import initialize_session_state, log_debug

# TODO: add summarization to the title
# TODO: add deepgram instead of gTTS, whisper, webs-crapping
# TODO: relocate vectorstore FAISS to GIN515-Deep Learning
# TODO: add groq-whisper3
# TODO: enhance tables
# TODO: load file online
# TODO: deepgram API for TTS
# TODO: embedding text 3

used_model = "gpt-4-turbo"
chunk_size = 750
chunk_overlap = 75
temperature = 0.01

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Deepgram API client
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # Make sure you have set your API key in the environment variable
deepgram_client = Deepgram(DEEPGRAM_API_KEY)

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Clear any existing audio files
clear_audio_files()

# Define pages for the Streamlit application
PAGES = ["Chat with Amanda", "Debugging Logs", "User Feedback", "All Chunks"]

# Set up Streamlit sidebar
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", PAGES)

# Initialize session state
initialize_session_state()

# Amanda Page: Main chat interface
if selection == "Chat with Amanda":
    # Title and introductory text
    st.title("🤖 Chat with Amanda (RAG Enhanced)")
    st.write("Amanda is now enhanced with Retrieval-Augmented Generation (RAG) for better, more accurate responses!")

    # Function to load the vector store, recreate if necessary
    try:
        log_debug("Loading vector store...")
        vectorstore = load_vector_store()
        log_debug("Vector store loaded successfully.")
        qa_chain = create_rag_chain(vectorstore, used_model)
    except ValueError as e:
        st.error(f"Error loading vector store: {e}")
        log_debug(f"Error loading vector store: {e}")
        st.stop()

    # New Chat button
    if st.sidebar.button("New Chat"):
        st.session_state["all_chats"].append([])  # Create a new chat session
        st.session_state["active_chat"] = len(st.session_state["all_chats"]) - 1  # Set this as the active chat
        st.session_state["chat_titles"] = st.session_state.get("chat_titles", [])
        st.session_state["chat_titles"].append(f"Chat {len(st.session_state['all_chats'])}")

    # Update chat title after the first user input
    if st.session_state["all_chats"][st.session_state["active_chat"]]:
        first_message = st.session_state["all_chats"][st.session_state["active_chat"]][0].get("content", "")
        if first_message and st.session_state["chat_titles"][st.session_state["active_chat"]].startswith("Chat"):
            new_chat_title = summarizer(first_message, max_length=4, min_length=2, do_sample=False)[0]['summary_text']
            st.session_state["chat_titles"][st.session_state["active_chat"]] = new_chat_title

    # Display chat list as a fixed list in the sidebar with an outline for the active chat
    st.sidebar.title("Chats")
    for i, title in enumerate(st.session_state.get("chat_titles", [])):
        if i == st.session_state["active_chat"]:
            st.sidebar.markdown(f"<div style='border: 2px solid #4CAF50; padding: 5px;'>{title}</div>", unsafe_allow_html=True)
        else:
            if st.sidebar.button(title, key=f"chat_{i}"):
                st.session_state["active_chat"] = i  # Set the selected chat as active

    # Set up the chat messages for the active chat
    active_messages = st.session_state["all_chats"][st.session_state["active_chat"]]

    # Sidebar: Record Audio with Deepgram
    st.sidebar.title("Audio Recorder")
    st.sidebar.write("Record and transcribe audio with Deepgram.")

    if st.sidebar.button("Record Audio"):
        log_debug("Recording audio...")
        audio_file = record_audio()
        st.sidebar.success(f"Audio recorded: {audio_file}")
        log_debug("Audio recorded successfully.")

        # Transcribe the recorded audio
        try:
            log_debug("Transcribing audio...")
            transcription = transcribe_audio_v3(audio_file)
            if transcription:
                transcript_text = transcription.get("results", {}).get("channels", [])[0].get("alternatives", [])[0].get("transcript", "")
                if transcript_text:
                    st.session_state["transcription"] = transcript_text
                    st.write(f"Transcription: {transcript_text}")
                    log_debug("Audio transcription successful.")
                else:
                    st.write("No transcription received.")
                    log_debug("No transcription received.")
            else:
                st.write("Transcription failed.")
                log_debug("Transcription failed.")
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            log_debug(f"Transcription error: {str(e)}")

    user_input = st.session_state.get("transcription") or st.chat_input("Type your message here...")

    # Process user input and query RAG chain
    if user_input:
        active_messages.append({"role": "user", "content": user_input})
        st.session_state["all_chats"][st.session_state["active_chat"]] = active_messages
        log_debug(f"User input: {user_input}")

        with st.spinner("Amanda is thinking..."):
            try:
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
                    num_tokens = len(user_input) + len(amanda_message)
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
            col1, col2, col3, col4, col5 = st.columns([0.1, 0.1, 0.1, 0.1, 0.1])

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
                        result = qa_chain({"query": active_messages[-2]["content"]})
                        amanda_message = result["result"].strip()
                        active_messages[-1]["content"] = amanda_message
                        st.session_state["all_chats"][st.session_state["active_chat"]] = active_messages  # Update after regeneration
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        log_debug(f"Error during regeneration: {str(e)}")

            with col5:
                copy_button = st.button("📋", key=f"copy_{idx}")
                if copy_button:
                    pyperclip.copy(message['content'])
                    st.success("Copied to clipboard!")

# Debugging Page: Display log messages
elif selection == "Debugging Logs":
    st.title("Debugging Logs")
    for log_message in st.session_state["debug_log"]:
        st.text(log_message)

# User Feedback Page: Display all feedback collected so far
elif selection == "User Feedback":
    st.title("User Feedback Summary")
    for i, feedback in enumerate(st.session_state["feedback"]):
        if feedback:
            st.markdown(f"Message {i}: {feedback}")

# Chunks Page: Display chunk settings and all available chunks
elif selection == "All Chunks":
    st.title("Document Chunks")
    st.write(f"**Chunk Size:** {chunk_size}, **Chunk Overlap:** {chunk_overlap}")
    if st.session_state["chunks"]:
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

# Footer
st.markdown("---")
