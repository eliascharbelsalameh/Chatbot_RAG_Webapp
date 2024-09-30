import streamlit as st
import openai
import os
import pyperclip  # for copy to clipboard functionality         # type: ignore
from langchain import PromptTemplate                            # type: ignore
from langchain.chains import RetrievalQA                        # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings          # type: ignore
from langchain.vectorstores import FAISS                        # type: ignore
from langchain.chat_models import ChatOpenAI                    # type: ignore
from langchain.text_splitter import CharacterTextSplitter       # type: ignore
from langchain.document_loaders import PyPDFLoader              # type: ignore
import tiktoken  # For token counting                           # type: ignore
import os.path
import time

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# CSS for layout and dark background
st.markdown(
    """
    <style>
    .reportview-container, .css-1outpf7 {
        background-color: #2c003e;  /* Midnight Purple */
        color: white;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown p {
        color: white;
    }
    .stTextInput>div>div {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        color: black;
    }
    .stMarkdown {
        color: white;
    }
    .chat-bubble-user {
        background-color: #3a3b3c;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        max-width: 80%;
    }
    .chat-bubble-assistant {
        background-color: #4c1f73;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        max-width: 80%;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and introductory text
st.title("🤖 Chat with Amanda (RAG Enhanced)")
st.write("Amanda is now enhanced with Retrieval-Augmented Generation (RAG) for better, more accurate responses!")

# Load Documents (PDFs or other formats)
def read_pdfs_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

# Define the directory where your PDFs are stored
pdf_directory = "C:/Users/elias/OneDrive/Bureau/USEK-Electrical and Electronics Engineer/Semesters/Term-9_Fall-202510/GIN515-Deep Learning/CVs_PDFs"

# Check if the FAISS index exists
def faiss_index_exists():
    return os.path.exists('vectorstore/db_faiss/index.faiss')

# Setup the vector store with the PDF directory
@st.cache_resource
def setup_vector_store(pdf_directory):
    if not faiss_index_exists():
        # Only create the vector store if it doesn't exist
        documents = read_pdfs_from_directory(pdf_directory)
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)

        # Create embeddings using Hugging Face model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/Paraphrase_multilingual_mpnet_base_v2")
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/All_mpnet_base_v2")
        
        # Create and store vector embeddings
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local('vectorstore/db_faiss')

        return vectorstore
    else:
        # If the vector store already exists, load it
        return load_vector_store()

# Load the vector store
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/Paraphrase_multilingual_mpnet_base_v2")
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/All_mpnet_base_v2")
    vectorstore = FAISS.load_local('vectorstore/db_faiss', embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Function to calculate token usage with error handling
def num_tokens_from_messages(messages, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message starts with 4 tokens
            for key, value in message.items():
                if isinstance(value, str):  # Ensure value is a string before encoding
                    num_tokens += len(encoding.encode(value))
            if message.get("role") == "user":
                num_tokens += 1  # additional token for the role 'user'
        num_tokens += 2  # every reply is primed with 2 tokens
        return num_tokens
    except Exception as e:
        st.error(f"Error in token calculation: {e}")
        return 0

# Function to calculate cost
def calculate_cost(num_tokens, model):
    # Pricing as per OpenAI's API rates (change according to the latest rates)
    if model == "gpt-3.5-turbo":
        cost_per_1k_tokens = 0.002  # cost per 1k tokens for gpt-3.5-turbo
    elif model == "gpt-4":
        cost_per_1k_tokens = 0.03  # adjust as needed
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    cost = (num_tokens / 1000) * cost_per_1k_tokens
    return cost

# Setup the QA Retrieval Chain
def create_rag_chain():
    vectorstore = setup_vector_store(pdf_directory)  # Ensure we create/load vector store

    # Define the custom prompt template
    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}

    Question: {question}
    Answer: """
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), # gpt-3.5-turbo / gpt-4
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Initialize the QA chain (can take time, so it's cached)
qa_chain = create_rag_chain()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are Amanda, a helpful assistant."}
    ]
if "feedback" not in st.session_state:
    st.session_state["feedback"] = []  # Store feedback for each response

# Display conversation history in a scrollable container
chat_container = st.container()
with chat_container:
    st.markdown("---")
    for idx, message in enumerate(st.session_state["messages"]):
        if message["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"<div class='chat-bubble-assistant'><b>Amanda 🤖:</b> {message['content']}</div>", unsafe_allow_html=True)

            # Display the most relevant source information if available
            if "source" in message:
                source_info = message["source"]
                st.markdown(f"""<p style='color: grey;'>Source: File: <i>{source_info['filename']}</i>, Page: <i>{source_info['page']}</i></p>""",
                            unsafe_allow_html=True)
                st.markdown(f"""<p style='color: grey;'>Tokens used: <i>{source_info['tokens']}</i>, Cost: <i>${source_info['cost']:.6f}</i></p>""",
                            unsafe_allow_html=True)

            # Like, Dislike, Re-generate, and Copy to Clipboard buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                if st.button(f"👍", key=f"like_{idx}"):
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Liked")
                    else:
                        st.session_state["feedback"][idx] = "Liked"
                    st.success(f"Feedback for message {idx} set to: Liked")
            with col2:
                if st.button(f"👎", key=f"dislike_{idx}"):
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Disliked")
                    else:
                        st.session_state["feedback"][idx] = "Disliked"
                    st.success(f"Feedback for message {idx} set to: Disliked")
            with col3:
                if st.button(f"🔄", key=f"regenerate_{idx}"):
                    try:
                        # Re-generate the response
                        user_message = st.session_state["messages"][-2]["content"]
                        with st.spinner("Re-generating response..."):
                            result = qa_chain({"query": user_message})
                            amanda_message = result["result"].strip()
                            st.session_state["messages"][-1]["content"] = amanda_message
                            st.experimental_rerun()  # Update the chat history
                    except Exception as e:
                        st.error(f"Error during regeneration: {e}")
            with col4:
                if st.button(f"📋", key=f"copy_{idx}"):
                    pyperclip.copy(message['content'])
                    st.success("Copied to clipboard!")

# Form to handle user input and submission at the bottom
st.markdown("---")
form_key = f"chat_input_{len(st.session_state['messages'])}"
with st.form(form_key, clear_on_submit=True):
    user_input = st.text_input("You:", key="input", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

# Handle user input and generate response with RAG
if submitted and user_input:
    # Store the user input in session state to retain it across reruns
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Add delay to simulate processing and improve responsiveness
    with st.spinner("Amanda is thinking..."):
        try:
            # Simulate a slight delay
            time.sleep(1)
            result = qa_chain({"query": user_input})

            # Validate result before accessing keys
            if "result" in result and result["result"]:
                amanda_message = result["result"].strip()
            else:
                st.error("Unexpected response from QA chain.")
                amanda_message = "Sorry, I couldn't process your query."

            # Calculate tokens and cost
            num_tokens = num_tokens_from_messages(st.session_state["messages"], model="gpt-3.5-turbo")
            cost = calculate_cost(num_tokens, model="gpt-3.5-turbo")

            # Append Amanda's response
            st.session_state["messages"].append({"role": "assistant", "content": amanda_message})
            st.session_state["feedback"].append(None)  # No feedback initially

            # Store source document if it exists
            source_documents = result.get("source_documents", [])
            if source_documents:
                most_relevant_source = source_documents[0]  # Get the most relevant source
                metadata = most_relevant_source.metadata
                filename = os.path.basename(metadata.get("source", "Unknown"))
                page_number = metadata.get("page", "Unknown")
                st.session_state["messages"][-1]["source"] = {
                    "filename": filename,
                    "page": page_number,
                    "tokens": num_tokens,
                    "cost": cost
                }

        except Exception as e:
            st.error(f"Error while generating response: {e}")

    # Scroll to bottom after a new message
    st.experimental_rerun()  # Ensure new message is reflected and scroll to bottom

# Footer
st.markdown("---")
st.write("© 2024 - Elias-Charbel Salameh and Antonio Haddad")
