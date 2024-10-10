import streamlit as st
import openai
import os, os.path
import pyperclip # type: ignore
from langchain import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from langchain.chat_models import ChatOpenAI # type: ignore
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter # type: ignore
from langchain.docstore.document import Document # type: ignore
import pandas as pd
import docx # type: ignore
import tiktoken # type: ignore
import pdfplumber # type: ignore

# TODO: fix similarity score
# TODO: fix setup_vector_store()
# TODO: add text-to-speech

# Define pages for the Streamlit application
PAGES = ["Amanda", "Debugging", "User Feedback", "Chunks", "Similarity"]

# Set up Streamlit sidebar
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", PAGES)

# Initialize debugging log, feedback, chunks, and similarity
if "debug_log" not in st.session_state:
    st.session_state["debug_log"] = []
if "feedback" not in st.session_state:
    st.session_state["feedback"] = []  # Store feedback for each response
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []  # Store document chunks
if "last_query" not in st.session_state:
    st.session_state["last_query"] = None  # Store the last query
if "similar_chunks" not in st.session_state:
    st.session_state["similar_chunks"] = []  # Store similarity results

# Function to log debug messages
def log_debug(message):
    st.session_state["debug_log"].append(message)

# Amanda Page: Main chat interface
if selection == "Amanda":

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
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and introductory text
    st.title("🤖 Chat with Amanda (RAG Enhanced)")
    st.write("Amanda is now enhanced with Retrieval-Augmented Generation (RAG) for better, more accurate responses!")

    # Load PDF, Word, and Excel Documents from directories (recursively)
    def read_files_from_directory(directory_path):
        log_debug(f"Reading files from directory: {directory_path}")
        documents = []
        try:
            for dirpath, dirnames, filenames in os.walk(directory_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    log_debug(f"Processing file: {filename}")

                    if filename.endswith('.pdf'):
                        # Extract tables and text using pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            for page_number, page in enumerate(pdf.pages, start=1):
                                # Extract plain text from page
                                text = page.extract_text()
                                if text:
                                    documents.append(Document(page_content=text, metadata={"source": filename, "page": page_number}))
                                    log_debug(f"Extracted text from PDF: {filename}, Page: {page_number}")

                                # Extract tables from page
                                tables = page.extract_tables()
                                for table in tables:
                                    # Convert table to plain text (or structured text for better context)
                                    table_text = pd.DataFrame(table).to_string()
                                    documents.append(Document(page_content=table_text, metadata={"source": filename, "page": page_number, "type": "table"}))
                                    log_debug(f"Extracted table from PDF: {filename}, Page: {page_number}")

                    elif filename.endswith('.docx'):
                        doc = docx.Document(file_path)
                        full_text = []
                        for paragraph in doc.paragraphs:
                            full_text.append(paragraph.text)
                        documents.append(Document(page_content="\n".join(full_text), metadata={"source": filename}))
                        log_debug(f"Extracted text from Word document: {filename}")

                    elif filename.endswith('.xlsx'):
                        # Load Excel files using pandas
                        try:
                            df = pd.read_excel(file_path)
                            full_text = df.to_string()
                            documents.append(Document(page_content=full_text, metadata={"source": filename}))
                            log_debug(f"Extracted data from Excel file: {filename}")
                        except Exception as e:
                            st.error(f"Error reading Excel file {filename}: {e}")
                            log_debug(f"Error reading Excel file {filename}: {e}")

        except Exception as e:
            st.error(f"Error reading files from directory: {e}")
            log_debug(f"Error reading files from directory: {e}")
        
        log_debug(f"Total documents loaded: {len(documents)}")
        return documents

    # Define the directory where your files are stored
    input_directory = r"C:\Users\elias\OneDrive\Bureau\USEK-Electrical and Electronics Engineer\Semesters\Term-9_Fall-202510\GIN515-Deep Learning-non_repository\Files_dir_RAG"

    # Check if the FAISS index files exist and are not corrupted
    def faiss_index_exists():
        index_file = 'vectorstore/db_faiss/index.faiss'
        metadata_file = 'vectorstore/db_faiss/docstore.pkl'  # Replace with appropriate file if needed
        return os.path.exists(index_file) and os.path.exists(metadata_file)

    # Function to load the vector store, only recreate if it is corrupted or missing
    @st.cache_resource
    def load_vector_store():
        if faiss_index_exists():
            try:
                log_debug("Loading existing FAISS vector store...")
                # Loading the vector store with Hugging Face Embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
                vectorstore = FAISS.load_local('vectorstore/db_faiss', embeddings, allow_dangerous_deserialization=True)
                log_debug("Vector store loaded successfully.")
                return vectorstore
            
            except Exception as e:
                st.error("Existing FAISS vector store is corrupted. Recreating it...")
                log_debug(f"Error loading existing vector store: {e}")

        return recreate_vector_store()

    # Function to recreate the vector store by processing files again
    def recreate_vector_store():
        log_debug("Creating new FAISS vector store...")

        # Read files and process documents
        documents = read_files_from_directory(input_directory)
        log_debug(f"Total documents processed: {len(documents)}")

        if not documents:
            st.error("No documents were loaded. Please check the input directory.")
            return None  # Avoid proceeding if no documents are loaded

        # Use RecursiveCharacterTextSplitter instead of CharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,          # Define the desired chunk size
            chunk_overlap=50,        # Overlap between chunks for better context
            separators=["\n\n", "\n", ".", " "]  # Define how to split: first by double newlines, then newlines, then periods, then spaces
        )  
        texts = text_splitter.split_documents(documents)
        log_debug(f"Text splitter created {len(texts)} chunks.")
        
        if not texts:
            st.error("Text splitting failed. No chunks were created.")
            return None  # Avoid proceeding if no chunks were created

        # Store chunks in session state
        st.session_state["chunks"] = texts

        # Initialize embeddings and FAISS vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local('vectorstore/db_faiss')
            log_debug("Vector store created and saved successfully.")
            return vectorstore
        except Exception as e:
            st.error(f"Error creating FAISS vector store: {str(e)}")
            log_debug(f"Error creating FAISS vector store: {str(e)}")
            return None

    # Setup the vector store: load if exists, recreate if necessary
    @st.cache_resource
    def setup_vector_store(directory):
        return load_vector_store()

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
        log_debug("Creating RAG chain...")
        try:
            vectorstore = setup_vector_store(input_directory)  # Ensure we create/load vector store

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
            log_debug("RAG chain created successfully.")
            return qa_chain
        except Exception as e:
            st.error(f"Error creating RAG chain: {str(e)}")
            log_debug(f"Error creating RAG chain: {str(e)}")
            raise

    # Initialize the QA chain (can take time, so it's cached)
    qa_chain = create_rag_chain()

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are Amanda, a helpful assistant."}
        ]

    # Handle user input and generate response with RAG using st.chat_input
    user_input = st.chat_input("Type your message here...")

    # Proceed if user input is provided
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["last_query"] = user_input  # Store the last query

        with st.spinner("Amanda is thinking..."):
            try:
                log_debug(f"User input received: {user_input}")
                result = qa_chain({"query": user_input})

                # Validate result before accessing keys
                if "result" in result and result["result"]:
                    amanda_message = result["result"].strip()
                    log_debug(f"Amanda's response: {amanda_message}")
                else:
                    st.error("Unexpected response from QA chain.")
                    amanda_message = "Sorry, I couldn't process your query."

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
                    
                    # Simplified token calculation for cost
                    num_tokens = len(user_input) + len(amanda_message)  # Adjust this as needed
                    cost = calculate_cost(num_tokens, model="gpt-3.5-turbo")

                    st.session_state["messages"][-1]["source"] = {
                        "filename": filename,
                        "page": page_number,
                        "tokens": num_tokens,  # Add 'tokens' to the dictionary
                        "cost": cost,          # Add 'cost' to the dictionary
                    }
                    log_debug(f"Source document for response: {filename}, page {page_number}")

            except Exception as e:
                st.error(f"Error: {e}")
                log_debug(f"Error during response generation: {e}")

    # Display conversation history
    st.markdown("---")
    for idx, message in enumerate(st.session_state["messages"]):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Amanda 🤖:** {message['content']}")

            # Display the most relevant source information if available
            if "source" in message:
                source_info = message["source"]
                filename = source_info.get("filename", "Unknown")
                page = source_info.get("page", "Unknown")
                tokens = source_info.get("tokens", "N/A")  # Safeguard to prevent KeyError
                cost = source_info.get("cost", 0.0)  # Safeguard to prevent KeyError

                st.markdown(f"""<p style='color: grey;'>Source: File: <i>{filename}</i>, Page: <i>{page}</i></p>""", unsafe_allow_html=True)
                st.markdown(f"""<p style='color: grey;'>Tokens used: <i>{tokens}</i>, Cost: <i>${cost:.6f}</i></p>""", unsafe_allow_html=True)


            # Like, Dislike, Re-generate, and Copy to Clipboard buttons with smaller size and outlined icons
            col1, col2, col3, col4 = st.columns([0.1, 0.1, 0.1, 0.1])  # Adjusted column sizes to make the buttons smaller
            with col1:
                like_button = st.button(f"👍", key=f"like_{idx}")
                if like_button:
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Liked")
                    else:
                        st.session_state["feedback"][idx] = "Liked"
                    st.success(f"Feedback for message {idx} set to: Liked")

            with col2:
                dislike_button = st.button(f"👎", key=f"dislike_{idx}")
                if dislike_button:
                    if len(st.session_state["feedback"]) <= idx:
                        st.session_state["feedback"].append("Disliked")
                    else:
                        st.session_state["feedback"][idx] = "Disliked"
                    st.success(f"Feedback for message {idx} set to: Disliked")

            with col3:
                regenerate_button = st.button(f"🔄", key=f"regenerate_{idx}")
                if regenerate_button:
                    try:
                        # Re-generate the response
                        log_debug(f"Re-generating response for message: {st.session_state['messages'][-2]['content']}")
                        result = qa_chain({"query": st.session_state["messages"][-2]["content"]})
                        amanda_message = result["result"].strip()
                        st.session_state["messages"][-1]["content"] = amanda_message
                    except Exception as e:
                        st.error(f"Error: {e}")
                        log_debug(f"Error during response regeneration: {e}")

            with col4:
                copy_button = st.button(f"📋", key=f"copy_{idx}")
                if copy_button:
                    pyperclip.copy(message['content'])
                    st.success("Copied to clipboard!")

            # CSS Styling to reduce the size of the icons and make them empty inside
            st.markdown(
                """
                <style>
                .stButton>button {
                    padding: 5px 10px;
                    font-size: 18px;
                    border: 2px solid white;
                    background: transparent;
                    color: white;
                    border-radius: 5px;
                }
                .stButton>button:hover {
                    background: rgba(255, 255, 255, 0.1);
                }
                </style>
                """,
                unsafe_allow_html=True
            )

# Debugging Page: Display log messages
elif selection == "Debugging":
    st.title("Debugging Logs")
    st.markdown("### Debugging Information")
    for log_message in st.session_state["debug_log"]:
        st.text(log_message)

# User Feedback Page: Display all feedback collected so far
elif selection == "User Feedback":
    st.title("User Feedback Summary")
    st.markdown("### Feedback Given by Users")
    for i, feedback in enumerate(st.session_state["feedback"]):
        if feedback:
            st.markdown(f"Message {i}: {feedback}")

# Chunks Page: Display chunk settings and all available chunks
elif selection == "Chunks":
    st.title("Document Chunks")
    
    # Display chunk size and overlap
    chunk_size = 500
    chunk_overlap = 50
    st.write(f"**Chunk Size:** {chunk_size}")
    st.write(f"**Chunk Overlap:** {chunk_overlap}")

    # Display all chunks
    if st.session_state["chunks"]:
        for idx, chunk in enumerate(st.session_state["chunks"]):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "N/A")
            st.markdown(f"**Chunk {idx + 1} from {source}, Page: {page}:**")
            st.text(chunk.page_content)
    else:
        st.write("No chunks available. Please load documents.")

# Similarity Page: Display the top 3 most similar chunks to the last query
elif selection == "Similarity":
    st.title("Top 3 Similar Chunks")

    # Display the last user query
    if st.session_state["last_query"]:
        st.write(f"**Last Query:** {st.session_state['last_query']}")

        # Display top 3 most similar chunks
        if st.session_state["similar_chunks"]:
            for idx, (chunk, score) in enumerate(st.session_state["similar_chunks"]):
                source = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "N/A")
                st.markdown(f"**Similar Chunk {idx + 1} (Score: {score:.4f}) from {source}, Page: {page}:**")
                st.text(chunk.page_content)
        else:
            st.write("No similar chunks available.")
    else:
        st.write("No query has been made yet.")

# Footer
st.markdown("---")
st.write("© 2024 - Elias-Charbel Salameh and Antonio Haddad")