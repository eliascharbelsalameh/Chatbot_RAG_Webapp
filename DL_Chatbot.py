import streamlit as st
import openai
import os
import pyperclip  # for copy to clipboard functionality #type: ignore

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app layout with updated CSS for background color and button styles
st.markdown(
    """
    <style>
    /* Set the main background color */
    .reportview-container, .css-1outpf7 {
        background-color: #2c003e;  /* Midnight Purple */
        color: white;
    }

    /* Set the text color for headers and markdown */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown p {
        color: white;
    }

    /* Style the text input to have a dark background with light text */
    .stTextInput>div>div {
        background-color: #1e1e1e;
        color: white;
    }

    /* Set button text color */
    .stButton>button {
        color: black;
    }

    /* Set text color for markdown in general */
    .stMarkdown {
        color: white;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title
st.title("🤖 Chat with Amanda")
st.write("Amanda is here to help you. Feel free to chat and interact with her responses!")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are Amanda, a helpful assistant."}
    ]
if "feedback" not in st.session_state:
    st.session_state["feedback"] = []  # Store feedback for each response

# Form to handle user input and submission
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("You:", key="input", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

# Handle user input and generate response
if submitted and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate Amanda's response
    with st.spinner("Amanda is thinking..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state["messages"],
                temperature=0.7,
                max_tokens=150,
            )
            amanda_message = response.choices[0].message["content"].strip()
            # Append Amanda's response
            st.session_state["messages"].append({"role": "assistant", "content": amanda_message})
            st.session_state["feedback"].append(None)  # No feedback initially
        except Exception as e:
            st.error(f"Error: {e}")

# Display conversation history
st.markdown("---")
for idx, message in enumerate(st.session_state["messages"]):
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.markdown(f"**Amanda 🤖:** {message['content']}")

        # Like, Dislike, Re-generate, and Copy to Clipboard buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button(f"👍", key=f"like_{idx}"):
                st.session_state["feedback"][idx] = "Liked"
        with col2:
            if st.button(f"👎", key=f"dislike_{idx}"):
                st.session_state["feedback"][idx] = "Disliked"
        with col3:
            if st.button(f"🔄", key=f"regenerate_{idx}"):
                try:
                    # Re-generate the response
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"][:-1],  # remove Amanda's last response
                        temperature=0.7,
                        max_tokens=150,
                    )
                    amanda_message = response.choices[0].message["content"].strip()
                    # Update Amanda's last message
                    st.session_state["messages"][-1]["content"] = amanda_message
                except Exception as e:
                    st.error(f"Error: {e}")
        with col4:
            if st.button(f"📋", key=f"copy_{idx}"):
                pyperclip.copy(message['content'])
                st.success("Copied to clipboard!")

# Footer
st.markdown("---")
st.write("© 2024 - Elias-Charbel Salameh and Antonio Haddad")
