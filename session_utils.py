import streamlit as st

def initialize_session_state():
    if "debug_log" not in st.session_state:
        st.session_state["debug_log"] = []
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = []
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "all_chats" not in st.session_state:
        st.session_state["all_chats"] = []
    if "active_chat" not in st.session_state:
        st.session_state["active_chat"] = None

def log_debug(message):
    st.session_state["debug_log"].append(message)
