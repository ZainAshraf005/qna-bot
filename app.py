import time

import streamlit as st
from dotenv import load_dotenv

from llm_integration import LLMIntegration

load_dotenv()

st.set_page_config(
    page_title="DocQnA — AI PDF Assistant",
    page_icon="📚",
    layout="wide",
)

bot = LLMIntegration()

# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "filename" not in st.session_state:
    st.session_state.filename = None

provider_name = bot.embedding_provider.capitalize()

# --- Sidebar ---
with st.sidebar:
    st.title("📚 DocQnA")
    st.caption("AI-Powered Document Assistant")
    st.divider()

    st.markdown("### ⚙️ Engine Info")
    st.markdown(f"**Embeddings:** `{provider_name}`")
    st.markdown("**LLM:** `Mistral-Large`")
    st.divider()

    if st.session_state.document_uploaded:
        st.markdown("### 📄 Active Document")
        st.success(f"**{st.session_state.filename or 'Uploaded PDF'}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New File", use_container_width=True):
                st.session_state.document_uploaded = False
                st.session_state.vector_db = None
                st.session_state.messages = []
                st.session_state.filename = None
                st.rerun()
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("Upload a PDF to get started.")

# --- Main UI ---
if not st.session_state.document_uploaded:
    st.title("📄 PDF Question & Answer")
    st.write(
        "Upload any PDF document to chat with its contents using semantic search and AI."
    )

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to index and query",
        )
        if uploaded_file:
            st.session_state.filename = uploaded_file.name
            with open("uploaded_document.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner(
                f"⚡ Processing document with {provider_name} embeddings..."
            ):
                vector_db = bot.document_process("./uploaded_document.pdf")
                st.session_state.vector_db = vector_db
                st.session_state.document_uploaded = True

            st.success("Document indexed successfully! Launching chat...")
            time.sleep(1)
            st.rerun()

else:
    # Header above chat
    col_title, col_action = st.columns([5, 1])
    with col_title:
        st.markdown(f"### 💬 Chat with `{st.session_state.filename}`")
    with col_action:
        st.caption("🟢 Ready")

    # Display empty-state helper if no messages yet
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("👋 **Ready! Ask any question about your document.**")
            st.markdown("Here are a few ideas to get started:")
            st.markdown("- *Can you summarize the main points of this document?*")
            st.markdown("- *What are the key findings or takeaways?*")
            st.markdown("- *Find any names, dates, or figures mentioned.*")

    # Display chat history
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask a question about your document...")
    if query:
        # Display user question immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Generate answer with spinner
        with (
            st.chat_message("assistant", avatar="🤖"),
            st.spinner("Searching document & formulating answer..."),
        ):
            result = bot.answer_query(
                vector_db=st.session_state.vector_db,
                query=query,
                chat_history=st.session_state.messages,
            )
            st.markdown(result)

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "ai", "content": result})
