import time

import streamlit as st
from dotenv import load_dotenv

from llm_integration import LLMIntegration

load_dotenv()

bot = LLMIntegration()

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

provider_name = bot.embedding_provider.capitalize()
st.subheader(f"QnA Bot — Powered by {provider_name} & Mistral")

if not st.session_state.document_uploaded:
    file = st.file_uploader("Upload a PDF document", type=["pdf"], key="file_uploader")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner(
            f"Processing document with {provider_name} embeddings..."
        ):
            vector_db = bot.document_process("./uploaded_document.pdf")
            st.session_state.vector_db = vector_db
            st.session_state.document_uploaded = True

        st.markdown("document processed successfully!")
        time.sleep(2)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db:
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask a question about the uploaded document: ")
    if query:
        st.chat_message("user").markdown(query)

        result = bot.answer_query(
            vector_db=st.session_state.vector_db,
            query=query,
            chat_history=st.session_state.messages,
        )

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "ai", "content": result})
        st.chat_message("ai").markdown(result)
