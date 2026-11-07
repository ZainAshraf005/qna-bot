import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from openai import OpenAI

load_dotenv()

llm = OpenAI(
    api_key=os.environ["XKIRO_API_KEY"],
    base_url="https://api.xkiro.com/v1",
)

embeddings = VoyageAIEmbeddings(
    model="voyage-4-lite",
    batch_size=10,
)


def invoke_llm(prompt):
    llm_response = llm.chat.completions.create(
        model="mistralai/mistral-large-2512",
        messages=[{"role": "user", "content": prompt}],
    )
    return (
        llm_response.choices[0].message.content
        or "Please try again later. I am unable to answer your question at the moment."
    )


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


def document_process(file_path):

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(documents)

    print(f"Total chunks: {len(docs)}")

    # 4. Initialize vector store
    vector_db = InMemoryVectorStore(embedding=embeddings)

    # 5. Ingest documents with rate-limit protection
    batch_size = 10

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]

        print(f"Embedding chunks {i + 1} to {i + len(batch)}")

        vector_db.add_documents(batch)

        # Stay below 3 requests per minute
        if i + batch_size < len(docs):
            time.sleep(31)

    # Wait for the previous API requests to leave the rate-limit window
    time.sleep(35)
    print("Document ingestion completed!")
    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True


st.subheader("QnA Bot — Powered by VoyageAI and Mistral")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if not st.session_state.document_uploaded:
    file = st.file_uploader("Upload a PDF document", type=["pdf"], key="file_uploader")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing the uploaded document..."):
            document_process("./uploaded_document.pdf")

        st.markdown("document processed successfully!")
        time.sleep(2)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db:
    query = st.chat_input("Ask a question about the uploaded document: ")
    if query:
        st.chat_message("user").markdown(query)

        retrieved_docs = st.session_state.vector_db.similarity_search(query, k=4)
        context = "\n\n".join(document.page_content for document in retrieved_docs)
        prompt = f"""You are a helpful assistant.
                  Answer using only the provided context.
                  Context: {context} Question: {query}"""
        result = invoke_llm(prompt)
        st.chat_message("ai").markdown(result)
