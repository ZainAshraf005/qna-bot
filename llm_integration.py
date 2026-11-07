import time
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings

Provider = Literal["gemini", "voyage"]


class LLMIntegration:
    def __init__(self, llm, embedding_provider: Provider = "gemini"):
        self.llm = llm
        self.embedding_provider = embedding_provider

    def invoke_llm(self, prompt: str) -> str:
        llm_response = self.llm.chat.completions.create(
            model="mistralai/mistral-large-2512",
            messages=[{"role": "user", "content": prompt}],
        )
        return (
            llm_response.choices[0].message.content
            or "Please try again later. I am unable to answer your question at the moment."
        )

    def embed_with_voyage(self, docs):
        embeddings = VoyageAIEmbeddings(
            model="voyage-4-lite",
            batch_size=10,
        )
        vector_db = InMemoryVectorStore(embedding=embeddings)

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
        print("Voyage document ingestion completed!")
        return vector_db

    def embed_with_gemini(self, docs):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        print(f"Embedding {len(docs)} chunks directly with Gemini...")
        vector_db = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
        print("Gemini document ingestion completed!")
        return vector_db

    def document_process(self, file_path: str):

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

        # 3. Create vector store based on selected provider
        if self.embedding_provider == "gemini":
            vector_db = self.embed_with_gemini(docs)
        elif self.embedding_provider == "voyage":
            vector_db = self.embed_with_voyage(docs)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding_provider}"
            )

        return vector_db

    def answer_query(self, vector_db, query: str, k: int = 4) -> str:
        retrieved_docs = vector_db.similarity_search(query, k=k)
        context = "\n\n".join(document.page_content for document in retrieved_docs)
        prompt = f"""You are a helpful assistant.
                  Answer using only the provided context.
                  Context: {context} Question: {query}"""
        return self.invoke_llm(prompt)
