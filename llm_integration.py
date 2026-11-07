import os
import time
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

Provider = Literal["gemini", "voyage"]


class LLMIntegration:
    DEFAULT_PROVIDER: Provider = "gemini"

    def __init__(self):
        self.embedding_provider = self.DEFAULT_PROVIDER
        self.llm = OpenAI(
            api_key=os.environ["XKIRO_API_KEY"],
            base_url="https://api.xkiro.com/v1",
        )

    def invoke_llm(
        self,
        prompt_or_messages: str
        | list[ChatCompletionMessageParam]
        | list[dict[str, Any]],
    ) -> str:
        """Send a prompt string or a list of chat messages to the LLM."""
        if isinstance(prompt_or_messages, str):
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt_or_messages}
            ]
        else:
            messages = cast(list[ChatCompletionMessageParam], prompt_or_messages)

        llm_response = self.llm.chat.completions.create(
            model="mistralai/mistral-large-2512",
            messages=messages,
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

    def answer_query(
        self,
        vector_db,
        query: str,
        chat_history: list[dict] | None = None,
        k: int = 4,
    ) -> str:
        """Retrieve relevant context and answer using both document chunks and chat history."""
        retrieved_docs = vector_db.similarity_search(query, k=k)
        context = "\n\n".join(document.page_content for document in retrieved_docs)

        system_instruction = f"""You are a helpful assistant.
Answer using the provided document context and conversation history.
If the answer is not available in the context, say "I don't know".

Document Context:
{context}"""

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_instruction}
        ]

        if chat_history:
            # Include recent chat history (e.g. last 6 turns)
            for msg in chat_history[-6:]:
                content = str(msg.get("content", ""))
                if msg.get("role") == "ai":
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "user", "content": content})

        # Append the current query
        messages.append({"role": "user", "content": query})

        return self.invoke_llm(messages)
