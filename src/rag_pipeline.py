import os
import requests
from dotenv import load_dotenv
from typing import Optional

# ✅ CORRECT LangChain imports (0.3+)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


def _call_local_asr(
    audio_path: str,
    server_url: str = "http://localhost:8000/transcribe"
) -> Optional[str]:
    """Send audio file to local FastAPI ASR service."""
    if not os.path.exists(audio_path):
        print("Audio file not found")
        return None

    try:
        with open(audio_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_path), f, "audio/wav")
            }
            resp = requests.post(server_url, files=files, timeout=30)
            resp.raise_for_status()
            return resp.json().get("transcription")
    except Exception as e:
        print(f"ASR Error: {e}")
        return None


def run_voice_rag(audio_path: str, top_k: int = 2) -> str:
    """End-to-end voice RAG pipeline."""

    # 1. ASR
    transcription = _call_local_asr(audio_path)
    if not transcription:
        return "ASR service not responding. Start FastAPI with uvicorn."

    # 2. Embeddings + Vector DB
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY missing in .env"

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    persist_dir = "data/chroma_db"
    if not os.path.exists(persist_dir):
        return "Vector DB not found. Run scraper first."

    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

    # 3. LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key
    )

    # 4. Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # 5. RAG chain
    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=doc_chain
    )

    # 6. Execute
    try:
        result = rag_chain.invoke({"input": transcription})
        return result.get("answer") or result.get("output_text", "")
    except Exception as e:
        return f"RAG execution failed: {e}"
