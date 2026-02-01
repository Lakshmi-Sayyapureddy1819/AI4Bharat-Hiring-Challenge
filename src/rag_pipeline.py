
import os
import requests
from dotenv import load_dotenv
from typing import Optional

# ✅ Modern LangChain 0.3+ Legacy Imports
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def _call_local_asr(audio_path: str, server_url: str = "http://localhost:8000/transcribe", retries: int = 1, timeout: int = 30) -> Optional[str]:
    """Task 3: Send audio file to local FastAPI ASR service with robust response parsing and optional retries."""
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return None

    for attempt in range(retries + 1):
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                resp = requests.post(server_url, files=files, timeout=timeout)
                resp.raise_for_status()

                # Support a few possible response shapes
                data = resp.json()
                if isinstance(data, str):
                    text = data
                elif isinstance(data, dict):
                    # Common keys returned by various ASR services
                    text = (
                        data.get("transcription")
                        or data.get("transcript")
                        or data.get("text")
                        or data.get("result")
                    )
                    # Some services nest data under "data" or "result"
                    if not text and "data" in data and isinstance(data["data"], dict):
                        nested = data["data"]
                        text = nested.get("transcription") or nested.get("text")
                else:
                    text = None

                if text:
                    text = text.strip()
                    if text:
                        return text
                    else:
                        print("ASR returned an empty transcription string.")
                else:
                    print(f"Unexpected ASR response format (attempt {attempt+1}): {data}")

        except requests.exceptions.RequestException as e:
            print(f"ASR HTTP error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"ASR Error (attempt {attempt+1}): {e}")

    return None


def run_voice_rag(audio_path: str, top_k: int = 2, server_url: str = "http://localhost:8000/transcribe") -> str:
    """Task 5: End-to-end voice RAG pipeline: Transcribe -> Retrieve -> Answer.

    This version adds clearer errors for ASR failures, better validations for
    transcription text, and stronger error messages around embedding/vector/LLM init.
    """

    # 1. ASR - Transcribe voice to text
    transcription = _call_local_asr(audio_path, server_url=server_url, retries=1)
    if transcription is None:
        return "ASR service not responding or returned no transcription. Start FastAPI with uvicorn and ensure ASR_MODEL_PATH is set correctly."

    transcription = transcription.strip()
    if len(transcription) < 3:
        return "Transcription too short. Please try again with clearer audio."

    # 2. Setup Embeddings and Vector DB
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY missing in .env"

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    except Exception as e:
        return f"Failed to initialize embeddings: {e}"

    persist_dir = "data/chroma_db"
    if not os.path.exists(persist_dir) or not any(os.scandir(persist_dir)):
        return "Vector DB not found or empty. Run scraper first."

    try:
        # Initialize Chroma using the updated langchain_chroma package
        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    except Exception as e:
        return f"Failed to initialize vector DB: {e}"

    # 3. Setup LLM (GPT-4o)
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=api_key
        )
    except Exception as e:
        return f"Failed to initialize LLM: {e}"

    # 4. Define Prompt and Retrieval Chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build and execute the modern retrieval chain logic
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": transcription})

        # Support a couple of possible return shapes
        if isinstance(response, dict):
            return response.get("answer") or response.get("output") or str(response)
        return str(response)

    except Exception as e:
        return f"RAG execution failed: {e}"