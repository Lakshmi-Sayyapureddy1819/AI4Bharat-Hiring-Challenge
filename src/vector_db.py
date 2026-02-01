import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables for API keys
load_dotenv()

def initialize_vector_db(file_path: str = "data/article.txt", persist_dir: str = "data/chroma_db"):
    """
    Task 2: Load scraped text, split into chunks, and persist in ChromaDB.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run the scraper or create the file manually.")
        return None

    # 1. Load the document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split text into manageable chunks for RAG
    # RecursiveCharacterTextSplitter is recommended for maintaining paragraph context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # 3. Setup Embeddings
    # Explicitly pass the API key to avoid environment resolution issues
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return None

    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        check_embedding_ctx_length=False  # Useful for specific Python 3.13/Pydantic environments
    )

    # 4. Create and Persist Vector Database
    print(f"Initializing Vector DB at {persist_dir}...")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # In some LangChain versions, an explicit persist() call is required to write to disk
    if hasattr(vector_db, 'persist'):
        vector_db.persist()
        
    print("Vector DB successfully initialized and persisted.")
    return vector_db

if __name__ == "__main__":
    # Allows running this script independently to fix the "Vector DB not found" error
    initialize_vector_db()