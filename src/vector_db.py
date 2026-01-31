import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def initialize_vector_db():
    # ... your splitting code ...
    
    # FORCED INITIALIZATION: 
    # Don't let LangChain "find" the key, give it directly.
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        check_embedding_ctx_length=False # Helps bypass some 3.13 validation bugs
    )
    
    vector_db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory="data/chroma_db"
    )
    return vector_db