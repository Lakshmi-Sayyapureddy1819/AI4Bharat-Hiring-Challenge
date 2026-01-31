# Voice-Enabled RAG System

This repository contains a minimal scaffold for a voice-enabled RAG (Retrieval-Augmented Generation) system.

Setup

1. Create a Python virtual environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env` and fill in your API keys.

Quick start

- Scrape a topic: `python src/scraper.py --topic "Artificial Intelligence"`
- Run ASR service: `uvicorn src.asr_service:app --reload --port 8000`
- Run Streamlit UI: `streamlit run app.py`

Files

- `src/scraper.py`: simple Wikipedia scraper
- `src/vector_db.py`: chunking + Chroma persistence
- `src/asr_service.py`: FastAPI wrapper for ASR model (Nemo)
- `src/translation.py`: Sarvam translation wrapper
- `src/rag_pipeline.py`: simple retrieval + LLM answer pipeline
- `app.py`: Streamlit UI
