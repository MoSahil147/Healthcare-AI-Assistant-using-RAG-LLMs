import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# tweak these if you want to experiment with different models or chunk sizes
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"   # used when the primary hits a rate limit
CHROMA_PATH         = "./store/chroma"
COLLECTION_NAME     = "healthcare_docs"
CHUNK_SIZE          = 800    # characters per chunk, bigger means more context per result
CHUNK_OVERLAP       = 100    # overlap so section headers do not get cut off at boundaries
TOP_K               = 4      # fetch 4 chunks, more coverage without getting too noisy
DATA_DIR            = "./data"
DISTANCE_THRESHOLD  = 0.8    # above this the chunk is too far away to be useful

# API keys loaded from .env, never hardcoded
HF_TOKEN     = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def validate_env() -> None:
    # called at startup so it fails loud and early, rather than with a cryptic error on the first request
    missing = [name for name, val in [("HF_TOKEN", HF_TOKEN), ("GROQ_API_KEY", GROQ_API_KEY)] if not val]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


# lazy singletons, models are created on first use rather than at import time
# this keeps unit tests fast (no network calls on import) and avoids slow cold starts
_embeddings   = None
_primary_llm  = None
_fallback_llm = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        _embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL, huggingfacehub_api_token=HF_TOKEN
        )
    return _embeddings


def get_primary_llm():
    global _primary_llm
    if _primary_llm is None:
        from langchain_groq import ChatGroq
        _primary_llm = ChatGroq(api_key=SecretStr(GROQ_API_KEY), model=GROQ_MODEL_PRIMARY, temperature=0)
    return _primary_llm


def get_fallback_llm():
    global _fallback_llm
    if _fallback_llm is None:
        from langchain_groq import ChatGroq
        _fallback_llm = ChatGroq(api_key=SecretStr(GROQ_API_KEY), model=GROQ_MODEL_FALLBACK, temperature=0)
    return _fallback_llm
