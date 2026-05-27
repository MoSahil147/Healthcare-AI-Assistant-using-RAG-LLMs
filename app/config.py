import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# tweak these if you want to experiment
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"   # used when primary hits rate limit
CHROMA_PATH        = "./store/chroma"
COLLECTION_NAME    = "healthcare_docs"
CHUNK_SIZE         = 500    # characters per chunk
CHUNK_OVERLAP      = 50     # overlap so nothing gets cut off at the seam
TOP_K              = 3      # how many chunks to fetch per question
DATA_DIR           = "./data"
DISTANCE_THRESHOLD = 0.8    # above this the chunk is too far away to be useful

# API keys — loaded from .env, never hardcoded
HF_TOKEN    = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# create the models once here, import them wherever needed
embeddings   = HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL, huggingfacehub_api_token=HF_TOKEN)
primary_llm  = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_PRIMARY,  temperature=0)
fallback_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_FALLBACK, temperature=0)
