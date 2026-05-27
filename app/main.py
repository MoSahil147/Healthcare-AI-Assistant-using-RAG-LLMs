from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import route
from app.config import DATA_DIR, GROQ_MODEL_FALLBACK, GROQ_MODEL_PRIMARY
from app.embeddings import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once on startup — ingest all docs so the vector store is ready
    # before the first request comes in, no manual /ingest needed
    logger.info("Starting up — ingesting from %s", DATA_DIR)
    try:
        count = ingest_documents()
        logger.info("Ready — %d chunks stored", count)
    except Exception as exc:
        # if ingest fails (e.g. HF API is down), log it and keep the server up
        # the /ingest endpoint can be called manually once the issue is fixed
        logger.error("Ingest failed on startup: %s", exc)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Healthcare AI Assistant",
    description="RAG-based healthcare Q&A with agentic appointment routing",
    version="1.0.0",
    lifespan=lifespan,
)

# allow requests from any origin so the UI and curl both work without CORS errors
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# serve the chat UI — the index.html lives in /static at the project root
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
def root():
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Healthcare AI Assistant — visit /docs for the API"}


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    confidence: str
    tool_used: str | None = None    # only set when the appointment tool was used
    tool_output: dict | None = None  # the raw slot data, used by the UI to render chips


@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL_PRIMARY, "fallback_model": GROQ_MODEL_FALLBACK}


@app.post("/ingest")
def ingest():
    # manually re-ingest if you've added or changed documents
    logger.info("POST /ingest triggered")
    try:
        count = ingest_documents()
        return {"status": "ok", "chunks_stored": count}
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    logger.info("POST /ask — %s", request.question)
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try:
        result = route(request.question)
        logger.info("Done — confidence: %s", result.get("confidence"))
        return result
    except Exception as exc:
        logger.error("Ask failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
