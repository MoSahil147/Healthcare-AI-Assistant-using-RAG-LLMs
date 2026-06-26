from __future__ import annotations

import logging                          # stdlib: write log messages to console/file
import uuid                             # stdlib: generate unique IDs for requests
from contextlib import asynccontextmanager  # stdlib: define async startup/shutdown lifecycle
from pathlib import Path                # stdlib: filesystem path manipulation

from fastapi import FastAPI, HTTPException, Request  # web framework: app, error responses, request obj
from fastapi.middleware.cors import CORSMiddleware   # allow cross-origin requests from the frontend
from fastapi.responses import FileResponse           # serve files (e.g. HTML) as HTTP responses
from fastapi.staticfiles import StaticFiles          # mount a folder to serve static assets
from pydantic import BaseModel, field_validator      # data validation and request/response schemas
from slowapi import Limiter, _rate_limit_exceeded_handler  # rate limiting for API endpoints
from slowapi.errors import RateLimitExceeded         # exception raised when a rate limit is hit
from slowapi.util import get_remote_address          # helper to extract caller's IP for rate limiting

from app.agent import route                          # local: LLM routing logic (picks model/agent)
from app.config import DATA_DIR, GROQ_MODEL_FALLBACK, GROQ_MODEL_PRIMARY, validate_env, EMBEDDING_MODEL  # local: env config
from app.embeddings import ingest_documents          # local: load docs into ChromaDB vector store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", # time, loglevel eg: DEBUG, INFO, WARNING, ERROR, etc; Slogger name, the msg
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# rate limiter keyed by client IP address
limiter = Limiter(key_func=get_remote_address)

# crazy feature by FastAPI that lets you run the code before
# the app start and also after it shut down, betetr nah will be preped befr any request served
# to be precise there is something called ASGI startup protocol, has a built in startup and shutdown events, 
# FASTAPI hooks into these and runs the lifesapn fucntion
@asynccontextmanager # converts the generator funct into something usable
async def lifespan(app: FastAPI):
    # validate env vars first so the server does not start in a broken state
    validate_env() # if keys missing khalas
    # ingest all docs so the vector store is ready before the first request comes in
    logger.info("Starting up -- ingesting from %s", DATA_DIR)
    try:
        count = ingest_documents()
        logger.info("Ready -- %d chunks stored", count)
    except Exception as exc:
        # if ingest fails (e.g. HF API is down), log it and keep the server up
        # the /ingest endpoint can be called manually once the issue is resolved
        logger.error("Ingest failed on startup: %s", exc)
    yield #pauses the function and hands control back to the caller
    logger.info("Shutting down")


app = FastAPI(
    title="Healthcare AI Assistant",
    description="RAG-based healthcare Q&A with agentic appointment routing",
    version="1.0.0",
    lifespan=lifespan,
)

# attaching the rate limiter to the app, so slowapi can access it globally
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# techincally when someone exceeds more that 20 req per minute, will return 429 too many req error, not crashing


# allow requests from any origin so the UI and curl both work without CORS errors
# allows requests from any domain, we dont want any issue, will mark * and allow everything
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# origin is domains, we wont block other ports right!
# method is GET, POST
# header sis allow Content Types Authorization, etc 


# without this, API only, no UI
# serve the chat UI, the index.html lives in /static at the project root
# server the ui from the /static folder
_static_dir = Path(__file__).parent.parent / "static"
# __file__         =  /project/app/main.py
# .parent          =  /project/app/
# .parent.parent   =  /project/
# / "static"       =  /project/static/
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
# mount tells FASTAPI to handle the folder of files
# any req starting with /static will be handled, kind of a prefix it sees
# StaticFiles is FastAPI build in file server
# static name label is used internally by FastAPI to reference it.

# see showing /ask, /ingest, /health over docs is great
# but showing ui is pointless, which is managed by /, false hides it
@app.get("/", include_in_schema=False)
# this is the function that runs when someone visits localhost
def root():
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index)) # will return the actual HTML file back to browswer, Chat UI loads
    return {"message": "Healthcare AI Assistant -- visit /docs for the API"}


class AskRequest(BaseModel):
    question: str # first validates that the question is a string
    session_id: str = "" # accepts session_id, default is empty
    @field_validator("session_id", mode="before") # runs before pydantic validates the field
    @classmethod
    def default_session_id(cls, v: str) -> str:
        # if user sends session_id, use as it is or auto-generate a unique one uuid
        # generate a session ID automatically if the client does not provide one
        return v or str(uuid.uuid4())
    # cls is class itself and v the value the user sent


class AskResponse(BaseModel):
    answer: str # will come from the LLM
    sources: list # from db
    confidence: str # calc in rag
    tool_used: str | None = None    # only set when the appointment tool was used
    tool_output: dict | None = None  # the raw slot data, used by the UI to render chips

# get is, reads something from the server, no data sent
# post, send data to server to do something
@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL_PRIMARY, "fallback_model": GROQ_MODEL_FALLBACK}

# @ is just a decorator, wraps the function below it with extra behaviour
@app.post("/ingest")
def ingest():
    # manually re-ingest if you have added or changed documents
    logger.info("POST /ingest triggered")
    try:
        count = ingest_documents()
        return {"status": "ok", "chunks_stored": count}
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
@limiter.limit("20/minute")  # 20 requests per minute per IP to protect Groq quota
def ask(request: Request, body: AskRequest):
    # logs every incoming request with sessionID and question
    logger.info("POST /ask [session=%s] -- %s", body.session_id, body.question)
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try:
        result = route(body.question, body.session_id)
        logger.info("Done -- confidence: %s", result.get("confidence"))
        return result
    except Exception as exc:
        logger.error("Ask failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    # 500 is internal server issue
