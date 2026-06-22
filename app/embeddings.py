import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE,
    COLLECTION_NAME, DATA_DIR, TOP_K, get_embeddings,
)

logger = logging.getLogger(__name__)

_CHROMA_META = {"hnsw:space": "cosine"}

# singleton, created on first query and reused for the lifetime of the process
# the original code opened a new Chroma connection on every request, which was slow
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
            collection_metadata=_CHROMA_META,
        )
    return _vectorstore


def ingest_documents(data_dir: str = DATA_DIR) -> int:
    global _vectorstore

    txt_files = list(Path(data_dir).glob("*.txt"))
    if not txt_files:
        logger.warning("no .txt files found in %s", data_dir)
        return 0

    all_docs = []
    for path in txt_files:
        # read the file directly rather than using TextLoader so we have no
        # dependency on langchain-community for something this straightforward
        text = path.read_text(encoding="utf-8")
        all_docs.append(Document(page_content=text, metadata={"source": str(path)}))
        logger.info("Loaded %s", path.name)

    # chop into overlapping chunks so sentences at chunk boundaries keep their context
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    logger.info("Got %d chunks total", len(chunks))

    # write to a temporary path first, then rename it into place
    # this way any in-flight queries against the old store are not disrupted
    tmp_path = CHROMA_PATH + "_tmp"
    if Path(tmp_path).exists():
        shutil.rmtree(tmp_path)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=tmp_path,
        collection_name=COLLECTION_NAME,
        collection_metadata=_CHROMA_META,
    )

    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)
    Path(tmp_path).rename(CHROMA_PATH)

    # reset the singleton so the next query picks up the freshly built store
    _vectorstore = None
    logger.info("Stored %d chunks in ChromaDB at %s", len(chunks), CHROMA_PATH)
    return len(chunks)


def query_similar(question: str, top_k: int = TOP_K) -> list[dict]:
    results = _get_vectorstore().similarity_search_with_score(question, k=top_k)

    hits = [
        {
            "text":     doc.page_content,
            "source":   Path(doc.metadata.get("source", "unknown")).name,
            "distance": float(dist),  # 0 means identical, 1 means completely different
        }
        for doc, dist in results
    ]
    if hits:
        logger.info("Retrieved %d chunks (top distance: %.3f)", len(hits), hits[0]["distance"])
    return hits
