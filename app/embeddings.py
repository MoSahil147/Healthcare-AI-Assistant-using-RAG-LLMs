import logging
import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE,
    COLLECTION_NAME, DATA_DIR, TOP_K, embeddings,
)

logger = logging.getLogger(__name__)


def ingest_documents(data_dir=DATA_DIR):
    txt_files = list(Path(data_dir).glob("*.txt"))
    if not txt_files:
        logger.warning("no .txt files found in %s", data_dir)
        return 0

    # load all the files
    all_docs = []
    for path in txt_files:
        loader = TextLoader(str(path), encoding="utf-8")
        all_docs.extend(loader.load())
        logger.info("Loaded %s", path.name)

    # chop into overlapping chunks — overlap means sentences at the boundary
    # wont get split across two chunks and lose context
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    logger.info("Got %d chunks total", len(chunks))

    # wipe old store and rebuild fresh so no stale data remains
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},  # cosine distance not euclidean
    )
    logger.info("Stored %d chunks in ChromaDB at %s", len(chunks), CHROMA_PATH)
    return len(chunks)


def query_similar(question, top_k=TOP_K):
    # load the saved store and find which chunks are closest to the question
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    results = vectorstore.similarity_search_with_score(question, k=top_k)

    hits = [
        {
            "text":     doc.page_content,
            "source":   Path(doc.metadata.get("source", "unknown")).name,
            "distance": float(dist),  # 0 = identical, 1 = completely different
        }
        for doc, dist in results
    ]
    if hits:
        logger.info("Retrieved %d chunks (top distance: %.3f)", len(hits), hits[0]["distance"])
    return hits
