import logging
import shutil # for deleting and renaming directories (used in atomic swap)
from pathlib import Path # cross-platform file and directory path handling

from langchain_chroma import Chroma # ChromaDB vector store wrapper
from langchain_core.documents import Document # container for text + metadata (source filename)
from langchain_text_splitters import RecursiveCharacterTextSplitter # splits large text into overlapping chunks

from app.config import (
    CHROMA_PATH,      # where ChromaDB is persisted on disk: ./store/chroma
    CHUNK_OVERLAP,    # 100 chars — shared overlap between consecutive chunks so context isn't lost at boundaries
    CHUNK_SIZE,       # 800 chars — max size of each chunk fed to the embedding model
    COLLECTION_NAME,  # logical name for this collection inside ChromaDB: "healthcare_docs"
    DATA_DIR,         # folder containing the source .txt files: ./data
    TOP_K,            # how many closest chunks to retrieve per query (4)
    get_embeddings,   # lazy singleton that returns the HuggingFace all-MiniLM-L6-v2 embedding model
)

logger = logging.getLogger(__name__)

# tells ChromaDB to measure similarity using cosine distance (0 = identical, 1 = completely different)
# cosine is preferred over euclidean for text embeddings because it ignores vector magnitude
_CHROMA_META = {"hnsw:space": "cosine"}

# singleton, created on first query and reused for the lifetime of the process
# instead of opening Chroma Connection for every run, we will load at once
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    # if none create a connection to ChromaDB files, not vectors warna return the same
    # just opening a communication beyween the Python and DB
    if _vectorstore is None: 
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
            collection_metadata=_CHROMA_META,
        )
    # second time will return the already created singleton, will skip above
    return _vectorstore


def ingest_documents(data_dir: str = DATA_DIR) -> int:
    global _vectorstore # we are telling python here to use the module level one here

    # Path.glob is lazy, doenst load all files into mem, finds them one at a time, list forces to find all files immediately and store them
    txt_files = list(Path(data_dir).glob("*.txt")) 
    if not txt_files:
        logger.warning("no .txt files found in %s", data_dir)
        return 0

    all_docs = []
    for path in txt_files:
        # read the file directly rather than using TextLoader so we have no
        # dependency on langchain-community for something this straightforward
        # how to read the bytes in a file and convert them to chars
        text = path.read_text(encoding="utf-8")
        all_docs.append(Document(page_content=text, metadata={"source": str(path)}))
        logger.info("Loaded %s", path.name)

    # chop into overlapping chunks so sentences at chunk boundaries keep their context
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    logger.info("Got %d chunks total", len(chunks))

    # write to a temporary path first, then rename it into place
    # this way any in-flight queries against the old store are not disrupted
    # temo coz, if we wrote directly to ./store/chroma, any ongoing /ask query would read a half-written DB and crash.
    tmp_path = CHROMA_PATH + "_tmp"
    if Path(tmp_path).exists(): # will delete already exits failed ingestion, will write fresh, if not will mix with new with the old
        shutil.rmtree(tmp_path)

    # will convert chunks to vectors and stores in DB
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=tmp_path, # writing in tmp path for now
        collection_name=COLLECTION_NAME,
        collection_metadata=_CHROMA_META,
    )

    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH) # whatever is ealrier in Chorma dubmp it
    Path(tmp_path).rename(CHROMA_PATH) # tmp path ke andar whatever is there will be stored now in DB

    # reset the singleton so the next query picks up the freshly built store
    # None because we deleted the old DB and built a new
    _vectorstore = None
    logger.info("Stored %d chunks in ChromaDB at %s", len(chunks), CHROMA_PATH)
    return len(chunks)


def query_similar(question: str, top_k: int = TOP_K) -> list[dict]:
    # through get_vectorstore we need to create a Chroma connection to disk, basically the python! 
    results = _get_vectorstore().similarity_search_with_score(question, k=top_k)

    hits = [
        {
            "text":     doc.page_content,
            # "./data/hipaa_privacy_guidelines.txt"  →  "hipaa_privacy_guidelines.txt"
            "source":   Path(doc.metadata.get("source", "unknown")).name, # with .name we will just extract the file name form the source 
            "distance": float(dist),  # 0 means identical, 1 means completely different
        }
        for doc, dist in results
    ]
    if hits:
        logger.info("Retrieved %d chunks (top distance: %.3f)", len(hits), hits[0]["distance"])
    return hits
