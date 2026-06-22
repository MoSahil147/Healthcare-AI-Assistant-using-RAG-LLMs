import logging
from collections import OrderedDict

from app.config import DISTANCE_THRESHOLD, get_primary_llm
from app.embeddings import query_similar
from app.llm import build_prompt, generate

logger = logging.getLogger(__name__)

FALLBACK = "I could not find this information in the provided documents."

# one history list per session, keyed by session_id
# using OrderedDict lets us evict the oldest session when the cap is reached
_MAX_SESSIONS = 1000
_session_histories: OrderedDict[str, list] = OrderedDict()


def _get_history(session_id: str) -> list:
    if session_id not in _session_histories:
        if len(_session_histories) >= _MAX_SESSIONS:
            _session_histories.popitem(last=False)  # drop the oldest session
        _session_histories[session_id] = []
    return _session_histories[session_id]


def rewrite_query(question: str, session_id: str) -> str:
    # first question in a session? nothing to rewrite, return as is
    history = _get_history(session_id)
    if not history:
        return question

    # build a short summary of the last few turns so vague follow-ups can be clarified
    history_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}"
        for t in history[-6:]  # last 3 turns is enough context
    )
    prompt = (
        "Given this conversation history and a vague follow-up question, "
        "rewrite it as a clear standalone question. Output only the rewritten question.\n\n"
        f"History:\n{history_text}\n\n"
        f"Vague: {question}\nRewritten:"
    )
    rewritten = get_primary_llm().invoke(prompt).content.strip()
    logger.info("Rewrite: '%s' -> '%s'", question, rewritten)
    return rewritten


def _confidence(distance: float) -> str:
    # how close was the best chunk to the question?
    if distance < 0.3:
        return "high"
    if distance < 0.6:
        return "medium"
    return "low"


def answer_question(question: str, session_id: str) -> dict:
    # step 1: clarify vague follow-ups before searching
    clear = rewrite_query(question, session_id)

    # step 2: find the most relevant chunks in ChromaDB
    chunks = query_similar(clear)

    # step 3: nothing close enough? better to say so than to make something up
    if not chunks or chunks[0]["distance"] > DISTANCE_THRESHOLD:
        logger.info("Nothing relevant found, returning fallback")
        return {"answer": FALLBACK, "sources": [], "confidence": "none"}

    # step 4: stitch chunks into context, build the prompt, call the LLM
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = build_prompt(context=context, question=clear)
    answer = generate(prompt)

    if not answer or len(answer) < 5:
        answer = FALLBACK

    # step 5: save the exchange to history so the next question can use it for rewriting
    history = _get_history(session_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    sources = [
        {
            "document": c["source"],
            "chunk": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources, "confidence": _confidence(chunks[0]["distance"])}
