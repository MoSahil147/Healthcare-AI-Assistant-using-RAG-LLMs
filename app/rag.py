import logging

from app.config import DISTANCE_THRESHOLD, primary_llm
from app.embeddings import query_similar
from app.llm import build_prompt, generate

logger = logging.getLogger(__name__)

FALLBACK = "I could not find this information in the provided documents."

# in-memory conversation history — used by query rewriting
# so follow-up questions like "what about that?" can be clarified
conversation_history = []


def rewrite_query(question):
    # first question ever? nothing to rewrite
    if not conversation_history:
        return question

    # build a short summary of the last few turns
    history_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}"
        for t in conversation_history[-6:]  # last 3 turns is enough
    )
    prompt = (
        "Given this conversation history and a vague follow-up question, "
        "rewrite it as a clear standalone question. Output only the rewritten question.\n\n"
        f"History:\n{history_text}\n\n"
        f"Vague: {question}\nRewritten:"
    )
    rewritten = primary_llm.invoke(prompt).content.strip()
    logger.info("Rewrite: '%s' -> '%s'", question, rewritten)
    return rewritten


def _confidence(distance):
    # how close was the best chunk to the question?
    if distance < 0.3: return "high"
    if distance < 0.6: return "medium"
    return "low"


def answer_question(question):
    # (1) clarify vague follow-ups before searching
    clear = rewrite_query(question)

    # (2) find the most relevant chunks in ChromaDB
    chunks = query_similar(clear)

    # (3) nothing close enough? better to say so than to make something up
    if not chunks or chunks[0]["distance"] > DISTANCE_THRESHOLD:
        logger.info("Nothing relevant found, returning fallback")
        return {"answer": FALLBACK, "sources": [], "confidence": "none"}

    # (4) stitch chunks into context, build prompt, call LLM
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = build_prompt(context=context, question=clear)
    answer = generate(prompt)

    if not answer or len(answer) < 5:
        answer = FALLBACK

    # (5) save exchange to history so next question can use it for rewriting
    conversation_history.append({"role": "user",      "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    sources = [
        {"document": c["source"], "chunk": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]}
        for c in chunks
    ]
    return {"answer": answer, "sources": sources, "confidence": _confidence(chunks[0]["distance"])}
