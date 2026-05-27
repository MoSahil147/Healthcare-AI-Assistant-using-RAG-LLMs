import logging

from groq import RateLimitError

from app.config import GROQ_MODEL_FALLBACK, GROQ_MODEL_PRIMARY, fallback_llm, primary_llm

logger = logging.getLogger(__name__)

# this prompt is the key to avoiding hallucinations —
# strict enough to prevent making things up, but flexible enough to handle
# questions that use different law names (e.g. "HIPAA" when doc covers DPDPA 2023)
PROMPT = """You are a healthcare assistant. Answer using ONLY the context provided below.
If the context contains relevant information — even if it uses different law names or terminology than the question — answer using what is in the context and note any differences.
Only respond with "I could not find this information in the provided documents." if the context has no relevant information at all.
Do not diagnose, prescribe, or give direct medical advice.
Keep your answer professional, accurate, and concise.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(context, question):
    return PROMPT.format(context=context, question=question)


def generate(prompt):
    try:
        logger.info("Calling %s", GROQ_MODEL_PRIMARY)
        answer = primary_llm.invoke(prompt).content.strip()
        logger.info("Got response (%d chars)", len(answer))
        return answer
    except RateLimitError:
        # primary got rate limited, quietly switch to the smaller model
        logger.warning("Rate limited, switching to %s", GROQ_MODEL_FALLBACK)
        return fallback_llm.invoke(prompt).content.strip()
