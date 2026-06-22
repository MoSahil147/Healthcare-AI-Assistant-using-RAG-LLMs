import logging

from groq import RateLimitError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.config import GROQ_MODEL_FALLBACK, GROQ_MODEL_PRIMARY, get_fallback_llm, get_primary_llm

logger = logging.getLogger(__name__)

# this prompt is the key to avoiding hallucinations
# strict enough to prevent making things up, but flexible enough to handle
# questions that use different law names (e.g. "HIPAA" when the doc covers DPDPA 2023)
PROMPT = """You are a healthcare assistant. Answer using ONLY the context provided below.
If the context contains relevant information -- even if it uses different law names or terminology than the question -- answer using what is in the context and note any differences.
Only respond with "I could not find this information in the provided documents." if the context has no relevant information at all.
Do not diagnose, prescribe, or give direct medical advice.
Keep your answer professional, accurate, and concise.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(context: str, question: str) -> str:
    return PROMPT.format(context=context, question=question)


# retries up to 3 times with exponential backoff for any transient error
# rate limit errors are excluded because we want to fall back to the smaller model immediately
@retry(
    retry=retry_if_not_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _invoke_with_retry(llm, prompt: str) -> str:
    return llm.invoke(prompt).content.strip()


def generate(prompt: str) -> str:
    try:
        logger.info("Calling %s", GROQ_MODEL_PRIMARY)
        answer = _invoke_with_retry(get_primary_llm(), prompt)
        logger.info("Got response (%d chars)", len(answer))
        return answer
    except RateLimitError:
        # primary got rate limited, quietly switch to the smaller fallback model
        logger.warning("Rate limited, switching to %s", GROQ_MODEL_FALLBACK)
        return _invoke_with_retry(get_fallback_llm(), prompt)
