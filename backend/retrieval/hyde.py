import logging
import os
from typing import List

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
_LOG = logging.getLogger(__name__)

_PROMPT = (
    "Write a short factual paragraph that would answer the following question. "
    "Be specific and concise. Do not say you don't know.\n\nQuestion: {query}"
)


def hyde_embedding_for_query(query: str) -> List[float]:
    """
    Generate a hypothetical document for the query and return its embedding.
    """
    try:
        from ingestion.embedder import _embed_texts
    except ModuleNotFoundError:
        from backend.ingestion.embedder import _embed_texts

    response = _CLIENT.chat.completions.create(
        model=_LLM_MODEL,
        messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
    )
    hypothetical = response.choices[0].message.content.strip()

    embeddings = _embed_texts([hypothetical])
    return embeddings[0] if embeddings else []