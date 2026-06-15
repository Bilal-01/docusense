import logging
import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")
_LOG = logging.getLogger(__name__)

_PROMPT = (
    "Write a short factual paragraph that would answer the following question. "
    "Be specific and concise. Do not say you don't know.\n\nQuestion: {query}"
)


def hyde_embedding_for_query(query: str) -> List[float]:
    """
    Generate a hypothetical document for the query and return its embedding.
    The embedding is used for dense retrieval in place of the raw query embedding.
    """
    try:
        from ingestion.embedder import _embed_texts
    except ModuleNotFoundError:
        from backend.ingestion.embedder import _embed_texts

    model = genai.GenerativeModel(_LLM_MODEL)
    hypothetical = model.generate_content(_PROMPT.format(query=query)).text.strip()

    embeddings = _embed_texts([hypothetical])
    return embeddings[0] if embeddings else []