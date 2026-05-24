import os
from typing import Tuple

import ollama
from dotenv import load_dotenv

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama2")


def hyde_hypothetical_answer(query: str) -> str:
    prompt = (
        f"Write a short hypothetical paragraph that would answer this question: {query}. "
        "Do not say you don't know. Just write a plausible answer."
    )

    try:
        with ollama.Client(host=OLLAMA_HOST) as client:
            # Try common generation API names
            try:
                res = client.generate(model=OLLAMA_LLM_MODEL, prompt=prompt)
            except Exception:
                try:
                    res = client.chat(model=OLLAMA_LLM_MODEL, prompt=prompt)
                except Exception:
                    res = None

            if res is None:
                return prompt

            # Attempt to extract text content from response object
            text = None
            for attr in ("text", "content", "output", "response"):
                text = getattr(res, attr, None)
                if isinstance(text, str) and text:
                    return text

            # Fallback to string representation
            return str(res)
    except Exception:
        return prompt


def hyde_embedding_for_query(query: str, embed_fn) -> list:
    """Generate hypothetical answer and return its embedding using embed_fn(func that accepts list[str])."""
    hypo = hyde_hypothetical_answer(query)
    embeddings = embed_fn([hypo])
    return embeddings[0] if embeddings else []


__all__ = ["hyde_hypothetical_answer", "hyde_embedding_for_query"]
