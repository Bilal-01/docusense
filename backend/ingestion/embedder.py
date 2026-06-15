import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from .models import Chunk

try:
    from backend.store.chroma_store import get_collection_name, get_or_create_collection
except ModuleNotFoundError:
    from store.chroma_store import get_collection_name, get_or_create_collection

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
_BATCH_SIZE = int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", "100"))


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using Gemini text-embedding-004."""
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        result = genai.embed_content(
            model=_EMBEDDING_MODEL,
            content=batch,
            task_type="retrieval_document",
        )
        embeddings.extend(result["embedding"])
    return embeddings


def embed_chunks(chunks: List[Chunk], doc_id: str) -> dict:
    """Embed chunks and upsert into ChromaDB. Raises on any failure."""
    if not chunks:
        return {"collection_name": get_collection_name(doc_id), "inserted": 0}

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]  # FIX: no text in metadata
    embeddings = _embed_texts(texts)

    collection_name = get_collection_name(doc_id)
    collection = get_or_create_collection(collection_name, metadata={"document_id": doc_id})

    # FIX: no try/except — let failures propagate so the caller knows the upsert failed
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return {"collection_name": collection_name, "inserted": len(chunks)}