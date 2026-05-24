import os
from typing import List, Sequence

import ollama
from dotenv import load_dotenv

from .models import Chunk

try:
    from backend.store.chroma_store import get_collection_name, get_or_create_collection, persist
except ModuleNotFoundError:
    from store.chroma_store import get_collection_name, get_or_create_collection, persist

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_BATCH_SIZE = int(os.getenv("OLLAMA_EMBEDDING_BATCH_SIZE", "100"))


def _batch_items(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    if not OLLAMA_HOST:
        raise EnvironmentError("OLLAMA_HOST must be configured for Ollama embedding generation")

    embeddings = []
    with ollama.Client(host=OLLAMA_HOST) as client:
        for batch in _batch_items(texts, batch_size=OLLAMA_BATCH_SIZE):
            response = client.embed(model=OLLAMA_MODEL, input=list(batch))
            embeddings.extend(response.embeddings)

    return embeddings


def embed_chunks(chunks: List[Chunk], doc_id: str) -> dict:
    """Embed chunks with Ollama and write them to persistent ChromaDB."""
    if not chunks:
        return {"collection_name": get_collection_name(doc_id), "inserted": 0}

    collection_name = get_collection_name(doc_id)
    collection = get_or_create_collection(
        name=collection_name,
        metadata={"document_id": doc_id}
    )

    texts = [chunk.text for chunk in chunks]
    embeddings = _embed_texts(texts)
    ids = [chunk.chunk_id for chunk in chunks]
    metadatas = [chunk.to_dict() for chunk in chunks]

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    persist()
    return {"collection_name": collection_name, "inserted": len(chunks)}
