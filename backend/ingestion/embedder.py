import os
from typing import List

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from .models import Chunk

try:
    from backend.store.chroma_store import get_collection_name, get_or_create_collection
except ModuleNotFoundError:
    from store.chroma_store import get_collection_name, get_or_create_collection

load_dotenv()

# Loaded once at import time — stays in memory across requests
_MODEL = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5"))


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts locally. normalize_embeddings=True makes L2 == cosine distance."""
    return _MODEL.encode(texts, normalize_embeddings=True).tolist()


def embed_chunks(chunks: List[Chunk], doc_id: str) -> dict:
    """Embed chunks and upsert into ChromaDB. Raises on any failure."""
    if not chunks:
        return {"collection_name": get_collection_name(doc_id), "inserted": 0}

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    embeddings = _embed_texts(texts)

    doc_id_clean = doc_id.rsplit(".", 1)[0]
    collection_name = get_collection_name(doc_id_clean)
    collection = get_or_create_collection(collection_name, metadata={"document_id": doc_id_clean})
    
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    return {"collection_name": collection_name, "inserted": len(chunks)}