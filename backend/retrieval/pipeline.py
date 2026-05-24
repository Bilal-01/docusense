from typing import List, Dict, Any

from ..ingestion import embedder
from ..store import bm25_store
from .hyde import hyde_embedding_for_query
from .dense_search import dense_query_with_embedding


def query_pipeline(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Run HyDE -> dense retrieval + BM25 retrieval.

    Returns a dict with keys: 'bm25' and 'dense' each a list of (chunk_id, score).
    """
    # BM25 results use the raw query
    bm25_results = bm25_store.query_bm25(query, top_n=top_k)

    # Dense HyDE: generate hypothetical answer and embed it using embedder
    # embedder exposes _embed_texts; use it as embed_fn
    embed_fn = embedder._embed_texts
    hyde_embedding = hyde_embedding_for_query(query, embed_fn)
    dense_results = []
    if hyde_embedding:
        dense_results = dense_query_with_embedding(hyde_embedding, top_n=top_k)

    return {
        "bm25": bm25_results,
        "dense": dense_results,
    }


__all__ = ["query_pipeline"]
