import logging
from typing import List, Tuple

try:
    from store.chroma_store import client
except ModuleNotFoundError:
    from backend.store.chroma_store import client

_LOG = logging.getLogger(__name__)


def dense_query_with_embedding(
    embedding: List[float], top_n: int = 20
) -> List[Tuple[str, float]]:
    """
    Query all ChromaDB collections with the given embedding.

    Returns (chunk_id, distance) pairs sorted ascending — lower distance
    means higher similarity. Callers that use RRF must rank accordingly
    (rank 1 = first item in this list = best match).
    """
    try:
        collections = client.list_collections()
    except Exception as e:
        _LOG.error(f"ChromaDB list_collections failed: {e}")
        return []

    results: List[Tuple[str, float]] = []

    for col_info in collections:
        # Handle both object-style (ChromaDB 0.4+) and dict-style responses
        if isinstance(col_info, str):
            name = col_info
        elif isinstance(col_info, dict):
            name = col_info.get("name")
        else:
            name = getattr(col_info, "name", None)

        if not name:
            continue

        try:
            col = client.get_collection(name=name)
            res = col.query(query_embeddings=[embedding], n_results=top_n)
            # ChromaDB always returns list-of-lists (one per query embedding)
            ids = res["ids"][0]
            distances = res["distances"][0]
            results.extend(zip(ids, distances))
        except Exception as e:
            _LOG.warning(f"Query failed on collection '{name}': {e}")

    # Sort ascending: lower distance = more similar = better rank
    results.sort(key=lambda x: x[1])
    return results[:top_n]