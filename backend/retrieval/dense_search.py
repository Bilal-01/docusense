import os
from typing import List, Tuple

try:
    from backend.store.chroma_store import client
except Exception:
    from store.chroma_store import client


def _list_collections() -> List[str]:
    try:
        # Newer chroma may expose list_collections()
        cols = client.list_collections()
        if isinstance(cols, list):
            return [c.get("name") if isinstance(c, dict) else getattr(c, "name", None) for c in cols]
    except Exception:
        pass
    # Fallback: attempt to read sqlite metadata file if present
    try:
        # client._persist_directory is internal; try common attr
        persist_dir = getattr(client, "persist_directory", None)
        if not persist_dir:
            persist_dir = getattr(client, "settings", {}).get("persist_directory") if hasattr(client, "settings") else None
        if persist_dir:
            # list folders
            for name in os.listdir(persist_dir):
                if name == "chroma.sqlite3":
                    continue
            # cannot reliably list collections; return empty
    except Exception:
        pass
    return []


def dense_query_with_embedding(embedding: List[float], top_n: int = 5) -> List[Tuple[str, float]]:
    """Query all collections using provided embedding. Returns list of (id, score).
    If no collections discoverable, returns empty list."""
    results = []
    collections = _list_collections()
    if not collections:
        # Attempt to query default collection names if known
        return []

    for name in collections:
        try:
            col = client.get_collection(name=name)
            qres = col.query(query_embeddings=[embedding], n_results=top_n)
            # qres may contain ids and distances
            ids = qres.get("ids") if isinstance(qres, dict) else getattr(qres, "ids", None)
            distances = qres.get("distances") if isinstance(qres, dict) else getattr(qres, "distances", None)
            if ids:
                # ids is a list of lists per query
                for idx_list, dist_list in zip(ids, distances or []):
                    for cid, score in zip(idx_list, dist_list):
                        results.append((cid, float(score)))
        except Exception:
            continue

    # sort by score descending (higher is better depending on chroma settings)
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    return results


__all__ = ["dense_query_with_embedding"]
