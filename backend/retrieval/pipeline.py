from typing import List, Dict, Any

try:
    from backend.ingestion import embedder
    from backend.store import bm25_store
except Exception:
    from ingestion import embedder
    from store import bm25_store

try:
    from .hyde import hyde_embedding_for_query
    from .dense_search import dense_query_with_embedding
except Exception:
    from hyde import hyde_embedding_for_query
    from dense_search import dense_query_with_embedding


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

    # Enrich results with text and metadata from Chroma
    try:
        try:
            from backend.store.chroma_store import client, get_collection_name
        except Exception:
            from store.chroma_store import client, get_collection_name

        def fetch_chunks(chunk_ids):
            # group by doc id prefix (before _p)
            groups = {}
            for cid in chunk_ids:
                doc_prefix = cid.split("_p")[0]
                groups.setdefault(doc_prefix, []).append(cid)

            res_map = {}
            for doc_prefix, ids in groups.items():
                try:
                    col_name = get_collection_name(doc_prefix)
                    col = client.get_or_create_collection(name=col_name)
                    data = col.get(ids=ids)

                    # normalize possible nested return shapes
                    def _unwrap(val):
                        if val is None:
                            return []
                        if isinstance(val, list):
                            # If nested lists (per-query), flatten
                            if val and isinstance(val[0], list):
                                return [item for sub in val for item in sub]
                            return val
                        if isinstance(val, dict):
                            return [val]
                        return [val]

                    ids_ret = _unwrap(data.get("ids") if isinstance(data, dict) else getattr(data, "ids", None))
                    docs = _unwrap(data.get("documents") if isinstance(data, dict) else getattr(data, "documents", None))
                    metadatas = _unwrap(data.get("metadatas") if isinstance(data, dict) else getattr(data, "metadatas", None))

                    # If ids_ret is list of lists (edge case), it's already flattened by _unwrap
                    for i, cid in enumerate(ids_ret):
                        res_map[cid] = {
                            "text": docs[i] if i < len(docs) else None,
                            "metadata": metadatas[i] if metadatas and i < len(metadatas) else None,
                        }
                except Exception:
                    continue
            return res_map

    except Exception:
        fetch_chunks = lambda ids: {}

    all_ids = [cid for cid, _ in bm25_results] + [cid for cid, _ in dense_results]
    unique_ids = list(dict.fromkeys(all_ids))
    chunk_map = fetch_chunks(unique_ids) if unique_ids else {}

    def enrich(results):
        out = []
        for cid, score in results:
            info = chunk_map.get(cid, {})
            out.append({
                "chunk_id": cid,
                "score": score,
                "text": info.get("text"),
                "metadata": info.get("metadata"),
            })
        return out

    return {
        "bm25": enrich(bm25_results),
        "dense": enrich(dense_results),
    }


__all__ = ["query_pipeline"]
