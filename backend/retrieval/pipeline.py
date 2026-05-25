from typing import List, Dict, Any, Tuple

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


def _apply_rrf_fusion(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 60,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Apply Reciprocal Rank Fusion (RRF) to combine dense and sparse retrieval results.
    
    Args:
        bm25_results: List of (chunk_id, bm25_score) tuples
        dense_results: List of (chunk_id, distance/similarity) tuples
        k: RRF constant (default 60)
        top_n: Number of top fused results to return
    
    Returns:
        List of (chunk_id, fused_rrf_score) tuples, sorted by RRF score descending
    """
    # Build rank maps: chunk_id -> rank (1-indexed, higher score = better)
    bm25_rank_map = {cid: idx + 1 for idx, (cid, _) in enumerate(bm25_results)}
    dense_rank_map = {cid: idx + 1 for idx, (cid, _) in enumerate(dense_results)}
    
    # Collect all unique chunk IDs
    all_chunks = set(bm25_rank_map.keys()) | set(dense_rank_map.keys())
    
    # Compute RRF scores
    rrf_scores = {}
    for chunk_id in all_chunks:
        # Get ranks, defaulting to (max_rank + 1) if not in top results
        bm25_rank = bm25_rank_map.get(chunk_id, len(bm25_results) + 1)
        dense_rank = dense_rank_map.get(chunk_id, len(dense_results) + 1)
        
        # RRF formula: score = 1/(k + rank_sparse) + 1/(k + rank_dense)
        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        rrf_scores[chunk_id] = rrf_score
    
    # Sort by RRF score descending and take top_n
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return fused


def query_pipeline(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Run hybrid retrieval with RRF fusion of dense (HyDE + ChromaDB) and sparse (BM25) results.
    
    Retrieves top 20 from both dense and sparse, normalizes to ranks, applies RRF formula,
    and returns top 10 fused results enriched with text and metadata.
    
    Returns a dict with key 'results': list of enriched chunk dicts with chunk_id, score, text, metadata.
    """
    # Retrieve top 20 from both dense and sparse in parallel semantics
    # (actual parallel execution would use asyncio, but for simplicity we run sequentially)
    bm25_results = bm25_store.query_bm25(query, top_n=20)
    
    # Dense HyDE: generate hypothetical answer and embed it using embedder
    embed_fn = embedder._embed_texts
    hyde_embedding = hyde_embedding_for_query(query, embed_fn)
    dense_results = []
    if hyde_embedding:
        dense_results = dense_query_with_embedding(hyde_embedding, top_n=20)
    
    # Apply RRF fusion to get top 10 fused results
    fused_results = _apply_rrf_fusion(bm25_results, dense_results, k=60, top_n=10)
    
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

    fused_ids = [cid for cid, _ in fused_results]
    chunk_map = fetch_chunks(fused_ids) if fused_ids else {}

    # Fallback: load BM25 persisted documents to fill missing texts
    try:
        from backend.store.bm25_store import DEFAULT_PERSIST_DIR
        import pickle, os
        bm25_path = os.path.join(DEFAULT_PERSIST_DIR, "bm25_data.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            bm25_mapping = bm25_data.get("mapping", [])
            bm25_docs = bm25_data.get("documents") or []
            bm25_map = {k: v for k, v in zip(bm25_mapping, bm25_docs)}
        else:
            bm25_map = {}
    except Exception:
        bm25_map = {}

    def enrich(results):
        out = []
        for cid, rrf_score in results:
            info = chunk_map.get(cid, {})
            text = info.get("text")
            metadata = info.get("metadata")
            if not text:
                text = bm25_map.get(cid)
            out.append({
                "chunk_id": cid,
                "score": rrf_score,
                "text": text,
                "metadata": metadata,
            })
        return out

    return {
        "results": enrich(fused_results),
    }


__all__ = ["query_pipeline"]
