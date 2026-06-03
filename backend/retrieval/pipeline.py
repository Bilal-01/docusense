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

# Optional cross-encoder for reranking
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

_CROSS_ENCODER = None

def _get_cross_encoder():
    global _CROSS_ENCODER
    if CrossEncoder is None:
        return None
    if _CROSS_ENCODER is None:
        try:
            _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception:
            _CROSS_ENCODER = None
    return _CROSS_ENCODER


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
            try:
                # list available collections once
                try:
                    cols = client.list_collections()
                    existing = set(c.get("name") if isinstance(c, dict) else getattr(c, "name", None) for c in cols)
                except Exception:
                    existing = set()

                for doc_prefix, ids in groups.items():
                    col_name = get_collection_name(doc_prefix)
                    col = None
                    if col_name in existing:
                        try:
                            col = client.get_collection(name=col_name)
                        except Exception:
                            try:
                                col = client.get_or_create_collection(name=col_name)
                            except Exception:
                                col = None
                    else:
                        # Collection name based on chunk prefix not found. Try to locate the
                        # collection that contains this chunk id by scanning existing collections.
                        for cand in existing:
                            try:
                                c = client.get_collection(name=cand)
                                # quick probe
                                # debug probe
                                print(f"[pipeline] probing collection {cand} for id {ids[0]}")
                                probe = c.get(ids=[ids[0]])
                                probe_ids = probe.get("ids") if isinstance(probe, dict) else getattr(probe, "ids", None)
                                if probe_ids:
                                    col = c
                                    print(f"[pipeline] found id {ids[0]} in collection {cand}")
                                    break
                            except Exception:
                                continue

                    for single_id in ids:
                        try:
                            if col is None:
                                # nothing we can do for this prefix
                                continue
                            data = col.get(ids=[single_id])
                            # try dict-like access first
                            if isinstance(data, dict):
                                ids_ret = data.get("ids") or []
                                docs = data.get("documents") or []
                                metadatas = data.get("metadatas") or []
                            else:
                                ids_ret = getattr(data, "ids", []) or []
                                docs = getattr(data, "documents", []) or []
                                metadatas = getattr(data, "metadatas", []) or []

                            # normalize nested lists
                            def _first_item(val):
                                if not val:
                                    return None
                                if isinstance(val, list) and isinstance(val[0], list):
                                    return val[0][0] if val[0] else None
                                return val[0] if isinstance(val, list) else val

                            doc_text = _first_item(docs)
                            meta = _first_item(metadatas)
                            # If doc_text is None but ids_ret contains string, attempt to match index
                            if doc_text is None and ids_ret:
                                try:
                                    # flatten ids_ret
                                    flat_ids = ids_ret[0] if isinstance(ids_ret[0], list) else ids_ret
                                    if single_id in flat_ids:
                                        idx = flat_ids.index(single_id)
                                        doc_text = docs[idx] if idx < len(docs) else None
                                        meta = metadatas[idx] if idx < len(metadatas) else None
                                except Exception:
                                    pass

                            res_map[single_id] = {"text": doc_text, "metadata": meta}
                        except Exception:
                            continue
            except Exception:
                return {}

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
            # If metadata missing but we have a fallback text, create minimal metadata
            if metadata is None and text is not None:
                metadata = {"source": "bm25_fallback"}

            out.append({
                "chunk_id": cid,
                "score": rrf_score,
                "text": text,
                "metadata": metadata,
            })
        return out

    # initial enriched top candidates (up to fused_results length, typically 10)
    candidates = enrich(fused_results)

    # Cross-encoder reranking: score (query, chunk_text) pairs and re-sort.
    try:
        ce = _get_cross_encoder()
    except Exception:
        ce = None

    final = candidates
    if ce is not None and candidates:
        # prepare pairs
        pairs = [(query, c.get("text") or "") for c in candidates]
        try:
            scores = ce.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            # sort by cross-encoder score desc and keep top 5
            final = sorted(candidates, key=lambda x: x.get("rerank_score", -1), reverse=True)[:5]
        except Exception:
            # fallback: take top 5 as-is
            final = candidates[:5]
    else:
        # no cross-encoder available: keep top 5 by original RRF score
        final = candidates[:5]

    return {"results": final}


__all__ = ["query_pipeline"]
