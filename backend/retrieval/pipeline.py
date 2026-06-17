import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from store import bm25_store
    from store.chroma_store import client as chroma_client, get_collection_name
except ModuleNotFoundError:
    from backend.store import bm25_store
    from backend.store.chroma_store import client as chroma_client, get_collection_name

from .hyde import hyde_embedding_for_query
from .dense_search import dense_query_with_embedding
from .generator import generate_answer
from .attribution import (
    build_attribution_prompt,
    parse_attributed_response,
    compute_ragas_faithfulness,
)

_LOG = logging.getLogger(__name__)

# Sentinel avoids retrying a failed CrossEncoder load on every request
_CROSS_ENCODER = None
_CE_FAILED = object()


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is _CE_FAILED:
        return None
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        _LOG.info("CrossEncoder loaded.")
    except Exception as e:
        _LOG.warning(f"CrossEncoder unavailable — reranking disabled: {e}")
        _CROSS_ENCODER = _CE_FAILED
    return _CROSS_ENCODER if _CROSS_ENCODER is not _CE_FAILED else None


def _apply_rrf_fusion(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    bm25_ranks = {cid: i + 1 for i, (cid, _) in enumerate(bm25_results)}
    dense_ranks = {cid: i + 1 for i, (cid, _) in enumerate(dense_results)}
    all_ids = set(bm25_ranks) | set(dense_ranks)

    scores = {
        cid: 1 / (k + bm25_ranks.get(cid, len(bm25_results) + 1))
             + 1 / (k + dense_ranks.get(cid, len(dense_results) + 1))
        for cid in all_ids
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def _fetch_chunks(chunk_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch chunk texts and metadata from ChromaDB.
    Groups chunk IDs by collection name (derived from the doc-name prefix
    before '_p'), then does a single get() per collection.
    """
    groups: Dict[str, List[str]] = {}
    for cid in chunk_ids:
        col_name = get_collection_name(cid.split("_p")[0])
        groups.setdefault(col_name, []).append(cid)

    result: Dict[str, Dict] = {}
    for col_name, ids in groups.items():
        try:
            col = chroma_client.get_collection(name=col_name)
            data = col.get(ids=ids)
            for i, cid in enumerate(data["ids"]):
                result[cid] = {
                    "text": data["documents"][i],
                    "metadata": data["metadatas"][i],
                }
        except Exception as e:
            _LOG.warning(f"Fetch failed for collection '{col_name}': {e}")

    return result

cols = chroma_client.list_collections()

def query_pipeline(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full RAG pipeline:
      BM25 + HyDE dense retrieval
      → RRF fusion
      → cross-encoder reranking
      → LLM generation with chunk-level attribution
      → RAGAS faithfulness evaluation

    Returns a dict with: answer, raw_answer, faithfulness_score,
    answer_relevancy, source_chunks.
    """
    # 1. Retrieval
    bm25_results = bm25_store.query_bm25(question, top_n=20)
    hyde_emb = hyde_embedding_for_query(question)
    dense_results = dense_query_with_embedding(hyde_emb, top_n=20) if hyde_emb else []

    # 2. RRF fusion
    fused = _apply_rrf_fusion(bm25_results, dense_results, k=60, top_n=10)

    # 3. Hydrate chunk IDs with text and metadata
    chunk_map = _fetch_chunks([cid for cid, _ in fused])
    candidates = []
    for cid, rrf_score in fused:
        info = chunk_map.get(cid)
        if not info:
            _LOG.warning(f"Chunk '{cid}' not found in ChromaDB — skipping")
            continue
        candidates.append({
            "chunk_id": cid,
            "rrf_score": rrf_score,
            "text": info["text"],
            "metadata": info["metadata"],
        })

    # 4. Cross-encoder reranking
    ce = _get_cross_encoder()
    if ce and candidates:
        try:
            scores = ce.predict([(question, c["text"]) for c in candidates])
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        except Exception as e:
            _LOG.warning(f"Reranking failed, using RRF order: {e}")

    top_chunks = candidates[:top_k]

    # 5. Generate attributed answer
    prompt = build_attribution_prompt(top_chunks, question)
    raw_answer = generate_answer(prompt["system"], prompt["user"])
    attributed = parse_attributed_response(raw_answer)

    # 6. RAGAS evaluation
    faith, relevancy = compute_ragas_faithfulness(
        question, raw_answer, [c["text"] for c in top_chunks]
    )

    return {
        "answer": attributed,
        "raw_answer": raw_answer,
        "faithfulness_score": faith,
        "answer_relevancy": relevancy,
        "source_chunks": top_chunks,
    }