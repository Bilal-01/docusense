import os
import pickle
from typing import List, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from backend.ingestion.models import Chunk
except ModuleNotFoundError:
    from ingestion.models import Chunk


# NLTK resources and helpers
_NLTK_READY = False
_STOPWORDS = None
_STEMMER = None


def _ensure_nltk():
    global _NLTK_READY, _STOPWORDS, _STEMMER
    if _NLTK_READY:
        return
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        _STOPWORDS = set(stopwords.words("english"))
        _STEMMER = PorterStemmer()
        _NLTK_READY = True
    except Exception:
        _NLTK_READY = False


def _tokenize(text: str) -> List[str]:
    _ensure_nltk()
    try:
        import nltk
        tokens = nltk.word_tokenize(text.lower())
        if _STOPWORDS is None or _STEMMER is None:
            return [t for t in tokens if t.isalpha()]
        return [
            _STEMMER.stem(t) for t in tokens
            if t.isalpha() and t not in _STOPWORDS
        ]
    except Exception:
        # Fallback simple split
        return [t for t in text.lower().split() if t.isalpha()]


DEFAULT_PERSIST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".bm25")
)


def build_and_persist_bm25(chunks: List[Chunk], persist_dir: Optional[str] = None) -> dict:
    if persist_dir is None:
        persist_dir = DEFAULT_PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)

    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    mapping = [c.chunk_id for c in chunks]

    if BM25Okapi is None:
        # Can't build BM25 without dependency; persist tokenized corpus and mapping
        data = {"tokenized_corpus": tokenized_corpus, "mapping": mapping}
    else:
        bm25 = BM25Okapi(tokenized_corpus)
        # Do not attempt to pickle BM25 object; persist tokenized corpus and mapping
        data = {"tokenized_corpus": tokenized_corpus, "mapping": mapping}

    out_path = os.path.join(persist_dir, "bm25_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    return {"persist_path": out_path, "num_chunks": len(chunks)}


def load_bm25(persist_dir: Optional[str] = None) -> Tuple[Optional[object], List[str]]:
    if persist_dir is None:
        persist_dir = DEFAULT_PERSIST_DIR
    path = os.path.join(persist_dir, "bm25_data.pkl")
    if not os.path.exists(path):
        return None, []
    with open(path, "rb") as f:
        data = pickle.load(f)

    tokenized_corpus = data.get("tokenized_corpus", [])
    mapping = data.get("mapping", [])

    if BM25Okapi is None:
        return None, mapping

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, mapping


def query_bm25(query: str, top_n: int = 5, persist_dir: Optional[str] = None) -> List[Tuple[str, float]]:
    bm25, mapping = load_bm25(persist_dir=persist_dir)
    if bm25 is None:
        return []
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    # get top indices
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [(mapping[i], float(score)) for i, score in ranked]


__all__ = ["build_and_persist_bm25", "load_bm25", "query_bm25"]
