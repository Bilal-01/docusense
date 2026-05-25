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


def build_and_persist_bm25(chunks: Optional[List[Chunk]] = None, persist_dir: Optional[str] = None) -> dict:
    """Build BM25 over provided chunks or, if None, over all chunks in Chroma.

    Returns metadata about persisted index.
    """
    if persist_dir is None:
        persist_dir = DEFAULT_PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)

    # If no chunks provided, attempt to read all chunks from Chroma
    if chunks is None:
        token_texts = []
        mapping = []
        try:
            try:
                from backend.store.chroma_store import client
            except Exception:
                from store.chroma_store import client

            # Try to list collections
            try:
                cols = client.list_collections()
            except Exception:
                cols = []

            col_names = []
            for c in cols:
                if isinstance(c, dict):
                    name = c.get("name")
                else:
                    name = getattr(c, "name", None)
                if name:
                    col_names.append(name)

            for name in col_names:
                try:
                    col = client.get_collection(name=name)
                    data = col.get()
                    ids = data.get("ids", []) if isinstance(data, dict) else getattr(data, "ids", [])
                    docs = data.get("documents", []) if isinstance(data, dict) else getattr(data, "documents", [])
                    for doc_id, doc_text in zip(ids, docs):
                        token_texts.append(_tokenize(doc_text))
                        mapping.append(doc_id)
                        # collect raw docs for persistence
                        try:
                            docs_list.append(doc_text)
                        except NameError:
                            docs_list = [doc_text]
                except Exception:
                    continue

            tokenized_corpus = token_texts
            # ensure docs_list exists
            try:
                docs = docs_list
            except NameError:
                docs = None
        except Exception:
            tokenized_corpus = []
            mapping = []
    else:
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        mapping = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]

    data = {"tokenized_corpus": tokenized_corpus, "mapping": mapping, "documents": docs}

    out_path = os.path.join(persist_dir, "bm25_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    return {"persist_path": out_path, "num_chunks": len(mapping)}


def load_bm25(persist_dir: Optional[str] = None) -> Tuple[Optional[object], List[str], List[List[str]]]:
    if persist_dir is None:
        persist_dir = DEFAULT_PERSIST_DIR
    path = os.path.join(persist_dir, "bm25_data.pkl")
    if not os.path.exists(path):
        return None, [], []
    with open(path, "rb") as f:
        data = pickle.load(f)

    tokenized_corpus = data.get("tokenized_corpus", [])
    mapping = data.get("mapping", [])
    documents = data.get("documents")

    if BM25Okapi is None:
        return None, mapping, tokenized_corpus, documents

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, mapping, tokenized_corpus, documents


def query_bm25(query: str, top_n: int = 5, persist_dir: Optional[str] = None) -> List[Tuple[str, float]]:
    bm25, mapping, tokenized_corpus, documents = load_bm25(persist_dir=persist_dir)
    if bm25 is not None:
        q_tokens = _tokenize(query)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [(mapping[i], float(score)) for i, score in ranked]

    # Fallback: compute simple overlap scores if tokenized_corpus is available
    if not tokenized_corpus or not mapping:
        return []

    q_tokens = set(_tokenize(query))
    scores = []
    for tokens in tokenized_corpus:
        if not tokens:
            scores.append(0.0)
            continue
        # simple overlap score
        overlap = len(q_tokens.intersection(tokens))
        scores.append(float(overlap))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [(mapping[i], float(score)) for i, score in ranked]


__all__ = ["build_and_persist_bm25", "load_bm25", "query_bm25"]
