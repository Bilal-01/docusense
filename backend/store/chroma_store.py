import os
import re

from chromadb import Client
from chromadb.config import Settings

# Persist ChromaDB storage at the repository root so indexes survive backend restarts.
CHROMA_PERSIST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".chroma")
)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

client = Client(
    settings=Settings(
        is_persistent=True,
        persist_directory=CHROMA_PERSIST_DIR,
    )
)


def sanitize_collection_name(doc_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)
    return sanitized.lower()


def get_collection_name(doc_id: str) -> str:
    return f"docusense_{sanitize_collection_name(doc_id)}"


def get_or_create_collection(name: str, metadata: dict | None = None):
    return client.get_or_create_collection(name=name, metadata=metadata or {})


def persist():
    try:
        # Some chroma client versions expose a `persist` method; others handle
        # persistence automatically. Call it if available, otherwise no-op.
        client.persist()
    except AttributeError:
        return
