import tempfile
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

try:
    from backend.ingestion import parse_document, chunk_text_blocks, embed_chunks
except ModuleNotFoundError:
    from ingestion import parse_document, chunk_text_blocks, embed_chunks

try:
    from backend.store.bm25_store import build_and_persist_bm25
except ModuleNotFoundError:
    from store.bm25_store import build_and_persist_bm25

try:
    from backend.retrieval.pipeline import query_pipeline
except ModuleNotFoundError:
    from retrieval.pipeline import query_pipeline

app = FastAPI(
    title="DocuSense API",
    description="Document ingestion and parsing API",
    version="0.1.0"
)

# initialize DB
try:
    from backend.store.db import init_db
except Exception:
    from store.db import init_db

try:
    init_db()
except Exception:
    pass


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None) -> dict:
    """
    Upload, parse, and semantically chunk a document (PDF, DOCX, or TXT).

    Returns a list of Chunk objects with semantic boundaries, position metadata,
    and deterministic chunk IDs for deduplication and retrieval.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file to temporary location
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Parse document into text blocks
        blocks = parse_document(temp_path, file_type=file_ext.lstrip('.'))

        # Semantically chunk text blocks
        chunks = chunk_text_blocks(blocks)

        # Embed and persist chunks to ChromaDB in background to speed response
        doc_id = os.path.splitext(file.filename)[0]
        if background_tasks is not None:
            background_tasks.add_task(embed_chunks, chunks, doc_id)
            background_tasks.add_task(build_and_persist_bm25, chunks)
        else:
            embed_chunks(chunks, doc_id)
            try:
                build_and_persist_bm25(chunks)
            except Exception:
                pass

        # persist doc metadata to SQLite
        try:
            from backend.store.db import get_session, Document
        except Exception:
            from store.db import get_session, Document

        try:
            sess = get_session()
            doc = sess.query(Document).filter(Document.doc_id == doc_id).first()
            if not doc:
                doc = Document(doc_id=doc_id, filename=file.filename, num_chunks=len(chunks))
                sess.add(doc)
            else:
                doc.num_chunks = len(chunks)
            sess.commit()
            sess.close()
        except Exception:
            pass

        return {"doc_id": doc_id, "num_chunks": len(chunks), "status": "uploaded"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IOError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/query")
async def query_endpoint(q: str):
    """Query documents using hybrid retrieval (dense + sparse with RRF fusion).
    
    Returns the top 10 results fused using Reciprocal Rank Fusion (RRF),
    which combines dense semantic search (HyDE + ChromaDB) and sparse BM25 keyword search.
    
    Args:
        q: Query string
        
    Returns:
        JSON with 'results' key containing list of ranked chunks with:
        - chunk_id: Unique chunk identifier
        - score: RRF fused score (higher is better)
        - text: Chunk text content
        - metadata: Chunk metadata dict
    """
    try:
        res = query_pipeline(q, top_k=5)
        return JSONResponse(status_code=200, content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(request: Request):
    body = await request.json()
    question = body.get("question")
    if not question:
        raise HTTPException(400, "Missing 'question'")
    try:
        return JSONResponse(query_pipeline(question, top_k=5))
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
