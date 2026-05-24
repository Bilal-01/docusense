import tempfile
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

try:
    from backend.ingestion import parse_document, chunk_text_blocks, embed_chunks
except ModuleNotFoundError:
    from ingestion import parse_document, chunk_text_blocks, embed_chunks

try:
    from backend.store.bm25_store import build_and_persist_bm25
except ModuleNotFoundError:
    from store.bm25_store import build_and_persist_bm25

app = FastAPI(
    title="DocuSense API",
    description="Document ingestion and parsing API",
    version="0.1.0"
)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> list[dict]:
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

        # Embed and persist chunks to ChromaDB
        doc_id = os.path.splitext(file.filename)[0]
        embed_chunks(chunks, doc_id)

        # Build and persist BM25 index for the uploaded chunks
        try:
            build_and_persist_bm25(chunks)
        except Exception:
            pass

        # Convert Chunk objects to dictionaries
        result = [chunk.to_dict() for chunk in chunks]

        return result

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
