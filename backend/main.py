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
async def query_endpoint_post(request: Request):
    """Query documents using hybrid retrieval (dense + sparse with RRF fusion).
    
    Returns the top 10 results fused using Reciprocal Rank Fusion (RRF),
    which combines dense semantic search (HyDE + ChromaDB) and sparse BM25 keyword search.
    Supports query parameter 'q', JSON body with 'q' field, or form data with 'q' field.
    
    Returns:
        JSON with 'results' key containing list of ranked chunks with:
        - chunk_id: Unique chunk identifier
        - score: RRF fused score (higher is better)
        - text: Chunk text content
        - metadata: Chunk metadata dict
    """
    try:
        # expect JSON body with `question` and `doc_id`
        body = {}
        try:
            body = await request.json()
        except Exception:
            try:
                form = await request.form()
                body = dict(form)
            except Exception:
                pass

        question = body.get("question") or body.get("q")
        doc_id = body.get("doc_id") or body.get("docId")
        if not question:
            raise HTTPException(status_code=400, detail="Missing 'question' in request body")
        # Run retrieval pipeline to get candidate chunks
        res = query_pipeline(question, top_k=5)
        candidates = res.get("results", [])

        # select top 5 (pipeline already returns top 5 after rerank)
        top_chunks = candidates[:5]

        # prepare chunk_inputs with full metadata for LLM polisher
        chunk_inputs = []
        for c in top_chunks:
            cid = c.get("chunk_id")
            text = c.get("text")
            meta = c.get("metadata") or {}
            chunk_inputs.append({
                "chunk_id": cid,
                "text": text,
                "page": meta.get("page_number") or meta.get("page"),
                "char_start": meta.get("char_start"),
                "char_end": meta.get("char_end"),
                "score": c.get("score"),
            })

        # Use LLM to generate polished answer with source tracking
        try:
            try:
                from backend.retrieval.llm_polisher import polish_answer_with_sources
            except Exception:
                from retrieval.llm_polisher import polish_answer_with_sources

            polished_answer, source_tracking = polish_answer_with_sources(question, chunk_inputs)
        except Exception as e:
            # Fallback if LLM fails: concatenate first 2 chunks
            polished_answer = " ".join([c.get("text", "")[:50] for c in chunk_inputs[:2]])
            source_tracking = []
            print(f"[main] LLM polishing failed: {e}")

        # Compute ragas scores for faithfulness and relevancy
        try:
            try:
                from backend.retrieval.attribution import compute_ragas_faithfulness
            except Exception:
                from retrieval.attribution import compute_ragas_faithfulness

            answer_text = polished_answer
            contexts = [c.get("text") for c in chunk_inputs]
            ragas_res = compute_ragas_faithfulness(question, answer_text, contexts)
        except Exception:
            ragas_res = None

        # Parse ragas results
        faith_val = None
        rel_val = None
        if isinstance(ragas_res, (tuple, list)):
            try:
                faith_val, rel_val = ragas_res[0], ragas_res[1]
            except Exception:
                faith_val, rel_val = None, None
        elif isinstance(ragas_res, dict):
            faith_val = ragas_res.get("faithfulness") or ragas_res.get("faithfulness_score")
            rel_val = ragas_res.get("relevancy") or ragas_res.get("answer_relevancy")

        # Build answer_chunks from source tracking
        answer_chunks = []
        if source_tracking:
            # Group source tracking by unique chunk_id and build answer_chunks
            chunk_tracking_map = {}
            for s in source_tracking:
                cid = s.get("chunk_id")
                if cid not in chunk_tracking_map:
                    chunk_tracking_map[cid] = {
                        "char_start": s.get("char_start"),
                        "char_end": s.get("char_end"),
                        "source_refs": [{
                            "chunk_id": cid,
                            "page": s.get("page"),
                            "char_start": s.get("original_char_start"),
                            "char_end": s.get("original_char_end"),
                        }]
                    }

            for cid, tracking in chunk_tracking_map.items():
                answer_chunks.append({
                    "text": polished_answer[tracking["char_start"]:tracking["char_end"]],
                    "source_refs": tracking["source_refs"],
                    "answer_char_start": tracking["char_start"],
                    "answer_char_end": tracking["char_end"],
                })
        else:
            # Fallback: include whole answer with first chunk as reference
            first_chunk = chunk_inputs[0] if chunk_inputs else {}
            answer_chunks.append({
                "text": polished_answer,
                "source_refs": [{
                    "chunk_id": first_chunk.get("chunk_id", "unknown"),
                    "page": first_chunk.get("page"),
                    "char_start": first_chunk.get("char_start"),
                    "char_end": first_chunk.get("char_end"),
                }]
            })

        out = {
            "answer": polished_answer,
            "faithfulness_score": faith_val,
            "answer_relevancy": rel_val,
            "source_chunks": answer_chunks,
        }
        return JSONResponse(status_code=200, content=out)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
