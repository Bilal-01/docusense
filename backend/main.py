import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from ingestion import parse_document, TextBlock

app = FastAPI(
    title="DocuSense API",
    description="Document ingestion and parsing API",
    version="0.1.0"
)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> list[dict]:
    """
    Upload and parse a document (PDF, DOCX, or TXT).

    Returns a list of TextBlock objects with extracted text and metadata.
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

        # Parse document
        blocks = parse_document(temp_path, file_type=file_ext.lstrip('.'))

        # Convert TextBlock objects to dictionaries
        result = [block.to_dict() for block in blocks]

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IOError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
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
