# Quick Start Guide - DocuSense API

## Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Ensure Ollama is Running
```bash
# Start Ollama service (if not already running)
ollama serve

# In another terminal, verify models are available
ollama list
# Should show: nomic-embed-text and llama2 (or your configured models)
```

### 3. Environment Variables (Optional)
```bash
# Create .env file in project root (if not exists)
cat > .env << EOF
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_GEN_MODEL=llama2
EOF
```

## Running the API

### Start Server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{"status": "ok"}
```

---

### 2. Upload Document

**Endpoint**: `POST /upload`

**Supported formats**: PDF, DOCX, TXT

**Request**:
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@resume.pdf"
```

**Response**:
```json
{
  "doc_id": "resume",
  "num_chunks": 45,
  "status": "uploaded"
}
```

**What happens**:
- Document is parsed into semantic chunks
- Each chunk is embedded using Ollama
- Chunks indexed in ChromaDB and BM25
- Metadata stored in SQLite

---

### 3. Query Documents

**Endpoint**: `POST /query`

**Request**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Does the resume have Langchain skill?",
    "doc_id": "resume"
  }'
```

**Response** (Full Structure):
```json
{
  "answer": "Yes, the resume includes LangChain as a core skill in the LLM & Agents section.",
  "faithfulness_score": 0.98,
  "answer_relevancy": 0.92,
  "source_chunks": [
    {
      "text": "Yes, the resume includes LangChain [Source 1]",
      "source_refs": [
        {
          "chunk_id": "resume_p1_c0",
          "page": 1,
          "char_start": 150,
          "char_end": 260
        }
      ],
      "answer_char_start": 0,
      "answer_char_end": 45
    }
  ]
}
```

**Field Descriptions**:
- `answer`: LLM-polished, conversational response (NO hardcoding)
- `faithfulness_score`: How well answer is supported by sources (0-1)
- `answer_relevancy`: How well answer addresses the question (0-1)
- `source_chunks`: List of text segments with source attribution
  - `text`: The portion of the answer
  - `source_refs`: Which original chunks contributed
    - `chunk_id`: Unique identifier
    - `page`: Original page number
    - `char_start`, `char_end`: Character positions in original document
  - `answer_char_start`, `answer_char_end`: Positions in the polished answer

---

## Flow Visualization

### What happens when you query:

```
Question: "Does the resume have Langchain skill?"
   │
   ▼
1. RETRIEVE (Hybrid)
   ├─ BM25 search    → "langchain" keyword matches
   ├─ Dense search   → semantic similarity via embeddings
   └─ Merge with RRF → ranked top results

2. RERANK (Cross-Encoder)
   └─ Score (question, chunk) pairs → top 5

3. POLISH (LLM)
   ├─ Format chunks with labels [Source 1], [Source 2], etc.
   ├─ Send to Ollama with prompt
   ├─ LLM generates: "Yes, the resume has LangChain [Source 1]"
   └─ Parse citations → track which source each part came from

4. SCORE (Ragas)
   ├─ Faithfulness: How much is supported? → 0.98
   └─ Relevancy: Answers the question? → 0.92

5. FORMAT RESPONSE
   └─ Return polished answer + source tracking + scores
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ollama'"
```bash
pip install ollama
```

### "Connection refused" to Ollama
```bash
# Make sure Ollama is running
ollama serve

# Check it's accessible
curl http://127.0.0.1:11434/api/tags
```

### Embeddings failing
```bash
# Ensure embeddings model is available
ollama pull nomic-embed-text

# Verify
ollama list | grep nomic
```

### Slow generation
- LLMs are slow by nature (30s-2min per query is normal)
- Use a faster model: `llama2-uncensored` or `neural-chat`
- Adjust `OLLAMA_GEN_MODEL` environment variable

### Character position issues
- Positions are 0-indexed (char 0 = first character)
- Use them to highlight/extract the exact text from source
- Example: `source_text[150:260]` gives the exact quoted portion

---

## Example: Extract cited text

Once you have the response, extract the exact quoted portion:

```python
response = {  # from API
    "answer": "Yes, the resume has LangChain...",
    "source_chunks": [
        {
            "text": "Yes, the resume has LangChain [Source 1]",
            "source_refs": [
                {
                    "chunk_id": "resume_p1_c0",
                    "char_start": 150,
                    "char_end": 260,
                }
            ]
        }
    ]
}

# Read original document
with open("resume.pdf", "rb") as f:
    # (Parse to get text)
    original_text = "...full document text..."

# Extract the cited portion
ref = response["source_chunks"][0]["source_refs"][0]
cited_text = original_text[ref["char_start"]:ref["char_end"]]
print(f"Cited from document: {cited_text}")
```

---

## Performance Tips

1. **First query is slow** (LLM warms up) - subsequent queries faster
2. **Use smaller models** for faster responses
3. **Batch uploads** - many documents at once OK
4. **Keep Ollama running** - don't restart frequently

---

## File Structure

```
docusense/
├── backend/
│   ├── main.py                 # FastAPI endpoints
│   ├── requirements.txt         # Dependencies
│   ├── ingestion/              # Document parsing
│   │   ├── chunker.py         # Semantic chunking
│   │   ├── embedder.py        # Embedding generation
│   │   └── parser.py          # PDF/DOCX/TXT parsing
│   ├── retrieval/             # Query pipeline
│   │   ├── pipeline.py        # Hybrid search + RRF
│   │   ├── generator.py       # Answer generation (legacy)
│   │   └── attribution.py     # Ragas scoring
│   └── store/                 # Data storage
│       ├── chroma_store.py    # ChromaDB dense index
│       ├── bm25_store.py      # BM25 sparse index
│       └── db.py              # SQLite metadata
└── FLOW_DOCUMENTATION.md       # This file
```

---

## Next Steps

1. **Upload a resume** - `POST /upload`
2. **Query it** - `POST /query`
3. **Check response** - Verify source_chunks with exact char positions
4. **Iterate** - Adjust prompts/models as needed

Happy querying! 🚀
