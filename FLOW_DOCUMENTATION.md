# DocuSense Query Pipeline - Fixed Flow

## Overview
The system now properly implements a complete LLM-driven query flow with source tracking, without any hardcoded responses.

## Request Flow

### 1. `/upload` Endpoint
**Purpose**: Ingest documents and prepare for retrieval

**What happens**:
- Document parsed into text blocks (PDF, DOCX, TXT)
- Text semantically chunked (intelligent boundaries, not naive)
- Chunks embedded using **Ollama** embeddings (nomic-embed-text)
- Embeddings persisted to **ChromaDB** (dense index)
- Chunks also indexed in **BM25** (sparse index)
- Document metadata stored in SQLite DB

**No changes needed**: Already working correctly

---

### 2. `/query` Endpoint (POST)
**Purpose**: Answer questions using retrieved & polished information

**Request**:
```json
{
  "question": "Does the resume have Langchain skill?",
  "doc_id": "resume"
}
```

**Processing Flow**:

#### Step 1: Hybrid Retrieval (pipeline.py)
```
retrieval/pipeline.py:query_pipeline()
├── BM25 sparse search      → top 20 results
├── HyDE dense search       → top 20 results (semantic)
│   └── Uses Ollama embeddings for HyDE
├── Reciprocal Rank Fusion  → merge & rank
└── Cross-Encoder rerank    → top 5 final candidates
```

**Key Classes**:
- `BM25Okapi` from `store/bm25_store.py` - keyword-based retrieval
- `ChromaDB` from `store/chroma_store.py` - dense similarity search
- `CrossEncoder` - semantic relevance scoring (if installed)

#### Step 2: LLM Polishing with Source Tracking (llm_polisher.py)
```python
polish_answer_with_sources(question, chunks) 
→ (polished_answer, source_tracking)
```

**How it works**:
1. Format chunks with labels: `[Source 1] text...`, `[Source 2] text...`
2. Send to Ollama LLM with prompt:
   ```
   Question: {question}
   Relevant sources: [labeled chunks]
   Instructions:
   - Generate concise answer using ONLY source info
   - Cite sources like [Source 1] or [Source 1, 2]
   - Paraphrase naturally
   ```
3. LLM returns polished conversational answer with citations
4. Parse `[Source N]` citations to map to chunk indices
5. Return: (polished_answer, source_tracking_list)

**Source Tracking Output**:
```python
[
  {
    "chunk_id": "doc1_p1_c0",
    "char_start": 0,           # Position in polished answer
    "char_end": 150,
    "page": 1,                 # Original document page
    "original_char_start": 50, # Position in original document
    "original_char_end": 200,
    "score": 0.95,             # Retrieval score
  },
  ...
]
```

#### Step 3: Compute Ragas Scores (attribution.py)
```python
compute_ragas_faithfulness(question, answer, contexts)
→ (faithfulness_score, answer_relevancy)
```

Uses local Ollama LLM to score:
- **Faithfulness**: How much is supported by source chunks? (0-1)
- **Answer Relevancy**: How well does answer address question? (0-1)

#### Step 4: Build Response
```python
response = {
  "answer": "Yes, the resume mentions LangChain as a technical skill in the LLM & Agents section.",
  "faithfulness_score": 0.98,
  "answer_relevancy": 0.92,
  "source_chunks": [
    {
      "text": "Yes, the resume mentions LangChain as a technical skill [Source 1]",
      "source_refs": [
        {
          "chunk_id": "resume_p1_c0",
          "page": 1,
          "char_start": 150,      # In original document
          "char_end": 260,
        }
      ],
      "answer_char_start": 0,     # In polished_answer
      "answer_char_end": 90,
    }
  ]
}
```

---

## Key Components Modified

### New Files

1. **`backend/retrieval/llm_polisher.py`**
   - `polish_answer_with_sources()` - Main polishing & tracking function
   - `call_ollama_llm()` - Wrapper for Ollama client/CLI
   - `_track_sources()` - Parse citations and map to char positions

### Modified Files

1. **`backend/main.py`** (`/query` POST)
   - Removed hardcoded `build_attribution_prompt()` + `generate_attributed_answer()` pipeline
   - Removed hardcoded `polish_answer_with_ollama()` fallback
   - Now uses `polish_answer_with_sources()` for LLM-driven polishing
   - Builds response with proper source tracking from `source_tracking` output

2. **`backend/ingestion/embedder.py`**
   - Enhanced `_embed_texts()` to try Ollama Client first, then fallback to CLI
   - Better error handling and logging

---

## LLM Integration

### Ollama Models Used
- **Embeddings**: `nomic-embed-text:latest` (via `OLLAMA_EMBEDDING_MODEL`)
- **Generation**: `llama2` (or via `OLLAMA_GEN_MODEL`) - used for polishing & scoring

### Fallback Behavior
1. Try Ollama Client library (`ollama.Client`)
2. Fallback to Ollama CLI (`ollama run ...`)
3. If all fail: Use heuristic keyword detection (returns raw text)

### Environment Variables
```bash
OLLAMA_HOST=http://127.0.0.1:11434          # Ollama server URL
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_GEN_MODEL=llama2
```

---

## No Hardcoded Responses

✅ **Everything flows through the LLM**:
- Embeddings: Generated via Ollama (not pretrained)
- BM25: Computed from actual document text
- Cross-encoder: Scores actual chunk pairs
- Final answer: Generated by LLM, not templated
- Source tracking: Parsed from LLM citations, not guessed

---

## Testing

### Quick Test
```bash
cd backend
python test_e2e_flow.py
```

### Full API Test
```bash
# Start server
uvicorn main:app --reload --port 8000

# Upload document
curl -X POST http://localhost:8000/upload \
  -F "file=@resume.pdf"

# Query with LLM polishing
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Does the resume have Langchain skill?","doc_id":"resume"}'
```

---

## Response Example

```json
{
  "answer": "Yes, the resume includes LangChain as a core technical skill in the LLM & Agents section, alongside OpenAI API, Claude, and other advanced frameworks.",
  "faithfulness_score": 0.98,
  "answer_relevancy": 0.92,
  "source_chunks": [
    {
      "text": "Yes, the resume includes LangChain as a core technical skill [Source 1]",
      "source_refs": [
        {
          "chunk_id": "resume_p1_c0",
          "page": 1,
          "char_start": 150,
          "char_end": 260
        }
      ],
      "answer_char_start": 0,
      "answer_char_end": 72
    }
  ]
}
```

---

## Architecture Diagram

```
┌─────────────┐
│   Document  │
└──────┬──────┘
       │
       ▼
   /upload
       │
       ├─► [Parse] ─► Text Blocks
       │
       ├─► [Chunk] ─► Semantic Chunks
       │
       ├─► [Embed] ─► Ollama Embeddings
       │                   │
       │                   ├─► ChromaDB (dense)
       │                   └─► BM25 (sparse)
       │
       └─► SQLite (metadata)

┌─────────────┐
│  Question   │
└──────┬──────┘
       │
       ▼
   /query
       │
       ├─► [BM25 Search] ──┐
       │                    ├─► [RRF Fusion]
       ├─► [Dense Search] ──┤
       │   (HyDE + ChromaDB) │
       │                    ├─► [Cross-Encoder]
       │                    │
       └────────────────────┤
                            ▼
                    Best 5 Chunks
                            │
                            ├─► [Format]
                            │
                            ▼
                    [LLM Polisher]
                    (Ollama)
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
            Polished Answer   Source Tracking
                    │                │
                    ├─► [Ragas] ─────┤
                    │   Scoring      │
                    │                │
                    └────────┬───────┘
                             │
                             ▼
                      JSON Response
                   (with char positions)
```

---

## Summary

✅ **Fixed**: All hardcoded responses removed  
✅ **Now**: Complete LLM-driven flow with source attribution  
✅ **Tracking**: Exact character positions in both answer and original document  
✅ **Scoring**: Faithfulness and relevancy via Ragas  
✅ **DB**: All data fetched from ChromaDB and BM25 indices, not templates  
