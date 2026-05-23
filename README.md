# DocuSense

A production-grade **attribution-first document Q&A engine** powered by RAG. Every sentence in the AI's answer is linked back to the exact chunk it came from — with a faithfulness score.

> Not a chatbot. A precision document retrieval and analysis system.

## What is DocuSense?

DocuSense combines semantic search, dense & sparse indexing, cross-encoder reranking, and LLM-powered generation into a single pipeline where:

- **Every claim** the AI makes is traced back to source document chunks
- **Faithfulness scores** validate that answers reflect the documents, not hallucinations
- **Hybrid retrieval** uses both semantic embeddings and BM25 for comprehensive coverage
- **Interactive attribution** lets users click through from answer back to source text

## Key Features

- 📄 **Multi-format ingestion**: PDF, DOCX with structure preservation
- 🔍 **Hybrid search**: Semantic embeddings + BM25 keyword matching + RRF fusion
- 🎯 **Semantic chunking**: Meaning-aware splits via LlamaIndex, not fixed-size windows
- 🏆 **Cross-encoder reranking**: Precise relevance scoring before generation
- ✍️ **Chunk-aware generation**: LLM knows which chunks it's citing
- ✅ **Faithfulness evaluation**: RAGAS metrics on every response
- 🎨 **Interactive frontend**: React + PDF viewer with highlight overlays
- 📊 **Attribution UI**: Shows which source chunks support each claim

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Document Parsing | `pdfplumber`, `python-docx` | Best structure preservation |
| Chunking | LlamaIndex SemanticSplitter | Meaning-aware splits |
| Embeddings | `text-embedding-3-small` (OpenAI) | Cost-efficient, high quality |
| Dense Index | ChromaDB | Local, no Docker needed |
| Sparse Index | `rank_bm25` | Keyword recall for exact terms |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Open-source, fast, accurate |
| LLM | GPT-4o or Claude 3.5 Sonnet | Via API |
| Evaluation | RAGAS | Faithfulness + answer relevancy |
| Backend | FastAPI | REST APIs |
| Frontend | React + `react-pdf` | Interactive document viewer |

## Project Structure

```
docusense/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── ingestion/
│   │   ├── parser.py            # PDF/DOCX → raw text + metadata
│   │   ├── chunker.py           # Semantic chunking + chunk ID tagging
│   │   └── embedder.py          # Embed chunks + write to ChromaDB
│   ├── retrieval/
│   │   ├── hybrid.py            # BM25 + semantic search + RRF fusion
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── hyde.py              # HyDE query expansion
│   ├── generation/
│   │   ├── llm.py               # LLM call with chunk-ID-aware prompt
│   │   ├── attribution.py       # Parse response → map claims to chunk IDs
│   │   └── scorer.py            # RAGAS faithfulness evaluation
│   └── store/
│       ├── chroma_store.py      # ChromaDB read/write wrapper
│       └── bm25_store.py        # BM25 index build + search
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DocumentViewer.jsx    # PDF viewer with highlight overlay
│   │   │   ├── ChatWindow.jsx        # Q&A interface
│   │   │   ├── AttributionPanel.jsx  # Shows source chunks per answer
│   │   │   └── FaithfulnessScore.jsx # Score badge per response
│   │   └── App.jsx
└── requirements.txt
```

## Quick Start

*(Build and setup steps coming next)*

## Architecture Overview

**Pipeline Flow:**

1. **Ingestion** → Parse documents → Extract text & metadata
2. **Chunking** → Semantic splits with chunk IDs
3. **Indexing** → Embed to ChromaDB + build BM25 index
4. **Retrieval** → Hybrid search (semantic + keyword) → RRF fusion
5. **Reranking** → Cross-encoder scores top candidates
6. **Generation** → LLM generates with chunk awareness
7. **Attribution** → Parse output, map claims to chunk IDs
8. **Evaluation** → RAGAS faithfulness scoring

---

Ready for build steps! 🚀