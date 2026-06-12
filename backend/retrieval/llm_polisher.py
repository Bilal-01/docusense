"""
LLM-based answer polishing with source tracking.
Generates concise, conversational answers from retrieved chunks and tracks which chunks contributed.
"""
import os
import re
from typing import Dict, List, Tuple, Any

try:
    import ollama
except Exception:
    ollama = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama2")


def call_ollama_llm(prompt: str) -> str:
    """
    Call Ollama LLM with given prompt. Returns response text or empty string if unavailable.
    Tries Client first, then CLI, then returns empty.
    """
    if not prompt:
        return ""

    # Try Ollama client
    if ollama is not None:
        try:
            with ollama.Client(host=OLLAMA_HOST) as client:
                try:
                    resp = client.generate(model=OLLAMA_GEN_MODEL, prompt=prompt)
                    if isinstance(resp, str):
                        return resp.strip()
                    return getattr(resp, "text", str(resp)).strip()
                except Exception:
                    try:
                        resp = client.chat(
                            model=OLLAMA_GEN_MODEL,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        if isinstance(resp, str):
                            return resp.strip()
                        return getattr(resp, "text", str(resp)).strip()
                    except Exception:
                        pass
        except Exception:
            pass

    # Fallback: try Ollama CLI
    try:
        import subprocess
        proc = subprocess.run(
            ["ollama", "run", OLLAMA_GEN_MODEL],
            input=prompt,
            capture_output=True,
            timeout=60,
            encoding='utf-8',
            errors='ignore'
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass

    return ""


def polish_answer_with_sources(
    question: str,
    chunks: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Use LLM to generate a concise, conversational answer from retrieved chunks.
    Track which source chunks contributed to the response.

    Args:
        question: The user's question
        chunks: List of dicts with keys: chunk_id, text, page, char_start, char_end, score

    Returns:
        Tuple of (polished_answer, source_tracking)
        where source_tracking is a list of dicts with:
        - chunk_id: which chunk contributed
        - char_start: start position in polished_answer
        - char_end: end position in polished_answer
        - page: original page number
        - original_char_start: start in original document
        - original_char_end: end in original document
    """
    if not chunks:
        return "", []

    # Build labeled chunks for the LLM
    labeled_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "")
        labeled_chunks.append(f"[Source {i}] (ID: {chunk_id})\n{text}")

    chunks_str = "\n\n".join(labeled_chunks)

    # Prompt LLM to generate answer and cite sources
    prompt = (
        f"Question: {question}\n\n"
        f"Relevant sources:\n{chunks_str}\n\n"
        "Instructions:\n"
        "1. Generate a concise, conversational answer (1-3 sentences).\n"
        "2. Answer using ONLY information from the sources above.\n"
        "3. After each sentence or claim, add a citation like [Source 1] or [Source 1, 2] to indicate which sources you used.\n"
        "4. Do NOT include the source content verbatim; paraphrase naturally.\n\n"
        "Answer:"
    )

    # Call LLM
    polished = call_ollama_llm(prompt)

    if not polished:
        # Fallback: simple extraction if LLM fails
        polished = " ".join([c.get("text", "")[:100] for c in chunks[:2]])

    # Parse source citations from the polished answer
    source_tracking = _track_sources(polished, chunks)

    return polished, source_tracking


def _track_sources(response: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse the LLM-generated response to extract which sources (chunk indices) were cited.
    Map citations to character positions in the response and original chunk metadata.

    Returns list of dicts with source attribution info.
    """
    source_tracking = []

    # Find all [Source N] or [Source N, M, ...] citations
    pattern = r"\[Source\s+([\d,\s]+)\]"
    citations = list(re.finditer(pattern, response))

    # If no citations found, assume all chunks contributed equally
    if not citations:
        for i, chunk in enumerate(chunks):
            source_tracking.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "char_start": 0,
                "char_end": len(response),
                "page": chunk.get("page"),
                "original_char_start": chunk.get("char_start"),
                "original_char_end": chunk.get("char_end"),
                "score": chunk.get("score"),
            })
        return source_tracking

    # Map each citation to character positions
    for citation_match in citations:
        citation_text = citation_match.group(1)
        source_indices = [int(x.strip()) - 1 for x in citation_text.split(",")]  # Convert to 0-indexed

        for src_idx in source_indices:
            if 0 <= src_idx < len(chunks):
                chunk = chunks[src_idx]
                source_tracking.append({
                    "chunk_id": chunk.get("chunk_id", f"chunk_{src_idx}"),
                    "char_start": citation_match.start(),
                    "char_end": citation_match.end(),
                    "page": chunk.get("page"),
                    "original_char_start": chunk.get("char_start"),
                    "original_char_end": chunk.get("char_end"),
                    "score": chunk.get("score"),
                })

    # Deduplicate and sort
    unique_tracking = {}
    for item in source_tracking:
        key = item["chunk_id"]
        if key not in unique_tracking:
            unique_tracking[key] = item

    return list(unique_tracking.values())


__all__ = ["polish_answer_with_sources", "call_ollama_llm"]
