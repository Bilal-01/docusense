import tiktoken
from typing import List

from .models import TextBlock, Chunk

# Module-level singleton — loaded once, shared across all calls
_ENC = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 300   # tokens
OVERLAP = 50       # tokens


def chunk_text_blocks(
    blocks: List[TextBlock],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[Chunk]:
    """
    Split text blocks into overlapping token-window chunks.

    Uses tiktoken for exact token counts and precise character offset tracking.
    Character offsets are computed by decoding the prefix token sequence, which
    gives exact positions without any approximation or fuzzy matching.
    """
    chunks: List[Chunk] = []
    global_index = 0

    for block in blocks:
        if not block.text.strip():
            continue

        doc_name = block.document_name.rsplit(".", 1)[0]
        tokens = _ENC.encode(block.text)
        i = 0

        while i < len(tokens):
            window = tokens[i : i + chunk_size]
            chunk_text = _ENC.decode(window).strip()

            if not chunk_text:
                i += chunk_size - overlap
                continue

            # Precise char offsets: decode the prefix to find exact positions.
            # This is deterministic and never approximates.
            char_start = len(_ENC.decode(tokens[:i]))
            char_end = len(_ENC.decode(tokens[: i + len(window)]))

            chunks.append(Chunk(
                chunk_id=f"{doc_name}_p{block.page_number}_c{global_index}",
                text=chunk_text,
                source_doc=block.document_name,
                page_number=block.page_number,
                section=block.section_heading,
                char_start=char_start,
                char_end=char_end,
                token_count=len(window),
            ))

            global_index += 1
            i += chunk_size - overlap

    return chunks