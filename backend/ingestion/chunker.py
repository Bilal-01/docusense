from typing import List
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document

from .models import TextBlock, Chunk


def chunk_text_blocks(
    text_blocks: List[TextBlock],
    target_chunk_size: int = 300,
    breakpoint_percentile_threshold: int = 95
) -> List[Chunk]:
    """
    Semantically chunk text blocks using LlamaIndex SemanticSplitterNodeParser.

    Uses sentence embeddings to detect semantic boundaries and only splits when
    meaning changes significantly. Generates deterministic chunk IDs and preserves
    position metadata for document highlighting.

    Args:
        text_blocks: Parser output (List[TextBlock]) from parse_document()
        target_chunk_size: Target chunk size in tokens (200-400 recommended)
        breakpoint_percentile_threshold: Similarity percentile for split detection (0-100)
                                        95 = only split at 95th percentile differences (conservative)

    Returns:
        List of Chunk objects with deterministic IDs and position metadata
    """

    # Initialize embedding model (HuggingFace sentence-transformers)
    embedding_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"  # Lightweight, fast model
    )

    # Initialize semantic splitter
    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        embed_model=embedding_model
    )

    chunks: List[Chunk] = []
    global_chunk_index = 0

    for block in text_blocks:
        # Convert TextBlock to LlamaIndex Document format
        doc = Document(
            text=block.text,
            metadata={
                "document_name": block.document_name,
                "page_number": block.page_number,
                "section_heading": block.section_heading,
                "source_file": block.source_file
            }
        )

        # Apply semantic splitting
        semantic_nodes = semantic_splitter.get_nodes_from_documents([doc])

        if not semantic_nodes:
            continue

        # Track cumulative position in original text
        text_len = len(block.text)
        char_position = 0

        for node in semantic_nodes:
            node_text = node.get_content().strip()
            if not node_text:
                continue

            # Find this chunk's position in the remaining text
            # Try exact match first, then broader search
            remaining_text = block.text[char_position:]
            found_idx = remaining_text.find(node_text)

            if found_idx >= 0:
                # Found exact match
                char_start = char_position + found_idx
                char_end = char_start + len(node_text)
                char_position = char_end
            else:
                # Exact match not found (node text might be trimmed/modified)
                # Use approximate position based on node content matching
                char_start = char_position
                char_end = min(char_position + len(node_text), text_len)
                char_position = char_end

            # Clamp to valid range
            char_start = max(0, min(char_start, text_len))
            char_end = max(char_start, min(char_end, text_len))

            # Estimate token count (rough: ~4 chars per token on average)
            token_count = max(1, len(node_text) // 4)

            # Generate deterministic chunk ID
            # Strip extension from document name for cleaner IDs
            doc_name = block.document_name.rsplit('.', 1)[0]
            chunk_id = f"{doc_name}_p{block.page_number}_c{global_chunk_index}"

            # Create Chunk object
            chunk = Chunk(
                chunk_id=chunk_id,
                text=node_text,
                source_doc=block.document_name,
                page_number=block.page_number,
                section=block.section_heading,
                char_start=char_start,
                char_end=char_end,
                token_count=token_count
            )

            chunks.append(chunk)
            global_chunk_index += 1

    return chunks
