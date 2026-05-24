from dataclasses import dataclass
from typing import Optional


@dataclass
class TextBlock:
    """Represents a parsed text block from a document with metadata."""
    text: str
    page_number: int
    document_name: str
    source_file: str
    section_heading: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "page_number": self.page_number,
            "section_heading": self.section_heading,
            "document_name": self.document_name,
            "source_file": self.source_file
        }


@dataclass
class Chunk:
    """Represents a semantically chunked text with metadata for retrieval."""
    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    section: Optional[str]
    char_start: int
    char_end: int
    token_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_doc": self.source_doc,
            "page_number": self.page_number,
            "section": self.section,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count
        }

