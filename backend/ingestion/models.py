from dataclasses import dataclass
from typing import Optional


@dataclass
class TextBlock:
    text: str
    page_number: int
    document_name: str
    source_file: str
    section_heading: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    section: Optional[str]
    char_start: int
    char_end: int
    token_count: int

    def to_metadata(self) -> dict:
        """
        ChromaDB metadata payload. Text is excluded — it is passed separately
        via the `documents` argument in upsert() and is already the source of truth.
        """
        return {
            "chunk_id": self.chunk_id,
            "source_doc": self.source_doc,
            "page_number": self.page_number,
            "section": self.section or "",
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
        }