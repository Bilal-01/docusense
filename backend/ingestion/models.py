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

