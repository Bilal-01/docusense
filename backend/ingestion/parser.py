import os
import statistics
from typing import List, Optional

import pdfplumber
from docx import Document as DocxDocument

from .models import TextBlock

def parse_pdf(file_path: str) -> List[TextBlock]:
    blocks = []
    document_name = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            chars = page.chars
            if not chars:
                continue

            # Find the modal (most common) font size — this is body text
            all_sizes = [c["size"] for c in chars]
            try:
                body_size = statistics.mode(all_sizes)
            except statistics.StatisticsError:
                body_size = sorted(all_sizes)[len(all_sizes) // 2]

            # Group characters into lines by vertical position
            lines: dict = {}
            for c in chars:
                key = round(c["top"], 1)
                lines.setdefault(key, []).append(c)

            heading_stack: List[str] = []
            page_body_lines: List[str] = []
            last_section: str | None = None

            for key in sorted(lines):
                runs = sorted(lines[key], key=lambda c: c["x0"])
                line_text = "".join(r["text"] for r in runs).strip()
                if not line_text:
                    continue

                avg_size = sum(r["size"] for r in runs) / len(runs)
                is_bold = any(
                    "bold" in r.get("font", "").lower() for r in runs
                )

                is_heading = (avg_size > body_size + 0.5) or (is_bold and avg_size >= body_size)

                if is_heading:
                    heading_stack.append(line_text)
                else:
                    last_section = " > ".join(heading_stack) if heading_stack else None
                    page_body_lines.append(line_text)

            # Emit one TextBlock per page so char offsets are page-scoped.
            # This ensures char_start/char_end in chunks map correctly to
            # the page-level character positions used by the PDF highlight overlay.
            if page_body_lines:
                blocks.append(TextBlock(
                    text=" ".join(page_body_lines),
                    page_number=page_num,
                    document_name=document_name,
                    source_file=file_path,
                    section_heading=last_section,
                ))

    return blocks
def parse_docx(file_path: str) -> List[TextBlock]:
    blocks = []
    document_name = os.path.basename(file_path)
    doc = DocxDocument(file_path)
    heading_stack: List[str] = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style = paragraph.style.name if paragraph.style else "Normal"

        if style.startswith("Heading"):
            try:
                level = int(style.split()[-1]) - 1
            except (ValueError, IndexError):
                level = 0
            while len(heading_stack) > level:
                heading_stack.pop()
            heading_stack.append(text)
        else:
            blocks.append(TextBlock(
                text=text,
                # FIX: DOCX has no reliable page API without rendering.
                # page_number is always 1. Document this and do not fake it.
                page_number=1,
                document_name=document_name,
                source_file=file_path,
                section_heading=" > ".join(heading_stack) if heading_stack else None,
            ))

    return blocks


def parse_txt(file_path: str) -> List[TextBlock]:
    document_name = os.path.basename(file_path)
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    paragraphs = content.split("\n\n") if "\n\n" in content else content.splitlines()
    return [
        TextBlock(
            text=p.strip(),
            page_number=1,
            document_name=document_name,
            source_file=file_path,
        )
        for p in paragraphs if p.strip()
    ]


def parse_document(file_path: str, file_type: Optional[str] = None) -> List[TextBlock]:
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    if file_type is None:
        file_type = os.path.splitext(file_path)[1].lower().lstrip(".")

    parsers = {"pdf": parse_pdf, "docx": parse_docx, "txt": parse_txt}

    if file_type not in parsers:
        raise ValueError(f"Unsupported file type: {file_type}. Supported: pdf, docx, txt")

    try:
        return parsers[file_type](file_path)
    except ValueError:
        raise
    except Exception as e:
        # FIX: chain original exception so the full traceback is preserved
        raise IOError(f"Failed to parse {file_type} file: {e}") from e