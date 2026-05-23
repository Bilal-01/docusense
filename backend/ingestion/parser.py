import os
from pathlib import Path
from typing import List, Optional
import pdfplumber
from docx import Document

from .models import TextBlock


def parse_pdf(file_path: str) -> List[TextBlock]:
    """Parse PDF and extract text blocks with heading hierarchy."""
    blocks = []
    document_name = os.path.basename(file_path)

    with pdfplumber.open(file_path) as pdf:
        heading_stack = []

        for page_num, page in enumerate(pdf.pages, start=1):
            text_runs = []

            for obj in page.chars:
                text_runs.append({
                    "text": obj["text"],
                    "size": obj["size"],
                    "font": obj.get("font", ""),
                    "x0": obj["x0"],
                    "top": obj["top"]
                })

            if not text_runs:
                continue

            # Detect font sizes on this page
            font_sizes = sorted(set(r["size"] for r in text_runs), reverse=True)

            # Group text by position (horizontal line)
            lines = {}
            for run in text_runs:
                line_key = round(run["top"], 1)
                if line_key not in lines:
                    lines[line_key] = []
                lines[line_key].append(run)

            # Process each line
            for line_key in sorted(lines.keys()):
                line_runs = sorted(lines[line_key], key=lambda x: x["x0"])
                line_text = "".join(r["text"] for r in line_runs).strip()

                if not line_text:
                    continue

                # Check if this is a heading: text in top 20% of font sizes or explicitly bold
                avg_size = sum(r["size"] for r in line_runs) / len(line_runs)
                is_bold = any("Bold" in r["font"] or "bold" in r["font"].lower()
                             for r in line_runs)

                # Identify heading tier: top 20% of unique font sizes
                unique_sizes = sorted(set(r["size"] for r in text_runs), reverse=True)
                heading_tier_count = max(1, len(unique_sizes) // 5)
                heading_sizes = set(unique_sizes[:heading_tier_count])
                is_heading_size = avg_size in heading_sizes

                if (is_heading_size or is_bold) and avg_size >= font_sizes[0] * 0.85:
                    # This is likely a heading
                    level = 0
                    for i, size in enumerate(unique_sizes):
                        if avg_size >= size * 0.95:
                            level = i
                            break

                    # Pop stack to appropriate level
                    while len(heading_stack) > level:
                        heading_stack.pop()
                    heading_stack.append(line_text)
                else:
                    # This is body text
                    section_heading = " > ".join(heading_stack) if heading_stack else None

                    blocks.append(TextBlock(
                        text=line_text,
                        page_number=page_num,
                        section_heading=section_heading,
                        document_name=document_name,
                        source_file=file_path
                    ))

    return blocks


def parse_docx(file_path: str) -> List[TextBlock]:
    """Parse DOCX and extract text blocks with heading hierarchy."""
    blocks = []
    document_name = os.path.basename(file_path)
    heading_stack = []
    page_number = 1

    doc = Document(file_path)

    for paragraph in doc.paragraphs:
        if not paragraph.text.strip():
            continue

        style_name = paragraph.style.name if paragraph.style else "Normal"

        # Check if this is a heading
        is_heading = style_name.startswith("Heading")

        if is_heading:
            # Extract heading level from style name (e.g., "Heading 1" -> 1)
            try:
                level = int(style_name.split()[-1]) - 1
            except (ValueError, IndexError):
                level = 0

            # Pop stack back to appropriate level
            while len(heading_stack) > level:
                heading_stack.pop()

            heading_stack.append(paragraph.text.strip())
            # Increment page number on new heading (heuristic for DOCX)
            if level == 0:
                page_number += 1
        else:
            # This is body text
            section_heading = " > ".join(heading_stack) if heading_stack else None

            blocks.append(TextBlock(
                text=paragraph.text.strip(),
                page_number=page_number,
                section_heading=section_heading,
                document_name=document_name,
                source_file=file_path
            ))

    return blocks


def parse_txt(file_path: str) -> List[TextBlock]:
    """Parse TXT file and extract text blocks."""
    blocks = []
    document_name = os.path.basename(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by paragraphs (double newline) or by line if no paragraphs
    paragraphs = content.split('\n\n') if '\n\n' in content else content.split('\n')

    for i, paragraph in enumerate(paragraphs, start=1):
        text = paragraph.strip()
        if text:
            blocks.append(TextBlock(
                text=text,
                page_number=1,
                section_heading=None,
                document_name=document_name,
                source_file=file_path
            ))

    return blocks


def parse_document(file_path: str, file_type: Optional[str] = None) -> List[TextBlock]:
    """
    Parse a document and extract text blocks with metadata.

    Args:
        file_path: Path to the document file
        file_type: File type ('pdf', 'docx', 'txt'). If None, detect from extension.

    Returns:
        List of TextBlock objects

    Raises:
        ValueError: If file type is unsupported or file is not found
        IOError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    # Determine file type
    if file_type is None:
        _, ext = os.path.splitext(file_path)
        file_type = ext.lower().lstrip('.')

    file_type = file_type.lower()

    try:
        if file_type == 'pdf':
            return parse_pdf(file_path)
        elif file_type == 'docx':
            return parse_docx(file_path)
        elif file_type == 'txt':
            return parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Supported: pdf, docx, txt")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise IOError(f"Error parsing {file_type} file: {str(e)}")
