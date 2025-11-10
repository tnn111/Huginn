"""Shared utilities for Huginn RAG system."""

import base64
import logging
from pathlib import Path
from typing import Optional
from docling_core.types.doc.document import DoclingDocument, DocItemLabel


def decode_doi_from_filename(filename: str) -> str:
    """Decode DOI from base64-encoded filename.

    Args:
        filename: Base64-encoded DOI (e.g., 'MTAuMjE3NjkvQmlvUHJvdG9jLjE4MTg')

    Returns:
        Decoded DOI (e.g., '10.21769/BioProtoc.1818')
    """
    # Remove file extension if present
    base_name = Path(filename).stem

    try:
        # Add padding if needed
        padding = len(base_name) % 4
        if padding:
            base_name += '=' * (4 - padding)

        decoded_bytes = base64.b64decode(base_name)
        doi = decoded_bytes.decode('utf-8')
        return doi
    except Exception as e:
        logging.warning(f'Failed to decode DOI from filename {filename}: {e}')
        return base_name


def extract_document_title(doc: DoclingDocument) -> Optional[str]:
    """Extract document title from DoclingDocument.

    Looks for the first section_header in the document.

    Args:
        doc: DoclingDocument instance

    Returns:
        Document title if found, None otherwise
    """
    for text_item in doc.texts:
        if text_item.label == DocItemLabel.SECTION_HEADER:
            return text_item.text

    # Fallback: try first text item
    if doc.texts and len(doc.texts) > 0:
        return doc.texts[0].text[:100]  # Truncate if needed

    return None


def extract_authors(doc: DoclingDocument) -> Optional[str]:
    """Attempt to extract author names from document.

    Looks for text patterns near the beginning that might contain authors.

    Args:
        doc: DoclingDocument instance

    Returns:
        Authors string if found, None otherwise
    """
    # Look in first few text items after title
    for i, text_item in enumerate(doc.texts[:5]):
        text = text_item.text.strip()

        # Skip if it's a section header or very long
        if text_item.label == DocItemLabel.SECTION_HEADER:
            continue
        if len(text) > 200:
            continue

        # Look for patterns that suggest authors
        # Common patterns: names with 'and', asterisks for corresponding author
        if ' and ' in text.lower() or '*' in text:
            # Check if it looks like author names (contains uppercase letters)
            if any(c.isupper() for c in text):
                return text

    return None


def get_page_number(chunk) -> Optional[int]:
    """Extract page number from chunk metadata.

    Args:
        chunk: DocChunk object

    Returns:
        Page number if available, None otherwise
    """
    try:
        if chunk.meta.doc_items and len(chunk.meta.doc_items) > 0:
            first_item = chunk.meta.doc_items[0]
            if first_item.prov and len(first_item.prov) > 0:
                return first_item.prov[0].page_no
    except (AttributeError, IndexError):
        pass

    return None


def format_citation(doi: str, headings: Optional[list[str]] = None, page_no: Optional[int] = None) -> str:
    """Format a citation string for Claude's responses.

    Args:
        doi: Document DOI
        headings: Section headings (hierarchical)
        page_no: Page number

    Returns:
        Formatted citation string
    """
    parts = [f'DOI: {doi}']

    if headings and len(headings) > 0:
        # Use the most specific (last) heading
        section = headings[-1]
        parts.append(f'{section} section')

    if page_no:
        parts.append(f'p. {page_no}')

    return ', '.join(parts)
