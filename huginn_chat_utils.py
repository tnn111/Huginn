"""Utilities for Huginn chatbot."""

from typing import Optional


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
