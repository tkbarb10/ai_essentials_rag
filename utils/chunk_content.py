"""Text chunking utilities for preparing documents for vector store ingestion.

This module provides functions to split markdown content into smaller chunks
suitable for embedding and semantic search, preserving header metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Optional


def chunk_markdown_text(
        paper_content: str | List[str],
        chunk_size: int=750,
        chunk_overlap: int=150,
        md_kwargs: Optional[dict]=None,
        rc_kwargs: Optional[dict]=None
        ):
    """Split markdown content into chunks for vector store embedding.

    Uses a two-stage splitting approach: first splits on markdown headers to
    preserve document structure as metadata, then recursively splits into
    smaller chunks based on character count. Header information (H1, H2, H3)
    is preserved in document metadata.

    Args:
        paper_content: Markdown string or list of markdown strings to chunk.
        chunk_size: Maximum characters per chunk (default: 750).
        chunk_overlap: Character overlap between chunks (default: 150).
        md_kwargs: Arguments for MarkdownHeaderTextSplitter. Defaults to
            splitting on H1, H2, and H3 headers.
        rc_kwargs: Arguments for RecursiveCharacterTextSplitter.

    Returns:
        List of LangChain Document objects with content and header metadata.
    """
    md_kwargs = md_kwargs or {"headers_to_split_on": [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]}
    rc_kwargs = rc_kwargs or {}

    markdown_splitter = MarkdownHeaderTextSplitter(
        **md_kwargs
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        **rc_kwargs
    )

    docs = []

    content_list = [paper_content] if isinstance(paper_content, str) else paper_content

    for content in content_list:

        markdown_docs = markdown_splitter.split_text(content)
        text_docs = text_splitter.split_documents(markdown_docs)
        docs.extend(text_docs)
        
    return docs
