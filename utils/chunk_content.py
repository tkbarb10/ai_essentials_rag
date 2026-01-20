"""Text chunking utilities for preparing documents for vector store ingestion.

This module provides functions to split markdown content into smaller chunks
suitable for embedding and semantic search, preserving header metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Optional
from config.load_env import TEXT_SPLIT


def chunk_markdown_text(
        paper_content: str | List[str],
        chunk_size: Optional[int]=None,
        chunk_overlap: Optional[int]=None,
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
        chunk_size: Maximum characters per chunk. Defaults to TEXT_SPLIT config.
        chunk_overlap: Character overlap between chunks. Defaults to TEXT_SPLIT config.
        md_kwargs: Arguments for MarkdownHeaderTextSplitter. Defaults to TEXT_SPLIT config.
        rc_kwargs: Arguments for RecursiveCharacterTextSplitter.

    Returns:
        List of LangChain Document objects with content and header metadata.
    """
    chunk_size = chunk_size if chunk_size is not None else TEXT_SPLIT['chunk_size']
    chunk_overlap = chunk_overlap if chunk_overlap is not None else TEXT_SPLIT['chunk_overlap']

    # Convert headers config from list of lists to list of tuples for MarkdownHeaderTextSplitter
    default_headers = [tuple(h) for h in TEXT_SPLIT['headers_to_split_on']]
    md_kwargs = md_kwargs or {"headers_to_split_on": default_headers}
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
