from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Optional

def chunk_markdown_text(
        paper_content: str | List[str], 
        chunk_size: int=750, 
        chunk_overlap: int=150, 
        md_kwargs: Optional[dict]=None, 
        rc_kwargs: Optional[dict]=None
        ):
    
    """Split markdown content into smaller chunks suitable for embedding/search.

    The function first splits on Markdown headers using `MarkdownHeaderTextSplitter`
    then further splits text into chunks with `RecursiveCharacterTextSplitter`
    using `chunk_size` and `chunk_overlap`. Accepts a single string or list of strings
    and returns a flattened list of documents (LangChain document objects).

    Args:
        paper_content: A markdown string or list of markdown strings.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        A flat list of chunked documents (as produced by the text splitter).
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
