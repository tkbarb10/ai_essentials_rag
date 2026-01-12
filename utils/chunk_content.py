from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List

def chunk_markdown_text(paper_content: str | List[str], chunk_size: int=750, chunk_overlap: int=50, **kwargs):
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

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Main Topic"), ("##", "Subtopic")]
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        **kwargs
    )

    docs = []

    if isinstance(paper_content, list):
        for content in paper_content:

            markdown_docs = markdown_splitter.split_text(content)
            text_docs = text_splitter.split_documents(markdown_docs)
            docs.append(text_docs)
    
    else:
        markdown_docs = markdown_splitter.split_text(paper_content)
        text_docs = text_splitter.split_documents(markdown_docs)
        docs.append(text_docs)
        
    return sum(docs, [])
