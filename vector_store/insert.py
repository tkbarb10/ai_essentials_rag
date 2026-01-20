"""Document insertion utilities for populating vector stores.

This module provides functions to load documents from disk, chunk them into
smaller pieces suitable for embedding, and upload them to a Chroma vector store.

Can be run as a CLI tool to interactively add documents to a vector store.
"""

from typing import Optional, Any
from utils.load_files import load_files_as_list
from pathlib import Path
from utils.chunk_content import chunk_markdown_text
from vector_store.initialize import create_vector_store, initialize_embedding_model
from utils.logging_helper import setup_logging
from config.load_env import TEXT_SPLIT, VECTOR_STORE
import sys
import os

logger = setup_logging(name='insert_docs')


def upload_content_to_store(
        documents_path: str | Path,
        persist_path: Optional[str]=None,
        collection_name: Optional[str]=None,
        store: Optional[Any]=None,
        **kwargs
        ):
    """Load, chunk, and upload documents to a vector store.

    Reads text/markdown files from the specified path, chunks them using
    markdown-aware splitting, and adds the resulting documents to a Chroma
    vector store. Initializes a new store if none is provided.

    Args:
        documents_path: Path to a file or directory containing documents.
        persist_path: Directory for Chroma persistence. Defaults to VECTOR_STORE config.
        collection_name: Collection name. Defaults to VECTOR_STORE config.
        store: Optional pre-initialized vector store. If None, creates one.
        **kwargs: Arguments passed to chunk_markdown_text() for chunking config.
    """
    persist_path = persist_path or VECTOR_STORE['default_persist_path']
    collection_name = collection_name or VECTOR_STORE['default_collection_name']
    if not store:
        print("Initializing store connection...")
        embedding_model = initialize_embedding_model()
        store = create_vector_store(persist_path, collection_name, embedding_model)

    try:
        publications = load_files_as_list(documents_path)
        docs = chunk_markdown_text(publications, **kwargs)
        store.add_documents(docs)

        print(f"\nSuccessfully added files from {documents_path} to vector store\n")
        logger.info(f"Uploaded {len(docs)} chunks from {documents_path} to {collection_name}") 

    except Exception as e:
        print(f"Error loading documents: {e}")
        logger.exception(f"=== Error Loading Docs ===\n\n{e}")


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Load or create a vector store and insert documents")
    parser.add_argument('--persist-path', type=str, default=None, help="Path where your vector store is.  If one does not exist then it will be created at this path")
    parser.add_argument('--collection-name', type=str, default=None, help='Name of the collection you wish to retrieve or create')
    parser.add_argument('--documents-path', type=str, default=None, help='Path to where the documents you wish to store are saved')

    args = parser.parse_args()

    persist_path = args.persist_path
    collection_name = args.collection_name
    documents_path = args.documents_path

    # Prompt for persist_path if not provided
    if not persist_path:
        default_path = VECTOR_STORE['default_persist_path']
        user_path = input(f"\nType in the path where your vector store is or where you'd like to create it (default: {default_path}): ")
        if user_path.strip():
            persist_path = user_path.strip()
        else:
            persist_path = default_path
            logger.info(f"User didn't provide a user_path so creating it at {persist_path}")

    # Prompt for collection_name if not provided
    if not collection_name:
        default_collection = VECTOR_STORE['default_collection_name']
        user_collection = input(f"\nType in the name of your collection (default: {default_collection}): ")
        if user_collection.strip():
            collection_name = user_collection.strip()
        else:
            collection_name = default_collection
            logger.info(f"User provided empty collection name so using {collection_name}")

    # Prompt for documents_path if not provided
    if not documents_path:
        documents_path = input("\nType in the path where your documents are located: ")
    
    documents_path = Path(documents_path)
    if not documents_path.exists():
        print(f"Path not found at {documents_path}. Exiting program")
        logger.error(
            f"FILE NOT FOUND ERROR: Attempted to access {documents_path}. "
            f"Current Working Directory: {os.getcwd()}"
        )
        sys.exit(1)

    try:
        upload_content_to_store(
            persist_path=persist_path,
            collection_name=collection_name,
            documents_path=documents_path,
        )
    except Exception as e:
        print(f"Failed to upload content: {e}")
        logger.exception(f"Upload failed {e}")
        sys.exit(1)
