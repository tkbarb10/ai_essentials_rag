from utils.load_files import load_files_as_list
from pathlib import Path
from utils.chunk_content import chunk_markdown_text
from vector_store.initialize import create_vector_store, initialize_embedding_model
from utils.logging_helper import setup_logging
import sys
import os

logger = setup_logging()


def upload_content_to_store(persist_path: str, collection_name: str, documents_path: str | Path, **kwargs):
    """Load, chunk, and upload documents into a vector store.

    Args:
        store: Vector store instance with an add_documents method.
        documents_path: File or directory path containing documents to ingest.
        **kwargs: Chunking keyword arguments forwarded to chunk_markdown_text.
    """
    embedding_model = initialize_embedding_model(show_progress=False)
    store = create_vector_store(persist_path, collection_name, embedding_model)
    publications = load_files_as_list(documents_path)
    docs = chunk_markdown_text(publications, **kwargs)
    store.add_documents(docs)

    print(f"\nSuccessfully added files from {documents_path} to vector store\n")
    logger.info(f"Uploaded {len(docs)} chunks from {documents_path} to {collection_name}") 

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Load or create a vector store and insert documents")
    parser.add_argument('--persist-path', type=str, default=None, help="Path where your vector store is.  If one does not exist then it will be created at this path")
    parser.add_argument('--collection-name', default=None, help='Name of the collection you wish to retrieve or create')
    parser.add_argument('--documents-path', default=None, help='Path to where the documents you wish to store are saved')

    args = parser.parse_args()

    persist_path = args.persist_path
    collection_name = args.collection_name
    documents_path = args.documents_path

    # Prompt for persist_path if not provided
    if not persist_path:
        user_path = input("\nType in the path where your vector store is or where you'd like to create it: ")
        if user_path.strip():
            persist_path = user_path.strip()
        else:
            persist_path = "./chroma/rag_material"
            logger.info(f"User didn't provide a user_path so creating it at {persist_path}")

    # Prompt for collection_name if not provided
    if not collection_name:
        collection_name = input("\nType in the name of your collection: ")
        if not collection_name.strip():
            collection_name = "user_collection"
            logger.info(f"User provided empty collection name so using {collection_name}")
        collection_name = collection_name.strip()

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
            documents_path=documents_path
        )
    except Exception as e:
        print(f"Failed to upload content: {e}")
        logger.exception(f"Upload failed {e}")
        sys.exit(1)
