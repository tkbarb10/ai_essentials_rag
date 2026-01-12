from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
import sys
from typing import Optional
from utils.kwarg_parser import parse_value
from utils.logging_helper import setup_logging
from pathlib import Path

logger = setup_logging()

def initialize_embedding_model(model_name: str="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs: dict={"normalize_embeddings": True}, show_progress: bool=True):
    """Initialize a HuggingFace embedding model, selecting an appropriate device.

    The function chooses `cuda` if available, otherwise `mps` (Apple silicon),
    and falls back to `cpu`. It constructs and returns a configured
    `HuggingFaceEmbeddings` instance with the provided `encode_kwargs` and
    optional progress display.

    Args:
        model_name: HuggingFace model identifier.
        encode_kwargs: Keyword arguments passed to the encoder (e.g. normalize_embeddings).
        show_progress: Whether to show embedding progress output.

    Returns:
        A configured `HuggingFaceEmbeddings` instance.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Device being used in the embedding model is {device}\n")

    model = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs,
        show_progress=show_progress
    )
        
    return model


def create_vector_store(persist_path: str, collection_name: str, embedding_model=None, db_kwargs: Optional[dict]=None):
    """Create a Chroma vector store with optional metadata overrides.

    Args:
        persist_path: Directory for persisted Chroma data.
        collection_name: Chroma collection name to create or load.
        embedding_model: Embedding model used for vectorization.
        db_kwargs: Additional keyword arguments for Chroma initialization.

    Returns:
        Initialized Chroma vector store.
    """
    
    db_kwargs = db_kwargs or {}

    # Extend collection metadata if the user wishes to update other parameters in the chroma client configuration or replace the distance measurement
    user_metadata = db_kwargs.pop("collection_metadata", {})
    collection_metadata={"hnsw:space": "cosine"}
    collection_metadata.update(user_metadata)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_path,
        collection_metadata=collection_metadata,
        **db_kwargs
    )

    return store


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize embedding model and create/load a Chroma vector store")
    parser.add_argument('--model-name', type=str, default="sentence-transformers/all-MiniLM-L6-v2", help='HuggingFace embedding model name')
    parser.add_argument('--no-normalize', action='store_true', help='Disable embedding normalization (default: normalize is enabled)')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar (default: progress is shown)')
    parser.add_argument('--persist-path', type=str, default=None, help='Path to save vector store embeddings')
    parser.add_argument('--collection-name', type=str, default=None, help='Name of the Chroma collection to create or load')

    args = parser.parse_args()

    model_name = args.model_name
    normalize = not args.no_normalize  # Invert the flag
    show_progress = not args.no_progress  # Invert the flag
    persist_path = args.persist_path
    collection_name = args.collection_name

    # Prompt for model if still using default
    if model_name == "sentence-transformers/all-MiniLM-L6-v2":
        user_choice = input("\nEnter the embedding model name you wish to use. If you're ok with the default (all-MiniLM-L6-v2), just hit Enter: ")
        if user_choice.strip():
            model_name = user_choice.strip()

    # Build encode_kwargs
    encode_kwargs = {"normalize_embeddings": normalize}

    # Initialize embedding model
    try:
        print("\nInitializing embedding model...\n")
        embedding_model = initialize_embedding_model(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            show_progress=show_progress
        )
        print("Embedding model initialized successfully!\n")
        logger.info(f"Embedding model initialized: {model_name}, normalize={normalize}, show_progress={show_progress}")
    except Exception as e:
        print("Oh dear, we are experiencing technical difficulties with initializing the embedding model")
        print(f"Your mission, should you choose to accept it, is to fix this error: {e}")
        logger.exception(f"Failed to initialize embedding model: {model_name}")
        sys.exit(1)

    print("Next, let's go ahead and set up the vector store. For this, we'll be using the LangChain chromadb wrapper\n")

    # Prompt for persist_path if not provided
    if not persist_path:
        user_path = input("Type in the path you wish to save the vector store embeddings to, or hit Enter to keep the default. The default path is ./chroma/rag_material: ")
        if user_path.strip():
            persist_path = user_path.strip()
        else:
            persist_path = "./chroma/rag_material"

    # Prompt for collection_name if not provided
    if not collection_name:
        collection_name = input("\nType in what you wish to name the vector store collection, or the collection you wish to load if one already exists: ")
        if not collection_name.strip():
            print("Collection name cannot be empty. Exiting program.")
            logger.error("User provided empty collection name")
            sys.exit(1)
        collection_name = collection_name.strip()

    # Prompt for db_kwargs (always interactive)
    db_kwargs = {}
    add_kwargs = input("\nDo you wish to change any other default parameter in the Langchain Chroma() class? Type 'yes' or 'no': ")

    if add_kwargs.lower() in ['yes', 'y', 'yeah', 'ya']:
        user_kwargs = input("\nOk then enter your parameters in this format: arg_name1=value1,arg_name2=value2: ")

        for kwarg in user_kwargs.split(","):
            dict_pair = parse_value(kwarg)
            if dict_pair and len(dict_pair) == 2:
                db_kwargs[dict_pair[0]] = dict_pair[1]
        
        print(f"\nHere are the custom parameters that will be used in the vector store: {db_kwargs}\n")

    # Create vector store
    try:
        print("\nCreating vector store...\n")
        store = create_vector_store(
            persist_path=persist_path, 
            embedding_model=embedding_model, 
            collection_name=collection_name, 
            db_kwargs=db_kwargs
        )
        print(f"Vector store created successfully at {persist_path}!\n")
        print(f"Collection name: {collection_name}\n")
        logger.info(f"Vector store created: path={persist_path}, collection={collection_name}, db_kwargs={db_kwargs}")
    except Exception as e:
        print("Oh dear, we are experiencing technical difficulties with initializing the vector store")
        print(f"Your mission, should you choose to accept it, is to fix this error: {e}")
        logger.exception(f"Failed to create vector store: path={persist_path}, collection={collection_name}")
        sys.exit(1)
