"""Vector store initialization utilities for Chroma with HuggingFace embeddings.

This module provides functions to initialize embedding models and create/load
Chroma vector stores. Supports automatic device selection (CUDA, MPS, CPU) for
embedding model inference.

Can be run as a CLI tool to interactively set up a new vector store.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
from typing import Optional
from utils.kwarg_parser import parse_value
from utils.logging_helper import setup_logging

logger = setup_logging(name="initialize")


def initialize_embedding_model(model_name: str="sentence-transformers/all-MiniLM-L6-v2", model_kwargs: dict={}, encode_kwargs: Optional[dict]={}):
    """Initialize a HuggingFace embedding model with automatic device selection.

    Detects available hardware (CUDA GPU, Apple MPS, or CPU) and configures the
    embedding model to use the best available device. Logs the model configuration
    for debugging.

    Args:
        model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2).
        model_kwargs: Arguments passed to the model (device is auto-added).
        encode_kwargs: Arguments passed to the encode method
            (e.g., normalize_embeddings).

    Returns:
        Configured HuggingFaceEmbeddings instance ready for vectorization.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Device being used in the embedding model is {device}\n")

    model_kwargs.update({"device": device})

    model = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    logger.info(f"Embedding model initialized: {model_name}\nConfiguration: \n{model.model_dump()}\n\n=== Model Params ===\n{model._client}")
    return model


def create_vector_store(persist_path: str, collection_name: str, embedding_model=None, collection_metadata={"hnsw:space": "cosine"}, **kwargs):
    """Create or load a Chroma vector store.

    Initializes a LangChain Chroma wrapper that persists data to disk. If a
    collection already exists at the path, it will be loaded; otherwise a new
    collection is created.

    Args:
        persist_path: Directory path for Chroma database persistence.
        collection_name: Name of the collection to create or load.
        embedding_model: HuggingFaceEmbeddings instance for vectorization.
        collection_metadata: Chroma collection config (default: cosine similarity).
        **kwargs: Additional arguments passed to Chroma constructor.

    Returns:
        Configured Chroma vector store instance.
    """
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_path,
        collection_metadata=collection_metadata,
        **kwargs
    )

    return store


if __name__ == "__main__":
    import argparse
    from utils.terminal_dict_parsing import pydict_type
    import sys
    
    parser = argparse.ArgumentParser(description="Initialize embedding model and create/load a Chroma vector store")
    parser.add_argument('--model-name', type=str, default="sentence-transformers/all-MiniLM-L6-v2", help='HuggingFace embedding model name')
    parser.add_argument('--model-kwargs', type=pydict_type, default="{}", help='Model arguments to pass to embedding model.  Pass as dictionary string')
    parser.add_argument('--encode-kwargs', type=pydict_type, default="{}", help='Model arguments to pass to encoding method.  Pass as dictionary string')
    parser.add_argument('--persist-path', type=str, default=None, help='Path to save vector store embeddings')
    parser.add_argument('--collection-name', type=str, default=None, help='Name of the Chroma collection to create or load')
    parser.add_argument('--collection-metadata', type=pydict_type, default='{"hnsw:space": "cosine"}', help="Config arguments to pass through to the chroma client.  Pass as dictionary string")

    args = parser.parse_args()

    model_name = args.model_name
    model_kwargs = args.model_kwargs
    encode_kwargs = args.encode_kwargs
    persist_path = args.persist_path
    collection_name = args.collection_name
    collection_metadata = args.collection_metadata

    print(encode_kwargs)

    # Prompt for model if still using default
    if model_name == "sentence-transformers/all-MiniLM-L6-v2":
        user_choice = input("\nEnter the embedding model name you wish to use. If you're ok with the default (all-MiniLM-L6-v2), just hit Enter: ")
        if user_choice.strip():
            model_name = user_choice.strip()

    # Initialize embedding model
    try:
        print("\nInitializing embedding model...\n")
        embedding_model = initialize_embedding_model(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            model_kwargs=model_kwargs
        )
        print("Embedding model initialized successfully!\n")
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

    # Prompt for kwargs to pass to Chroma (always interactive)
    kwargs = {}
    add_kwargs = input("\nDo you wish to change any other default parameter in the Langchain Chroma() class? Type 'yes' or 'no': ")

    if add_kwargs.lower() in ['yes', 'y', 'yeah', 'ya']:
        user_kwargs = input("\nOk then enter your parameters in this format: arg_name1=value1,arg_name2=value2: ")

        for kwarg in user_kwargs.split(","):
            dict_pair = parse_value(kwarg)
            if dict_pair and len(dict_pair) == 2:
                kwargs[dict_pair[0]] = dict_pair[1]
        
        print(f"\nHere are the custom parameters that will be used in the vector store: {kwargs}\n")

    # Create vector store
    try:
        print("\nCreating vector store...\n")
        store = create_vector_store(
            persist_path=persist_path, 
            embedding_model=embedding_model, 
            collection_name=collection_name, 
            collection_metadata=collection_metadata,
            **kwargs
        )
        print(f"Vector store created successfully at {persist_path}!\n")
        print(f"Collection name: {collection_name}\n")
        logger.info(f"Vector store created: path={persist_path}, collection={collection_name}, db_kwargs={kwargs}")
    except Exception as e:
        print("Oh dear, we are experiencing technical difficulties with initializing the vector store")
        print(f"Your mission, should you choose to accept it, is to fix this error: {e}")
        logger.exception(f"Failed to create vector store: path={persist_path}, collection={collection_name}")
        sys.exit(1)
