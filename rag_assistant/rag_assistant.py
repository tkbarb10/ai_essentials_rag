from typing import List, Optional, Dict, Any
from config.load_env import load_env, MODEL_CONFIG, EMBEDDING_MODEL, VECTOR_STORE, RAG
from config.paths import PROMPTS_DIR
from config.types import ComponentsDict
from langchain_core.prompts import ChatPromptTemplate
from utils.load_yaml_config import load_all_prompts
from utils.prompt_builder import build_prompt
from langchain_core.output_parsers import StrOutputParser
from vector_store.initialize import create_vector_store, initialize_embedding_model
from langchain.chat_models import init_chat_model
from utils.load_files import load_files_as_list
from utils.chunk_content import chunk_markdown_text
from utils.logging_helper import setup_logging
import time

# Load environment variables
load_env()


class RAGAssistant:
    """RAG (Retrieval-Augmented Generation) assistant for contextual question answering.

    A configurable assistant that combines a Chroma vector store with an LLM chat model
    to answer user queries using retrieved context from stored documents. The assistant
    supports dynamic topic substitution and modular prompt components for flexible
    deployment across different domains.

    Attributes:
        topic: The domain topic this assistant specializes in.
        logger: Logger instance for tracking operations.
        components: Dict of prompt components (tones, reasoning_strategies, tools).
        llm: Initialized LangChain chat model instance.
        embed_model: HuggingFace embedding model for vectorization.
        vector_db: Chroma vector store for document storage and retrieval.
        prompt_template: Assembled ChatPromptTemplate for LLM interactions.
        chain: LangChain chain combining prompt, LLM, and output parser.

    Example:
        Basic usage with default settings:

        >>> from rag_assistant.rag_assistant import RAGAssistant
        >>> assistant = RAGAssistant(
        ...     topic="Machine Learning Fundamentals",
        ...     persist_path="./chroma/ml_docs",
        ...     collection_name="ml_collection"
        ... )
        >>> response = assistant.invoke("What is gradient descent?")
        >>> print(response)

        Advanced usage with custom components:

        >>> components = {
        ...     "tones": "technical",
        ...     "reasoning_strategies": "CoT",
        ...     "tools": True
        ... }
        >>> assistant = RAGAssistant(
        ...     topic="Cloud Architecture",
        ...     persist_path="./chroma/cloud",
        ...     collection_name="cloud_docs",
        ...     prompt_template="qa_assistant",
        ...     components=components
        ... )

        Adding documents and querying with conversation history:

        >>> from utils.load_files import load_files_as_list
        >>> from utils.chunk_content import chunk_markdown_text
        >>> docs = load_files_as_list("./data/documents")
        >>> chunked = chunk_markdown_text(docs)
        >>> assistant.add_documents(chunked)
        >>> history = [
        ...     {"role": "user", "content": "What is AWS?"},
        ...     {"role": "assistant", "content": "AWS is Amazon Web Services..."}
        ... ]
        >>> response = assistant.invoke(
        ...     query="How does S3 fit into that?",
        ...     conversation=history,
        ...     n_results=5
        ... )
    """

    def __init__(
        self,
        topic: str,
        persist_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        prompt_template: Optional[str] = None,
        components: Optional[ComponentsDict] = None,
        store: Optional[Any] = None,
        **kwargs
    ):
        """Initialize the RAG assistant with modular prompt configuration.

        Sets up the LLM, embedding model, vector store, and prompt template. The
        assistant is ready to accept queries immediately after initialization if
        documents already exist in the specified collection.

        Args:
            topic: The domain topic this assistant specializes in. Replaces {topic}
                placeholders in prompts (e.g., "Blueprint Text Analytics in Python").
            persist_path: Directory path for Chroma database persistence. Defaults to VECTOR_STORE config.
            collection_name: Name of the Chroma collection to create or load. Defaults to VECTOR_STORE config.
            prompt_template: Name of the prompt template from prompts YAML config. Defaults to RAG config.
            components: Optional dict of reusable prompt components with these keys:
                - 'tones': One of 'conversational', 'professional', or 'technical'
                - 'reasoning_strategies': One of 'CoT', 'ReAct', or 'Self-Ask'
                - 'tools': bool (True to enable tool descriptions in prompt)
            store: Optional pre-initialized vector store instance. If provided,
                persist_path and collection_name are ignored for store creation.
            **kwargs: Additional arguments passed to create_vector_store().

        Raises:
            ValueError: If the LLM fails to initialize (check MODEL_CONFIG and API keys).
        """
        persist_path = persist_path or VECTOR_STORE['default_persist_path']
        collection_name = collection_name or VECTOR_STORE['default_collection_name']
        prompt_template = prompt_template or RAG['default_prompt_template']
        # Store topic for later use
        self.topic = topic
        self.logger = setup_logging(name='rag_assistant')

        # Apply default components if not provided
        if components is None:
            self.logger.info("No components provided, using defaults: conversational tone, Self-Ask reasoning, tools enabled")
            # Defaults will be applied in build_prompt() function
            components = {}

        self.components = components

        # Initialize LLM
        try:
            self.llm = init_chat_model(**MODEL_CONFIG)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize LLM: {e}\n"
                "Please check MODEL_CONFIG and API keys in .env"
            ) from e

        # Initialize Embedding Model
        self.embed_model = initialize_embedding_model(model_name=EMBEDDING_MODEL)

        # Initialize vector database
        if store:
            self.vector_db = store
        else:
            self.vector_db = create_vector_store(
                persist_path=persist_path,
                collection_name=collection_name,
                embedding_model=self.embed_model,
                **kwargs
            )

        # Create RAG prompt template
        self.prompt_template = self._build_prompt_template(
            prompt_name=prompt_template, # type: ignore
            topic=topic,
            components=components
        )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print(f"RAG Assistant initialized successfully for topic: {topic}")


    def _build_prompt_template(
        self,
        prompt_name: str,
        topic: Optional[str] = None,
        components: Optional[ComponentsDict] = None
    ):
        """Build a LangChain ChatPromptTemplate from YAML configuration.

        Loads prompt configurations from the prompts directory and assembles
        the specified template with dynamic {topic} substitution and modular
        components. Falls back to a minimal default prompt if loading fails.

        Args:
            prompt_name: Key identifying the prompt template in YAML config.
            topic: Domain topic to substitute into {topic} placeholders.
            components: Optional components dict for tone, reasoning, and tools.

        Returns:
            ChatPromptTemplate with system message, conversation placeholder,
            and human message slot configured.
        """
        try:
            # Load all prompts from the modular directory structure
            all_prompts = load_all_prompts(PROMPTS_DIR)
            self.logger.info(f"Successfully loaded prompts from {PROMPTS_DIR}")

            # Get the specific prompt template
            if prompt_name not in all_prompts:
                raise KeyError(
                    f"Prompt template '{prompt_name}' not found. "
                    f"Available templates: {list(all_prompts.keys())}"
                )

            prompt_config = all_prompts[prompt_name]

            # Build the system prompt with topic substitution and components
            prompt = build_prompt(
                config=prompt_config,
                topic=topic,
                components=components # type: ignore
            )

            self.logger.info(f"Built prompt template '{prompt_name}' with topic '{topic}'")

        except FileNotFoundError as e:
            self.logger.error(f"Prompts directory not found: {e}")
            print(f"Warning: Could not load prompts from {PROMPTS_DIR}. Using minimal default prompt.")
            # Fallback to a basic prompt if files aren't found
            prompt = f"You are a helpful assistant that answers questions about {topic}."

        except Exception as e:
            self.logger.error(f"Error building prompt template: {e}")
            print(f"Warning: Error loading prompt '{prompt_name}': {e}. Using minimal default prompt.")
            # Fallback prompt
            prompt = f"You are a helpful assistant that answers questions about {topic}."

        return ChatPromptTemplate.from_messages([
            ("system", f"{prompt}\n\nContext to use for answering:\n{{context}}"),
            ('placeholder', "{conversation}"),
            ("human", "{query}"),
        ])
    
    def _format_docs_with_metadata(self, docs):
        """Format retrieved documents into a context string for LLM consumption.

        Transforms a list of (Document, score) tuples into a numbered, formatted
        string containing relevance scores, metadata fields, and content for each
        retrieved document.

        Args:
            docs: List of (Document, float) tuples from similarity search.

        Returns:
            Formatted string with numbered documents, relevance scores, metadata,
            and content sections separated by double newlines.
        """
        formatted = []
        for i, item in enumerate(docs):
            doc = item[0]
            sim_score = round(item[1], 4)
            full_metadata = [f"{k}: {v}" for k, v in doc.metadata.items()]
            content = doc.page_content

            formatted.append(
                f"Document {i+1}:\n"
                f"Relevance Score on a 0 to 1 scale: {sim_score}\n"
                f"{"\n".join(full_metadata)}\n"
                f"Content: \n\n{content}"
            )
        return "\n\n".join(formatted)

    def add_documents(self, documents: List) -> None:
        """Add documents to the vector store for retrieval.

        Args:
            documents: List of LangChain Document objects to embed and store.
        """
        self.vector_db.add_documents(documents)

    def search(self, query: str, k: int):
        """Search the vector store for documents similar to the query.

        Args:
            query: Search query string to match against stored documents.
            k: Number of top-matching documents to retrieve.

        Returns:
            List of (Document, float) tuples with relevance scores (0-1 scale).
        """
        return self.vector_db.similarity_search_with_relevance_scores(query=query, k=k)

    def invoke(self, query: str, conversation: Optional[List]=[], n_results: Optional[int] = None) -> str:
        """Query the RAG assistant and get a response.

        Retrieves relevant context from the vector store, formats it with the
        query and conversation history, and invokes the LLM chain to generate
        a response.

        Args:
            query: User's question or input string.
            conversation: Prior conversation history as list of message dicts,
                each with 'role' ('user' or 'assistant') and 'content' keys.
            n_results: Number of documents to retrieve from vector store. Defaults to RAG config.

        Returns:
            LLM-generated response as a string.

        Raises:
            ValueError: If conversation contains malformed messages missing
                required 'role' or 'content' keys.
        """
        n_results = n_results or RAG['default_n_results']
        # Validate conversation history
        if conversation:
            for i, msg in enumerate(conversation):
                if not isinstance(msg, dict):
                    raise ValueError(f"Conversation item at index {i} must be a dict, got {type(msg).__name__}")
                if 'role' not in msg:
                    raise ValueError(f"Conversation item at index {i} is missing required 'role' key: {msg}")
                if 'content' not in msg:
                    raise ValueError(f"Conversation item at index {i} is missing required 'content' key: {msg}")

        start = time.perf_counter()
        docs = self.search(query=query, k=n_results) # type: ignore
        finish = time.perf_counter()

        context = self._format_docs_with_metadata(docs)

        self.logger.info(f"Retrieved context for query: {query}\n\n{context}\n\nTime to retrieve: {round((finish - start), 4)}")

        message_payload = {
            "context": context,
            "conversation": conversation,
            "query": query
        }

        llm_answer = self.chain.invoke(message_payload)
        return llm_answer

def main(documents_path: Optional[str], persist_path: str, collection_name: str, topic: str, **kwargs):
    """Run an interactive CLI session with the RAG assistant.

    Initializes a RAGAssistant instance and optionally loads documents from the
    specified path. Enters a REPL loop where users can ask questions and receive
    contextual answers until they type 'quit'.

    Args:
        documents_path: Optional path to documents to load and chunk. If None,
            assumes documents already exist in the collection.
        persist_path: Directory path for Chroma database persistence.
        collection_name: Name of the Chroma collection to use.
        topic: Domain topic for the assistant.
        **kwargs: Additional arguments passed to RAGAssistant initialization.
    """
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant(
            persist_path=persist_path,
            collection_name=collection_name,
            topic=topic,
            **kwargs
        )

        print(assistant.prompt_template.messages[0].prompt.template) # type: ignore

        if documents_path:
            # Load sample documents
            print("\nLoading documents...")
            sample_docs = load_files_as_list(documents_path=documents_path)
            print(f"Loaded {len(sample_docs)} sample documents")

            print("\nChunking documents...")
            chunked_docs = chunk_markdown_text(sample_docs)
            print(f"Chunked {len(chunked_docs)} sample documents")

            assistant.add_documents(chunked_docs)

        done = False

        conversation = []
        while not done:
            question = input("\nEnter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(query=question, conversation=conversation)
                print("\n", result)
            conversation.append({"role": "user", "content": question})
            conversation.append({"role": "assistant", "content": result})

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key")

if __name__ == "__main__":

    import argparse
    from utils.terminal_dict_parsing import pydict_type
    
    parser = argparse.ArgumentParser(description="RAG Assistant to query your vector store")
    parser.add_argument('--documents-path', type=str, default=None, help='Where your docs are stored if you wish to upload any')
    parser.add_argument('--persist-path', type=str, default=None, help='Path to save vector store embeddings. Defaults to VECTOR_STORE config')
    parser.add_argument('--collection-name', type=str, default=None, help='Name of the Chroma collection to create or load. Defaults to VECTOR_STORE config')
    parser.add_argument('--topic', type=str, default="Blueprint Analytics in Python textbook", help='Topic this RAG assistant specializes in')
    parser.add_argument('--kwargs', type=pydict_type, default=None, help='Additional kwargs to pass to the assistant.  If using powershell, wrap full arg in """arg""" and internal keys/values in single quotes')

    args = parser.parse_args()

    documents_path = args.documents_path
    persist_path = args.persist_path
    collection_name = args.collection_name
    topic = args.topic
    kwargs = args.kwargs

    main(
        documents_path=documents_path, 
        persist_path=persist_path, 
        collection_name=collection_name, 
        topic=topic,
        **kwargs
        )
