from typing import List, Optional, Dict, Any
from config.load_env import load_env, MODEL_CONFIG
from config.paths import PROMPTS_DIR
from langchain_core.prompts import ChatPromptTemplate
from utils.load_yaml_config import load_all_prompts
from utils.prompt_builder import build_prompt
from langchain_core.output_parsers import StrOutputParser
from vector_store.initialize import create_vector_store, initialize_embedding_model
from langchain.chat_models import init_chat_model
from utils.load_files import load_files_as_list
from utils.chunk_content import chunk_markdown_text
from utils.logging_helper import setup_logging

# Load environment variables
env_config = load_env()

class RAGAssistant:
    """
    RAG (Retrieval-Augmented Generation) assistant that uses a Chroma vector store
    and a configurable chat model to answer user queries with retrieved context.

    It initializes LLM and embedding models, manages a vector DB collection, builds
    prompt templates from YAML config, and exposes a simple `invoke` method to
    query the assistant and return a string answer.

    NEW: Now supports dynamic topic substitution and modular prompt components.
    """

    def __init__(
        self,
        persist_path: str,
        collection_name: str,
        topic: str,
        prompt_template: str = 'educational_assistant',
        components: Optional[Dict[str, Any]] = None,
        db_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize the RAG assistant with modular prompt configuration.

        Args:
            persist_path: Directory for Chroma persistence.
            collection_name: Name of the Chroma collection to create or load.
            topic: The topic this RAG assistant specializes in (e.g., "Blueprints for Text Analytics in Python textbook").
                This will replace {topic} placeholders in prompts.
            prompt_template: Name of the prompt template to use from prompts YAML.
                Defaults to 'educational_assistant'.
            components: Optional dict of reusable prompt components:
                - 'tones': Dict of named tone configurations (default: 'conversational')
                - 'reasoning_strategies': Dict of reasoning strategies (default: 'Self-Ask')
                - 'tools': List of tool descriptions (default: web search enabled)
                If None, defaults are applied automatically by build_prompt().
            db_kwargs: Optional dict of kwargs passed to `create_vector_store`.

        Raises:
            ValueError: If the underlying chat model fails to initialize.

        Example:
            >>> # Basic usage with defaults
            >>> assistant = RAGAssistant(
            ...     persist_path="./store/healthcare",
            ...     collection_name='healthcare_rag',
            ...     topic="Aya Healthcare"
            ... )
            >>>
            >>> # Advanced usage with custom components
            >>> components = {'tools': ['Custom tool description']}
            >>> assistant = RAGAssistant(
            ...     persist_path="./store/tech",
            ...     collection_name='tech_rag',
            ...     topic="Cloud Computing",
            ...     prompt_template='qa_assistant',
            ...     components=components
            ... )
        """
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
        self.embed_model = initialize_embedding_model(model_name=env_config['EMBEDDING_MODEL'], show_progress=False)

        # Initialize vector database
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.db_kwargs = db_kwargs or {}
        self.vector_db = create_vector_store(
            persist_path=self.persist_path,
            collection_name=self.collection_name,
            embedding_model=self.embed_model,
            db_kwargs=self.db_kwargs
        )

        # Create RAG prompt template
        self.prompt_template = self._build_prompt_template(
            prompt_name=prompt_template,
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
        components: Optional[Dict[str, Any]] = None
    ):
        """Build a LangChain ChatPromptTemplate from a prompt name in YAML.

        Loads all prompt configurations from the prompts directory and assembles
        the specified template using `build_prompt`. Supports dynamic {topic}
        substitution and modular components.

        Args:
            prompt_name: Name of the prompt template (e.g., 'qa_assistant').
            topic: Topic name to replace {topic} placeholders.
            components: Optional reusable components (tones, reasoning_strategies, tools).

        Returns:
            ChatPromptTemplate configured with the assembled prompt.

        Raises:
            KeyError: If prompt_name is not found in the YAML files.
            FileNotFoundError: If the prompts directory doesn't exist.
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
                components=components
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
        """Format retrieved documents into a readable context string.

        Each retrieved document is rendered with its index, 'Main Topic', 'Subtopic',
        and the page content. Returns a single string suitable for injecting into
        the prompt context.
        """
        formatted = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            content = doc.page_content
            
            formatted.append(
                f"Document {i+1}:\n"
                f"Topic: {metadata.get('Main Topic', 'None')}\n"
                f"SubTopic: {metadata.get('Subtopic', 'None')}\n"
                f"Content: {content}\n"
            )
        
        return "\n\n".join(formatted)

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: Iterable of LangChain `Document` objects or dict-like records
                that the vector store's `add_documents` method accepts.
        """
        self.vector_db.add_documents(documents)

    def invoke(self, query: str, conversation: Optional[List]=[], n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            query: User's input string.
            conversation: Optional conversation history (list of messages) to include.
            n_results: Number of top-matching chunks to retrieve from the vector store.

        Returns:
            The LLM answer as a string.
        """
        docs = self.vector_db.search(query=query, search_type='similarity', k=n_results)
        context = self._format_docs_with_metadata(docs)

        message_payload = {
            "context": context,
            "conversation": conversation,
            "query": query
        }
        
        llm_answer = self.chain.invoke(message_payload)
        return llm_answer

def main(documents_path: Optional[str], persist_path: str, collection_name: str, topic: str):
    """Demonstration CLI for the RAG assistant.

    This function initializes `RAGAssistant` and (optionally) loads and chunks
    documents from `documents_path` into the vector store. It then enters a simple
    REPL allowing the user to ask questions against the RAG system.
    """
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant(
            persist_path=persist_path,
            collection_name=collection_name,
            topic=topic
        )

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
    
    parser = argparse.ArgumentParser(description="RAG Assistant to query your vector store")
    parser.add_argument('--documents-path', type=str, default=None, help='Where your docs are stored if you wish to upload any')
    parser.add_argument('--persist-path', type=str, default="./store/aya_healthcare", help='Path to save vector store embeddings')
    parser.add_argument('--collection-name', type=str, default="aya_healthcare_rag", help='Name of the Chroma collection to create or load')
    parser.add_argument('--topic', type=str, default="Blueprint Analytics in Python textbook", help='Topic this RAG assistant specializes in')

    args = parser.parse_args()

    documents_path = args.documents_path
    persist_path = args.persist_path
    collection_name = args.collection_name
    topic = args.topic

    main(documents_path=documents_path, persist_path=persist_path, collection_name=collection_name, topic=topic)
