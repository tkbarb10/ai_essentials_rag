from typing import List, Optional, Generator
from rag_assistant.rag_assistant import RAGAssistant
from utils.tools import web_search
from langchain_core.messages import ToolMessage
from utils.logging_helper import setup_logging
import time

logger = setup_logging(name="gradio_logs")
chat_logger = setup_logging(name='chat_logs')

class GradioInterface:
    """Streaming chat interface for RAG Assistant with tool calling support.

    Wraps a RAGAssistant to provide streaming responses suitable for Gradio's
    ChatInterface component. Supports LangChain tool execution (e.g., web search)
    with real-time streaming of both initial responses and tool-augmented follow-ups.

    Attributes:
        assistant: The underlying RAGAssistant instance.
        vector_db: Reference to the assistant's vector store.
        logger: Logger for operational events and usage metadata.
        chat_logger: Logger for conversation history tracking.
        tools: List of LangChain tools available for the LLM to call.
        llm_with_tools: LLM instance with tools bound for function calling.
    """

    def __init__(self, assistant: RAGAssistant, tools: Optional[List] = None):
        """Initialize the Gradio streaming interface.

        Args:
            assistant: Configured RAGAssistant instance for context retrieval
                and prompt management.
            tools: Optional list of LangChain tools. Defaults to [web_search]
                if not provided.
        """
        self.assistant = assistant
        self.vector_db = assistant.vector_db
        self.logger = logger
        self.chat_logger = chat_logger
        self.tools = tools or [web_search]
        self.llm_with_tools = self.assistant.llm.bind_tools(self.tools)

    def stream_chat(self, query: str, history: List[dict], n_results: int = 3) -> Generator[str, None, None]:
        """Stream chat responses with RAG context and optional tool execution.

        Retrieves relevant documents, formats the conversation, and streams the
        LLM response. If the LLM invokes tools (e.g., web search), executes them
        and streams a follow-up response incorporating the tool results.

        Args:
            query: User's current question or message.
            history: Gradio conversation history as list of message dicts with
                'role' and 'content' keys.
            n_results: Number of documents to retrieve from vector store.

        Yields:
            Accumulated response string after each chunk, suitable for Gradio's
            streaming display.
        """

        # 1. Retrieve context from vector store
        start = time.perf_counter()
        docs = self.assistant.search(query=query, k=n_results)
        finish = time.perf_counter()

        context = self.assistant._format_docs_with_metadata(docs)

        self.logger.info(f"Retrieved context for query: {query}\n\n{context}\n\nTime to retrieve: {round((finish - start), 4)}")

        conversation = self._format_history(history)

        self.chat_logger.info(f"=== Ongoing Conversation History ===\n\n{conversation}")

        messages = self.assistant.prompt_template.format_messages(
            query=query,
            context=context,
            conversation=conversation
        )

        output = []
        reasoning = []
        full_response = None

        for chunk in self.llm_with_tools.stream(messages):
            if chunk.content:
                output.append(chunk.content)
                yield "".join(output)
            
            if chunk.additional_kwargs.get('reasoning_content'):
                reasoning.append(chunk.additional_kwargs.get('reasoning_content'))
            
            if chunk.usage_metadata:
                self.logger.info(chunk.usage_metadata)
            
            if full_response is None:
                full_response = chunk
            else:
                full_response += chunk
        
        if hasattr(full_response, 'tool_calls') and full_response.tool_calls: # type: ignore
                messages.append(full_response) # type: ignore

                for tool_call in full_response.tool_calls: # type: ignore
                    # Execute the tool
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    # Notify user about tool usage
                    tool_msg = f"\n\n🔍 *Searching: {tool_args.get('web_query', {query})}*\n\n"

                    self.logger.info(f"Tool query executed for {tool_name} with these args {tool_args}")
                    
                    tool_result = None
                    for tool in self.tools:
                        if tool.name == tool_name:

                            try:
                                tool_result = tool.invoke(tool_args)
                            except Exception as e:
                                self.logger.exception(f"Tool {tool_name} failed")
                                tool_result = f"Tool execution failed: {e}"
                            break
        

                    output.append(tool_msg)
                    yield "".join(output)

                    # Append tool result to messages
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    ))
                
                # 6. Second LLM call with tool results (streaming)
                for chunk in self.llm_with_tools.stream(messages):
                    if chunk.content:
                        output.append(chunk.content)
                        yield "".join(output)

                    if chunk.additional_kwargs.get('reasoning_content'):
                        reasoning.append(chunk.additional_kwargs.get('reasoning_content'))
                
                    if chunk.usage_metadata:
                        self.logger.info(chunk.usage_metadata)
                    
        self.logger.info(f"Reasoning behind response to user query: {query}\n\nReasoning:\n{''.join(reasoning)}")

    
    def _format_history(self, history: List[dict]) -> List:
        """Convert Gradio chat history to LangChain message format.

        Transforms Gradio's message format (which may have nested content) into
        flat dicts with 'role' and 'content' keys for LangChain compatibility.

        Args:
            history: Gradio chat history as list of message dicts.

        Returns:
            List of normalized message dicts with string content.

        Raises:
            ValueError: If messages are missing required 'role' or 'content' keys.
        """
        formatted = []
        for i, msg in enumerate(history):
            if not isinstance(msg, dict):
                raise ValueError(f"History item at index {i} must be a dict, got {type(msg).__name__}")

            if 'role' not in msg:
                raise ValueError(f"History item at index {i} is missing required 'role' key: {msg}")
            if 'content' not in msg:
                raise ValueError(f"History item at index {i} is missing required 'content' key: {msg}")

            role = msg['role']
            content = msg['content']

            if isinstance(content, list) and len(content) > 0:
                # Extract text from Gradio's nested format
                text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
            else:
                text = str(content)

            formatted.append({"role": role, "content": text})

        return formatted