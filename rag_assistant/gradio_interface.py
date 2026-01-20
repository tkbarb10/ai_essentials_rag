from typing import List, Optional, Generator
from rag_assistant.rag_assistant import RAGAssistant
from utils.tools import web_search
from langchain_core.messages import ToolMessage
from utils.logging_helper import setup_logging
import time

logger = setup_logging(name="gradio_logs")
chat_logger = setup_logging(name='chat_logs')

class GradioInterface:
    """Gradio-specific streaming interface for RAG Assistant."""
    def __init__(self, assistant: RAGAssistant, tools: Optional[List] = None):

        """Initialize Gradio interface.
        
        Args:
            assistant: RAGAssistant instance to use for queries.
            tools: List of LangChain tools to make available.
        """
        self.assistant = assistant
        self.vector_db = assistant.vector_db
        self.logger = logger
        self.chat_logger = chat_logger
        self.tools = tools or [web_search]
        self.llm_with_tools = self.assistant.llm.bind_tools(self.tools)

    def stream_chat(self, query: str, history: List[dict], n_results: int = 3) -> Generator[str, None, None]:
        """Stream chat responses for Gradio with RAG and tool calling.
        
        Args:
            query: User's query string.
            history: Gradio chat history (list of message dicts).
            n_results: Number of documents to retrieve.
            
        Yields:
            Incremental response chunks as strings.
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
        """Convert Gradio history format to LangChain message format.
        
        Args:
            history: Gradio chat history.
            
        Returns:
            List of formatted message dicts.
        """
        formatted = []
        for msg in history:
            role = msg.get('role')
            content = msg.get('content')

            if isinstance(content, list) and len(content) > 0:
                # Extract text from Gradio's nested format
                text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
            else:
                text = str(content)
            
            formatted.append({"role": role, "content": text})
        
        return formatted