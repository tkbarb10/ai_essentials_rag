from tavily import TavilyClient
from config.load_env import load_env
from langchain_core.tools import tool

env_config = load_env()
tavily_client = TavilyClient()

@tool
def web_search(web_query: str) -> str:
    """Search the web for current information using Tavily.

    This wrapper calls `TavilyClient.search(..., include_answer=True, max_results=3)`
    and returns the top answer string. It is intended for short, up-to-date lookups
    not covered by the local knowledge base.

    Args:
        web_query: The search query string.

    Returns:
        The best answer as a string, or a fallback message if no answer is available.

    Raises:
        Any exceptions raised by `TavilyClient` are propagated.
    """
    response = tavily_client.search(web_query, include_answer=True, max_results=3)
    return response.get("answer", "Tell user that no additional information could be found for their query")