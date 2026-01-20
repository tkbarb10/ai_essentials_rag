"""Rate limit tracking utilities for LLM API calls.

This module provides tools to discover and track token rate limits from LLM
provider APIs by inspecting response headers. Useful for managing batch
processing within rate constraints.
"""

import httpx
from config.load_env import MODEL_CONFIG
from langchain.chat_models import init_chat_model # type: ignore


class UniversalRateLimitTracker:
    """HTTP response hook that extracts rate limit headers.

    Captures 'remaining-tokens' headers from API responses to track available
    token budget. Used as an httpx response event hook.

    Attributes:
        limits: Dict of rate limit header keys to values from last response.
    """

    def __init__(self):
        self.limits = {}

    def __call__(self, response: httpx.Response):
        self.limits = {
            k.lower(): v
            for k, v in response.headers.items()
            if "remaining-tokens" in k.lower()
        }


tracker = UniversalRateLimitTracker()
client = httpx.Client(event_hooks={'response': [tracker]})


def ping():
    """Query the LLM API to retrieve current token rate limit.

    Sends a minimal completion request to trigger a response with rate limit
    headers, then extracts the remaining token count from the response.

    Returns:
        Remaining token limit as int, or 8000 as default if header not found.
    """
    model = init_chat_model(**MODEL_CONFIG, http_client=client)
    response = model.invoke('p', max_tokens=1)

    keys = list(tracker.limits.keys())[0]
    

    return tracker.limits.get(keys, 8000)
