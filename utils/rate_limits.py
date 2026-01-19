import httpx
from config.load_env import MODEL_CONFIG
from langchain.chat_models import init_chat_model # type: ignore

class UniversalRateLimitTracker:
    def __init__(self):
        self.limits = {}

    def __call__(self, response: httpx.Response):
        # Dictionary comprehension to grab anything containing 'ratelimit'
        self.limits = {
            k.lower(): v 
            for k, v in response.headers.items() 
            if "remaining-tokens" in k.lower()
        }

tracker = UniversalRateLimitTracker()
client = httpx.Client(event_hooks={'response': [tracker]})

def ping():
    """Ping the LLM to retrieve the token rate limit header.

    Sends a lightweight completion request to the Groq client and returns the
    value of the 'x-ratelimit-limit-tokens' header if present.

    Args:
        model: Model identifier to query (e.g., 'gpt-xyz').

    Returns:
        The token rate limit value as a string, or `None` if the header is not present.

    Raises:
        Exceptions from the client may be raised if the request fails.
    """
    model = init_chat_model(**MODEL_CONFIG, http_client=client)
    response = model.invoke('p', max_tokens=1)

    keys = list(tracker.limits.keys())[0]
    

    return tracker.limits.get(keys, 8000)
