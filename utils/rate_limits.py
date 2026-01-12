from groq import Groq
from config.load_env import load_env

load_env()

client = Groq()

def ping(model: str):
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
    ping = client.chat.completions.with_raw_response.create(
        messages=[{"role": "user", "content": "p"}],
        model=model, 
        max_completion_tokens=1 
    )

    return ping.headers.get('x-ratelimit-limit-tokens')
