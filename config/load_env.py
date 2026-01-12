from dotenv import load_dotenv
import os

load_dotenv()

MODEL_CONFIG = {
    "model": os.getenv("MODEL", "openai/gpt-oss-20b"),
    "model_provider": os.getenv('PROVIDER', "groq"),
    "temperature": float(os.getenv('MODEL_TEMP', 0.5)),
    "reasoning_effort": os.getenv('REASONING_EFFORT', 'low')
}

def load_env():
    """Load API keys from environment variables.
    
    Returns:
        Dictionary mapping key names to their values (None if not set).
    """

    env_config = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    }

    return env_config
