from dotenv import load_dotenv
import os
from config.paths import SETTINGS
from utils.load_yaml_config import load_yaml_config

load_dotenv()

# Load centralized settings from settings.yaml
_settings = load_yaml_config(SETTINGS)
MODEL_CONFIG = _settings['MODEL_CONFIG']
EMBEDDING_MODEL = _settings['EMBEDDING_MODEL']
TEXT_SPLIT = _settings['TEXT_SPLIT']
VECTOR_STORE = _settings['VECTOR_STORE']
RAG = _settings['RAG']
APP = _settings['APP']


def load_env():
    """Load API keys from environment variables.

    Returns:
        Dictionary mapping key names to their values (None if not set).
    """

    env_config = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }

    return env_config
