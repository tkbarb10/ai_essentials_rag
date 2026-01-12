import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ENV_FPATH = os.path.join(ROOT_DIR, ".env")

# CODE_DIR = os.path.join(ROOT_DIR, "code")

# APP_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "config.yaml")

# Legacy path - kept for backward compatibility
PROMPT_CONFIG_FPATH = os.path.join(ROOT_DIR, "config", "prompts.yaml")

# New modular prompts directory structure
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")
RAG_PROMPTS_FPATH = os.path.join(PROMPTS_DIR, "rag_prompts.yaml")
INGESTION_PROMPTS_FPATH = os.path.join(PROMPTS_DIR, "ingestion_prompts.yaml")
COMPONENTS_FPATH = os.path.join(PROMPTS_DIR, "components.yaml")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")


# DATA_DIR = os.path.join(ROOT_DIR, "data")
# PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")

# VECTOR_DB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")

# CHAT_HISTORY_DB_FPATH = os.path.join(OUTPUTS_DIR, "chat_history.db")