"""Path constants and directory initialization for the ai_essentials project.

This module defines all file and directory paths used throughout the application,
including prompt templates, outputs, and data directories. Required directories
are automatically created on module import if they don't exist.

Module Attributes:
    ROOT_DIR: Absolute path to the project root directory.
    PROMPTS_DIR: Directory containing YAML prompt templates.
    RAG_PROMPTS_FPATH: Path to RAG-specific prompts YAML file.
    INGESTION_PROMPTS_FPATH: Path to ingestion prompts YAML file.
    COMPONENTS_FPATH: Path to reusable prompt components YAML file.
    OUTPUTS_DIR: Directory for generated outputs and logs.
    DATA_DIR: Directory for input data files.
    RESPONSE_METADATA: Directory for LLM response metadata CSVs.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ENV_FPATH = os.path.join(ROOT_DIR, ".env")

# CODE_DIR = os.path.join(ROOT_DIR, "code")

# APP_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "config.yaml")

# New modular prompts directory structure
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")
RAG_PROMPTS_FPATH = os.path.join(PROMPTS_DIR, "rag_prompts.yaml")
INGESTION_PROMPTS_FPATH = os.path.join(PROMPTS_DIR, "ingestion_prompts.yaml")
COMPONENTS_FPATH = os.path.join(PROMPTS_DIR, "components.yaml")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESPONSE_METADATA = os.path.join(OUTPUTS_DIR, "metadata")

# PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")

# VECTOR_DB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")

# CHAT_HISTORY_DB_FPATH = os.path.join(OUTPUTS_DIR, "chat_history.db")

direcs_to_create = [PROMPTS_DIR, OUTPUTS_DIR, DATA_DIR, RESPONSE_METADATA]

for dir in direcs_to_create:
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)