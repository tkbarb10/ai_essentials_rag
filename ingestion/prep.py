"""Content preparation module for vector store ingestion.

This module preprocesses cleaned web content using an LLM to organize and structure
it in markdown format suitable for chunking and storage in a vector database. The
LLM categorizes sections and removes redundant information.

Can be run as a CLI tool to interactively prepare content files.
"""

from utils.load_yaml_config import load_all_prompts
from utils.prompt_builder import build_prompt
from config.load_env import load_env, MODEL_CONFIG
from pathlib import Path
import sys
import os
from datetime import datetime
from config.paths import PROMPTS_DIR, DATA_DIR
from langchain.chat_models import init_chat_model # type: ignore
from utils.logging_helper import setup_logging
from utils.kwarg_parser import parse_value
from typing import List, Optional

# Setup environment
load_env()
logger = setup_logging(name='prep')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = Path(DATA_DIR) / f"prepped_rag_material_{timestamp}.md"

# Load prompt template from new modular structure
try:
    prompt_options = load_all_prompts(PROMPTS_DIR)
    prep_parts = prompt_options.get("prep_prompt", {})
except Exception as e:
    logger.error(f"Error loading prompts: {e}")
    print(f"Warning: Could not load prompts from {PROMPTS_DIR}. Using minimal default.")
    prep_parts = {
        "instructions": "Prep this string for storage into a vector database by removing redundant information and categorizing sections"
    }

def prepare_web_content(
    file_path: Path | str,
    **kwargs
):
    """Preprocess cleaned content for vector store ingestion using an LLM.

    Reads cleaned content from a file, sends it to the LLM with structuring
    instructions, and saves the organized markdown output to the data directory.
    The LLM removes redundant information and formats content with headers.

    Args:
        file_path: Path to the cleaned content file to preprocess.
        **kwargs: Additional arguments passed to build_prompt() for prompt
            customization (e.g., categories, topic).

    Note:
        Output is saved to DATA_DIR with a timestamped filename. Response
        metadata is logged for tracking. Exits with code 1 if file not found.
    """

    # Build prompt with categories placeholder substitution

    model = init_chat_model(**MODEL_CONFIG)

    try:
        prompt = build_prompt(prep_parts, **kwargs)
        logger.info(f"==== PREP PROMPT ===\n\n{prompt}")
    except Exception as e:
        logger.error(f"Error building prompt: {e}")
        print(f"Warning: Error building prompt. Using minimal default.")
        prompt = "Prep this string for storage into a vector database by removing redundant information and categorizing sections"

    content_path = Path(file_path)

    if content_path.is_file():
        print("\nPath exists\n")
    else:
        print(f"Path not found as {content_path}. Exiting program")
        logger.error(
            f"FILE NOT FOUND ERROR: Attempted to access {content_path}. "
            f"Current Working Directory: {os.getcwd()}"
        )
        sys.exit(1)
    
    cleaned_content = content_path.read_text(encoding='utf-8')

    message = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": cleaned_content}
    ]

    print("Sending web content to the model, this may take a second...\n")

    try:

        response = model.invoke(message)

        updated_content = response.content
        logger.info(f"=== Response Metadata ===\n\n{response.response_metadata}")

        with open(save_path, "w", encoding='utf-8') as f:
            f.write(updated_content) # type: ignore
        
        print(f"Response successfully saved at {save_path}\n")
    
    except Exception as e:
        print(f"Sorry, the request could not be completed.  This is the error: {e}.  Please take care of this and try again")
        logger.exception(e)


if __name__ == "__main__":

    # need to add kwargs parser and changes to model config

    import argparse

    parser = argparse.ArgumentParser(description="Takes your string content and organizes it in markdown format to prep it for ingestion into a vector store")
    parser.add_argument('--file-path', type=str, default=None, help='Path to the content you wish to have organized')

    args = parser.parse_args()

    file_path = args.file_path
    kwargs = {}

    # Build initial prompt to show user
    try:
        preview_prompt = build_prompt(prep_parts)
        print(f"\n#### CURRENT PROMPT (Preview) ####\n\n{preview_prompt[:500]}...")
    except Exception as e:
        logger.error(f"Error building preview prompt: {e}")
        print("\nCould not generate prompt preview")

    if not file_path:
        file_path = input("\nPlease enter the file path to the content you wish to prepare for vector storage: ")
    
    # check for model update
    customize = input(f"\nWould you like to choose a different model to use? Current choice is {MODEL_CONFIG.get('model')}?  Type Yes or No: ")
    if customize.lower() in ['yes', 'y', 'yeah', 'ya']:
        model = input("\nOk, enter a valid model name: ")
        MODEL_CONFIG['model'] = model

    add_kwargs = input("\nDo you wish to customize any other arguments in the init_chat_model method for the client?  Type Yes or No: ")

    if add_kwargs.lower() in ['yes', 'y', 'yeah', 'ya']:

        kwargs_list = input("\nEnter your args in this format, arg_name1=value1,arg_name2=value2: ")

        for kwarg in kwargs_list.split(","):
            dict_pair = parse_value(kwarg)
            if dict_pair and len(dict_pair) == 2:
                kwargs[dict_pair[0]] = dict_pair[1]

    prepare_web_content(
        file_path=file_path,
        **kwargs # type: ignore
        )
