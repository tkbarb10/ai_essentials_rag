"""Web content cleaning module using LLM-based text processing.

This module provides functions to clean raw web content (HTML artifacts, redundant
text, formatting issues) using an LLM. It processes content in batches while
respecting rate limits and saves response metadata for tracking.

Can be run as a CLI tool to interactively clean web content from files.
"""

from config.load_env import load_env, MODEL_CONFIG
import sys
from langchain.chat_models import init_chat_model # type: ignore
from utils.load_yaml_config import load_all_prompts
from config.paths import PROMPTS_DIR, OUTPUTS_DIR, RESPONSE_METADATA
from utils.prompt_builder import build_prompt
from utils.kwarg_parser import parse_value
from datetime import datetime
from typing import List
from utils.rate_limits import ping
from pathlib import Path
import json
from tqdm import tqdm
from utils.logging_helper import setup_logging
import pandas as pd

# Configure environment
load_env()
logger = setup_logging(name='clean')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load prompts
try:
    prompt_options = load_all_prompts(PROMPTS_DIR)
    scrape_parts = prompt_options.get("scrape_prompt", {})
    if scrape_parts:
        scrape_prompt = build_prompt(scrape_parts)
        print("Successfully loaded scrape_prompt from prompts directory")
    else:
        # Fallback if scrape_prompt not found
        scrape_prompt = "Clean this string of html tags and other web artifacts"
        logger.warning("scrape_prompt not found, using default")
except Exception as e:
    logger.error(f"Error loading prompts: {e}")
    print(f"Warning: Could not load prompts from {PROMPTS_DIR}. Using default.")
    scrape_prompt = "Clean this string of html tags and other web artifacts"


def create_message_payload(web_content: List[str] | str, prompt: str) -> List:
    """Build message payloads for LLM cleaning requests.

    Creates a list of chat completion message payloads, one per content string,
    with the cleaning prompt as the system message.

    Args:
        web_content: Raw web content string or list of strings to clean.
        prompt: System prompt instructing the LLM how to clean content.

    Returns:
        List of message payload lists, each containing system and user messages.
    """
    payloads = []

    if not isinstance(web_content, List):
        web_content = [web_content]

    for string in web_content:
        message_payload = ([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Raw Content:\n {string}"}
        ])

        payloads.append(message_payload)

    return payloads

def cleaned_content(web_content: List[str] | str, prompt: str=scrape_prompt, **kwargs):
    """Clean web content using an LLM with rate limit awareness.

    Processes each content string through the LLM, tracking token usage to stay
    within rate limits. Skips content that would exceed the remaining token budget.
    Saves response metadata to CSV for usage tracking.

    Args:
        web_content: Raw web content string or list of strings to clean.
        prompt: System prompt for cleaning instructions.
        **kwargs: Additional arguments passed to init_chat_model().

    Returns:
        Concatenated cleaned content with section headers separating each piece.
    """
    model = init_chat_model(**MODEL_CONFIG, **kwargs)
    rate_limit = int(ping())
    logger.info(f"Current Rate Limit: {rate_limit}")
    message_payloads = create_message_payload(web_content=web_content, prompt=prompt)
    response_metadata = []
    tokens_used = 0

    cleaned_content = []
    for i, message in enumerate(tqdm(message_payloads, desc="Processing web content")):

        char_count = sum(len(m.get('content', '')) for m in message)
        estimated_tokens = char_count / 4

        if estimated_tokens > (rate_limit - tokens_used):
            print(f"Web content at index {i} is estimated to be {estimated_tokens} tokens, greater than your rate limit of {rate_limit} tokens.  Skipping this message and processing the rest")
            logger.info(f"Skipping web content at index {i}.  Token count estimated to be {estimated_tokens} tokens, greater than rate limit")
            continue
        
        try:
            response = model.invoke(message)
            
            # Track usage metadata
            tokens_used += response.usage_metadata.get('total_tokens', 0)
            response_metadata.append(response.response_metadata['token_usage'])

            # Append content of response to list
            cleaned_content.append(response.content)

        except Exception as e:
            print(f"Error in response for message at index {i}")
            logger.exception(f"Error at index {i}: {e}")

    # Join final list together with a header to separate each individual result
    cleaned_string = "\n\n".join(f"=== WEB CONTENT ===\n\n{item}" for item in cleaned_content)
    pd.json_normalize(response_metadata).to_csv(f"{RESPONSE_METADATA}/llm_clean_{timestamp}.csv", index=False)
    
    return cleaned_string

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Takes the raw content from scrapped websites and removes html tags, redundant and irrelevant content")
    parser.add_argument('--file-path', type=str, default=None, help='File name in the outputs directory you want to clean')

    args = parser.parse_args()

    file_path = args.file_path
    kwargs = {}

    print(f"\n#### CURRENT PROMPT ####\n\n{scrape_prompt}")

    add_to_prompt = input("\nDo you wish to add to this? Type Yes or No: ")
    if add_to_prompt.lower() in ['yes', 'y', 'yay', 'yee', 'yep']:
        prompt_adds = input("\nEnter your additions here: ")
        scrape_prompt = scrape_prompt + "\n\nAdditional Instructions\n\n" + prompt_adds
    
    if not file_path:
        file_path = input("\nEnter file path of raw web content to clean: ")

    content_path = Path(OUTPUTS_DIR) / file_path

    if content_path.is_file():
        print("\nPath exists\n")
    else:
        print(f"Path not found as {content_path}. Exiting program")
        sys.exit(1)

    raw_text = content_path.read_text(encoding='utf-8')

    try:
        raw_content = json.loads(raw_text)
    except json.JSONDecodeError:
        raw_content = raw_text

    # Check for model update
    customize = input(f"\nWould you like to choose a different model to use? Current choice is {MODEL_CONFIG.get('model')}?  Type Yes or No: ")
    if customize.lower() in ['yes', 'y', 'yeah', 'ya']:
        model = input("\nOk, enter a valid model name: ")
        MODEL_CONFIG['model'] = model
    
    add_kwargs = input("\nDo you wish to customize any other arguments in the init_chat_model method for the client?  Type Yes or No: \n")

    if add_kwargs.lower() in ['yes', 'y', 'yeah', 'ya']:

        kwargs_list = input("\nEnter your args in this format, arg_name1=value1,arg_name2=value2: \n")

        for kwarg in kwargs_list.split(","):
            dict_pair = parse_value(kwarg)
            if dict_pair and len(dict_pair) == 2:
                kwargs[dict_pair[0]] = dict_pair[1]

    cleaned_string = cleaned_content(
        web_content=raw_content, 
        prompt=scrape_prompt,
        **kwargs
        )
    
    save_path = Path(OUTPUTS_DIR) / f"cleaned_content_{timestamp}.txt"
    
    print(f"\n##### PREVIEW OF CLEANED CONTENT #####\n\n{cleaned_string[:250]}\n")

    try:
        save_path.write_text(cleaned_string, encoding='utf-8')
        print(f"Web content successfully saved at {save_path}\n")

    except Exception as e:
        print(f"There was an error saving your file.\nError: {e}")
        content_len = len(cleaned_string)
        user_request = input(f"Content is {content_len} characters long, do you wish to print to console to copy/paste instead? Type Yes or No: ")

        if user_request.lower() in ['yes', 'y', 'yeah', 'ya']:
            print(cleaned_string)



