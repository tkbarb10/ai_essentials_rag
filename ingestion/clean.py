from config.load_env import load_env
import sys
from groq import Groq
from utils.load_yaml_config import load_all_prompts
from config.paths import PROMPTS_DIR, OUTPUTS_DIR
from utils.prompt_builder import build_prompt
from utils.kwarg_parser import parse_value
from datetime import datetime
from typing import List
from utils.rate_limits import ping
from pathlib import Path
import json
from tqdm import tqdm
from utils.logging_helper import setup_logging

load_env()
logger = setup_logging(name='clean')

# Load prompts from new modular structure
try:
    prompt_options = load_all_prompts(PROMPTS_DIR)
    scrape_parts = prompt_options.get("scrape_prompt", {})
    if scrape_parts:
        scrape_prompt = build_prompt(scrape_parts)
        logger.info("Successfully loaded scrape_prompt from prompts directory")
    else:
        # Fallback if scrape_prompt not found
        scrape_prompt = "Clean this string of html tags and other web artifacts"
        logger.warning("scrape_prompt not found, using default")
except Exception as e:
    logger.error(f"Error loading prompts: {e}")
    print(f"Warning: Could not load prompts from {PROMPTS_DIR}. Using default.")
    scrape_prompt = "Clean this string of html tags and other web artifacts"

# change this to accommodate different providers
client = Groq()

def create_message_payload(web_content: List[str] | str, prompt: str):
    """Build message payloads for LLM cleaning requests.

    Args:
        web_content: List of raw web content strings to clean.

    Returns:
        List of message payloads suitable for Groq chat completions.
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

def cleaned_content(web_content: List[str] | str, prompt: str, model: str='openai/gpt-oss-20b', reasoning_effort: str='low', temperature: float=0.5, **kwargs):
    """Clean web content strings using a chat model and return joined output.

    Args:
        web_content: List of raw web content strings to clean.
        model: LLM model identifier to use.
        reasoning_effort: Provider-specific reasoning effort setting.
        temperature: Sampling temperature for the model.

    Returns:
        Combined cleaned content as a single string.
    """
    cleaned_content = []
    skipped_messages = {}

    rate_limit = int(ping(model))

    message_payloads = create_message_payload(web_content=web_content, prompt=prompt)

    for i, message in enumerate(tqdm(message_payloads, desc="Processing web content")):

        char_count = sum(len(m.get('content', '')) for m in message)
        estimated_tokens = char_count / 4

        if estimated_tokens > rate_limit:
            print(f"Web content at index {i} is estimated to be {estimated_tokens} tokens, greater than your rate limit of {rate_limit} tokens per minute.  Skipping this message and processing the rest")
            skipped_messages[i] = message
            continue

        response = client.chat.completions.create(
            model=model,
            messages=message,
            reasoning_effort=reasoning_effort, # type: ignore
            temperature=temperature,
            **kwargs
        )

        cleaned_content.append(response.choices[0].message.content)
    
    cleaned_string = "\n\n".join(f"=== WEB CONTENT ===\n\n{item}" for item in cleaned_content)
    
    return cleaned_string, skipped_messages

# either need to feed content from previous script or check for json

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Takes the raw content from scrapped websites and removes html tags, redundant and irrelavant content")
    parser.add_argument('--file-path', type=str, default=None, help='File name in the outputs directory you want to clean')
    parser.add_argument('--model', type=str, default="openai/gpt-oss-20b", help='LLM to use')

    args = parser.parse_args()

    file_path = args.file_path
    model = args.model
    kwargs = {}


    print(f"\n#### CURRENT PROMPT ####\n\n{scrape_prompt}")

    add_to_prompt = input("\nDo you wish to add to this? Type Yes or No: ")
    if add_to_prompt.lower() in ['yes', 'y', 'yay', 'yee', 'yep']:
        prompt_adds = input("\nEnter your additions here: ")
        scrape_prompt = scrape_prompt + "\n\nAdditional Instructions\n\n" + prompt_adds
    
    if not file_path:
        file_path = input("\nEnter file name of raw web content to clean: ")

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

    if model == "openai/gpt-oss-20b":

        customize = input("Would you like to choose a different model to use? Default is openai/gpt-oss-20b?  Type Yes or No: ")
        if customize.lower() in ['yes', 'y', 'yeah', 'ya']:
            model = input("\nOk, enter a valid model name: ")
    
    add_kwargs = input("\nDo you wish to customize any other arguments in the chat_completions method for the client?  Type Yes or No: \n")

    if add_kwargs.lower() in ['yes', 'y', 'yeah', 'ya']:

        kwargs_list = input("\nEnter your args in this format, arg_name1=value1,arg_name2=value2: \n")

        for kwarg in kwargs_list.split(","):
            dict_pair = parse_value(kwarg)
            if dict_pair and len(dict_pair) == 2:
                kwargs[dict_pair[0]] = dict_pair[1]

    cleaned_string, skipped = cleaned_content(
        web_content=raw_content, 
        prompt=scrape_prompt,
        model=model, 
        reasoning_effort='low', 
        temperature=0.5, 
        **kwargs
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(OUTPUTS_DIR) / f"cleaned_content_{timestamp}.txt"
    skipped_path = Path(OUTPUTS_DIR) / f"skipped_{timestamp}.json"
    
    if skipped:
        with open(skipped_path, "w", encoding='utf-8') as f:
            json.dump(skipped, f, ensure_ascii=False)
        print(f"Saved skipped messages at {skipped_path}")

    
    print(f"\n##### PREVIEW OF CLEANED CONTENT #####\n\n{cleaned_string[:250]}")

    try:
        save_path.write_text(cleaned_string, encoding='utf-8')
        print(f"Web content successfully saved at {save_path}")

    except Exception as e:
        print(f"There was an error saving your file.\nError: {e}")
        content_len = len(cleaned_string)
        user_request = input(f"Content is {content_len} characters long, do you wish to print to console to copy/paste instead? Type Yes or No: ")

        if user_request.lower() in ['yes', 'y', 'yeah', 'ya']:
            print(cleaned_string)



