from utils.load_yaml_config import load_all_prompts
from utils.prompt_builder import build_prompt
from config.load_env import load_env
from pathlib import Path
import sys
import os
from datetime import datetime
from config.paths import PROMPTS_DIR, OUTPUTS_DIR
from groq import Groq
from utils.logging_helper import setup_logging
from utils.kwarg_parser import parse_value
from typing import List, Optional

load_env()
logger = setup_logging(name='prep')

# change this to accomodate different providers
client = Groq()

# Default categories for Aya Healthcare (can be overridden)
DEFAULT_CATEGORIES = [
    'Frequently Asked Questions (FAQ)',
    'Basic Questions about Travel Nursing: How it works, general industry info.',
    'Pay: Salary, rates, and compensation structure.',
    'Reviews: Include all nurse reviews with the nurse name, specialty, and the full review.',
    'Benefits and Perks',
    'Job Details: General information on available job types, locations, and the application and hiring process (not specific job listings).',
    'Compliance: How the compliance process works for clinicians.',
    'General Information',
    'Scholarships and Education Programs: Provide detailed information on available programs.',
    'Awards and Recognitions: Include full details on the award, its purpose, nomination process, and criteria for receiving it.',
    'Clinician Stories and Testimonials: Include the clinician name, profession, and their full story or testimonial.',
    'Leadership and Team Profiles: Provide names and titles for all executives and founders mentioned in the text.',
    'Housing and Relocation',
    'Technology and Platforms',
    'Advisory and Workforce Analytics Solutions',
    'Community Impact and CSR',
    'International and Global Operations',
    'Employment Types and Contract Models',
    'Research, Demographics and Market Studies',
    'Media, Podcasts, Blogs and Press Coverage: For each piece, provide the title and a 2-3 sentence detailed summary.',
    'Regulatory and Licensing Details',
    'Career Development and Professional Growth',
    'Pricing and Stipends: Specifically covering Housing, Meal, and Incidental stipends.',
    'Emergency and 24-Hour Support',
    'Links: Provide relevant links for further action such as searching for jobs, specific blogs, or career tips.'
]

# Load prompt template from new modular structure
try:
    prompt_options = load_all_prompts(PROMPTS_DIR)
    prep_parts = prompt_options.get("prep_prompt", {})
    logger.info("Successfully loaded prep_prompt from prompts directory")
except Exception as e:
    logger.error(f"Error loading prompts: {e}")
    print(f"Warning: Could not load prompts from {PROMPTS_DIR}. Using minimal default.")
    prep_parts = {
        "instructions": "Prep this string for storage into a vector database by removing redundant information and categorizing sections"
    }

# make sure output is in markdown format

def prepare_web_content(
    file_path: Path | str,
    categories: Optional[List[str]] = None,
    model: str = 'openai/gpt-oss-120b',
    reasoning_effort: str = 'med',
    **kwargs
):
    """Send cleaned web content to the LLM for preprocessing and save output.

    Args:
        file_path: Path to the cleaned content file to preprocess.
        categories: Optional list of categories for organizing content.
            If None, DEFAULT_CATEGORIES will be used.
        model: LLM model identifier to use.
        reasoning_effort: Provider-specific reasoning effort setting.
        **kwargs: Additional keyword arguments forwarded to the LLM call.

    Returns:
        Tuple of usage statistics and reasoning text when a response is saved.
    """
    # Use default categories if none provided
    if categories is None:
        categories = DEFAULT_CATEGORIES
        logger.info("Using default categories")

    # Build prompt with categories placeholder substitution
    try:
        prompt = build_prompt(prep_parts, categories=categories)
        logger.info(f"Built prep prompt with {len(categories)} categories")
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
        response = client.chat.completions.create(
            model=model,
            messages=message, # type: ignore
            reasoning_effort=reasoning_effort, # type: ignore
            **kwargs
        )

        updated_content = response.choices[0].message.content
        usage_stats = dict(response.usage) # type: ignore
        reasoning = response.choices[0].message.reasoning

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(OUTPUTS_DIR) / f"prepped_rag_material_{timestamp}.md"

        with open(save_path, "w", encoding='utf-8') as f:
            f.write(updated_content) # type: ignore
        
        print(f"Response successfully saved at {save_path}")

        logger.info(f"Response Metadata: {usage_stats}")
        logger.info(f"Reasoning: {reasoning}")
    
    except Exception as e:
        print(f"Sorry, the request could not be completed.  This is the error: {e}.  Please take care of this and try again")
        logger.exception(e)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Takes your string content and organizes it in markdown format to prep it for ingestion into a vector store")
    parser.add_argument('--file-path', type=str, default=None, help='Path to the content you wish to have organized')
    parser.add_argument('--model', type=str, default="openai/gpt-oss-120b", help='LLM to use')

    args = parser.parse_args()

    file_path = args.file_path
    model = args.model
    kwargs = {}

    # TODO: In future, pass categories from clean.py step to prep.py for dynamic categorization
    # For now, using DEFAULT_CATEGORIES defined at top of file
    categories = None  # Will use DEFAULT_CATEGORIES in prepare_web_content()

    # Build initial prompt to show user
    try:
        preview_prompt = build_prompt(prep_parts, categories=DEFAULT_CATEGORIES)
        print(f"\n#### CURRENT PROMPT (Preview) ####\n\n{preview_prompt[:500]}...")
    except Exception as e:
        logger.error(f"Error building preview prompt: {e}")
        print("\nCould not generate prompt preview")

    if not file_path:
        file_path = input("\nPlease enter the file path to the content you wish to prepare for vector storage: ")

    if model == "openai/gpt-oss-120b":

        customize = input("\nWould you like to choose a different model to use? Default is openai/gpt-oss-120b?  Type Yes or No: ")
        if customize.lower() in ['yes', 'y', 'yeah', 'ya']:
            model = input("\nOk, enter a valid model name: ")

    add_kwargs = input("\nDo you wish to customize any other arguments in the chat_completions method for the client?  Type Yes or No: ")

    if add_kwargs.lower() in ['yes', 'y', 'yeah', 'ya']:

        kwargs_list = input("\nEnter your args in this format, arg_name1=value1,arg_name2=value2: ")

        for kwarg in kwargs_list.split(","):
            dict_pair = parse_value(kwarg)
            if dict_pair and len(dict_pair) == 2:
                kwargs[dict_pair[0]] = dict_pair[1]

    prepare_web_content(
        file_path=file_path,
        categories=categories,
        model=model,
        **kwargs # type: ignore
        )
