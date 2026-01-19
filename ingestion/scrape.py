from tavily import TavilyClient
from config.load_env import load_env
from utils.kwarg_parser import parse_value
from typing import List, Dict, Optional
import os
import json
from config.paths import OUTPUTS_DIR
from urllib.parse import urlparse
from datetime import datetime
from utils.logging_helper import setup_logging

logger = setup_logging(name = "web_scraping")

load_env()

tavily_client = TavilyClient()

def website_map(root_url: str, instructions: str='Avoid returning utility links', max_depth: int=5, include_usage: bool=True, **kwargs):
    """Fetch a crawl map of URLs starting at a root location.

    Args:
        root_url: Starting URL to map.
        instructions: Provider-specific mapping instructions.
        max_depth: Maximum depth to crawl from the root URL.
        include_usage: Whether to include provider usage metadata in the response.
        **kwargs: Additional keyword arguments passed to the map request.

    Returns:
        Dictionary of mapping results returned by the Tavily client.
    """
    logger.info(f"root_url: {root_url}\ninstructions: {instructions}\nmax_depth: {max_depth}\nOther key word args: {kwargs}")

    try:
        print("Beginning map quest...")

        map_results = tavily_client.map(
            root_url,
            instructions=instructions,
            max_depth=max_depth,
            include_usage=include_usage,
            **kwargs
            )

        print("Map quest completed!")
        logger.info(f"Response Time: {map_results.get("response_time")}\nAPI Credits: {map_results.get('usage')}")

        return map_results['results']
    
    except Exception as e:
        print("Map quest unsuccessful")
        logger.error(f"Thwarted by: {e}", exc_info=True)

# returns a dict object that includes the base_url, results (list of links), usage, response time in seconds and request_id

# def extract_links(url_list: List[str]) -> dict:
#     """Group URLs into batches of 20 for downstream extraction.  The tavily.extract() method only accepts 20 links at a time

#     Args:
#         url_list: Flat list of URLs to group.

#     Returns:
#         Dictionary mapping group names in the format "group_n" to URL lists.
#     """
#     url_dict = {}
#     n_group = math.ceil(len(url_list) / 20)

#     print(f"\nThere are {len(url_list)} urls, dividing into {n_group} groups for content extraction")
#     logger.info(f"There are {len(url_list)} urls, batching into {n_group}")

#     for i in range(n_group):
#         n = i * 20
#         url_dict[f"group_{i}"] = url_list[n:n + 20]
    
#     return url_dict


def extract_content(url_list: List[str]) -> List[Dict]:
    """Extract raw markdown content for each URL group.

    Args:
        url_dict: Mapping of group names to lists of URLs.

    Returns:
        Dictionary of extraction results keyed by group.
    """
    result_set = []
    response_time = 0
    usage = 0

    for i, url in enumerate(url_list):
        try:
            result = tavily_client.extract(
                urls=url, 
                extract_depth='basic',
                timeout=60,
                include_usage=True
            )

            result_set.extend(result['results'])
            response_time += result.get("response_time", 0)
            usage += result.get('usage', 0).get('credits', 0)

            if result.get('failed_results'):
                logger.info(f"Failed Link: {result.get('failed_results')}")

        except Exception as e:
            print(f"\nError extracting from link {url} at index {i}")
            logger.error(f"Error: {e}", exc_info=True)

    logger.info(f"Response time to extract {len(result_set)} links: {response_time}\nCredit Usage: {usage}")
    return result_set


def raw_web_content(root_url: str, instructions: str="Avoid returning utility links", max_depth: int=5, include_usage: bool=True, **kwargs):
    """Map a site, extract markdown content, and return raw strings.

    Args:
        root_url: Starting URL to map.
        instructions: Provider-specific mapping instructions.
        max_depth: Maximum depth to crawl from the root URL.
        include_usage: Whether to include provider usage metadata in the response.
        **kwargs: Additional keyword arguments passed to the map request.

    Returns:
        List of raw markdown content strings from extracted pages.
    """
    url_list = website_map(
        root_url=root_url,
        instructions=instructions,
        max_depth=max_depth,
        include_usage=include_usage,
        **kwargs
    )

    # url_dict = extract_links(
    #     url_list=url_list
    # )

    print("\nExtracting content, stand by...")
    result_set = extract_content(
        url_list=url_list # type: ignore
    )

    print("\nParsing results...")

    content_strings = []

    for item in result_set:
        content_strings.append(item.get('raw_content'))

    print("\nAll done!")
    
    return content_strings

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Takes a root url and maps out links to a specified depth then scrapes the web content from each link")
    parser.add_argument('--url', type=str, default=None, help='Root URL to scrape')
    parser.add_argument('--instructions', type=str, default=None, help='Instructions for the mapping process')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum crawl depth')
    parser.add_argument('--include-usage', action="store_true", default=None, help='Include usage metadata')

    args = parser.parse_args()

    url = args.url
    instructions = args.instructions
    max_depth = args.max_depth
    include_usage = args.include_usage
    kwargs = {}

    if not url:
        url = input("\nPlease enter a root url that you wish to map from: ")

    if not instructions:
        instructions = input("\nPlease enter instructions that you wish to pass to the crawler if you'd like to filter out certain links or focus on specific types of content: ")

    if not max_depth:

        customize = input("\nWould you like to adjust max_depth (how far from the root url the crawler will explore.  Default is 5)?  Type Yes or No: ")
        if customize.lower() in ['yes', 'y', 'yeah', 'ya']:
            max_depth = int(input("\nOk, enter a number between 1 and 5 that you'd like the crawler to explore: "))
        else:
            max_depth = 5
    
    if not include_usage:

        customize = input("\nUsage metadata is included by default. Do you want to EXCLUDE it? Type Yes or No: ")
        if customize.lower() in ['yes', 'y', 'yeah', 'ya']:
            include_usage = False
    
    add_kwargs = input("\nDo you wish to customize any other arguments in the map() method for the Tavily Client?  Type Yes or No: ")

    if add_kwargs.lower() in ['yes', 'y', 'yeah', 'ya']:

        kwargs_list = input("\nEnter your args in this format, arg_name1=value1,arg_name2=value2: ")

        for kwarg in kwargs_list.split(","):
            dict_pair = parse_value(kwarg)
            if dict_pair and len(dict_pair) == 2:
                kwargs[dict_pair[0]] = dict_pair[1]

    print("\nStarting extraction process now.  Depending on the website and depth, this could take a few minutes\n")

    content_strings = raw_web_content(
        root_url=url, 
        instructions=instructions, 
        max_depth=max_depth, 
        include_usage=include_usage, 
        **kwargs
        )

    print(f"\n##### PREVIEW OF SCRAPED CONTENT #####\n\n{content_strings[0][:250]}")

    domain = urlparse(url).netloc
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUTS_DIR, f"{domain}_raw_content_{timestamp}.json")

    try:
        with open(path, "w", encoding='utf-8') as f:
            json.dump(content_strings, f, ensure_ascii=False)
        print(f"\nWeb content successfully saved at {path}")

    except Exception as e:
        print(f"There was an error saving your file.\nError: {e}")
        content_len = len("".join(content_strings))
        user_request = input(f"Content is {content_len} characters long, do you wish to print to console to copy/paste instead? Type Yes or No: ")

        if user_request.lower() in ['yes', 'y', 'yeah', 'ya']:
            print(content_strings)
    
