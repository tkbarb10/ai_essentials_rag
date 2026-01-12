from tavily import TavilyClient
from config.load_env import load_env
from utils.kwarg_parser import parse_value
from typing import List, Dict
import os
import json
import math
from config.paths import OUTPUTS_DIR
from urllib.parse import urlparse
from datetime import datetime

load_env()

tavily_client = TavilyClient()

def website_map(root_url: str, instructions: str, max_depth: int=5, include_usage: bool=True, **kwargs):
    """Fetch a crawl map of URLs starting at a root location.

    Args:
        root_url: Starting URL to map.
        instructions: Provider-specific mapping instructions.
        max_depth: Maximum depth to crawl from the root URL.
        include_usage: Whether to include provider usage metadata in the response.
        **kwargs: Additional keyword arguments passed to the map request.

    Returns:
        Mapping results returned by the Tavily client.
    """

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

        return map_results
    
    except Exception as e:
        print("Map quest unsuccessful")
        print(f"Thwarted by: {e}")

# returns a dict object that includes the base_url, results (list of links), usage, response time in seconds and request_id

def extract_links(url_list: List[str]) -> dict:
    """Group URLs into batches of 20 for downstream extraction.

    Args:
        url_list: Flat list of URLs to group.

    Returns:
        Dictionary mapping group names to URL lists.
    """
    url_dict = {}
    n_group = math.ceil(len(url_list) / 20)

    print(f"There are {len(url_list)} urls, dividing into {n_group} groups for content extraction")

    for i in range(n_group):
        n = i * 20
        url_dict[f"group_{i}"] = url_list[n:n + 20]
    
    return url_dict


def extract_content(url_dict: Dict[str, str]):
    """Extract raw markdown content for each URL group.

    Args:
        url_dict: Mapping of group names to lists of URLs.

    Returns:
        Dictionary of extraction results keyed by group.
    """
    result_set = {}

    for key, url_list in url_dict.items():
        
        # Initialize a list to hold results for this specific key
        key_results = []

        for url in url_list:
            try:
                # Note: We pass [url] as a list containing a single string
                result = tavily_client.extract(
                    urls=[url], 
                    extract_depth='advanced',
                    timeout=60,
                    include_usage=True
                )

                key_results.append(result)

            except Exception as e:
                print(f"Skipping individual URL {url} in {key} due to error: {e}")
                continue # Move to the next URL in the list

        result_set[key] = key_results

    return result_set


def raw_web_content(root_url: str, instructions: str, max_depth: int=5, include_usage: bool=True, **kwargs):
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
    map_results = website_map(
        root_url=root_url,
        instructions=instructions,
        max_depth=max_depth,
        include_usage=include_usage,
        **kwargs
    )

    url_list = map_results['results'] # type: ignore

    url_dict = extract_links(
        url_list=url_list
    )

    result_set = extract_content(
        url_dict=url_dict
    )

    content_strings = []

    for key, value in result_set.items():
        for item in value:
            if item.get('results'):
                content_strings.append(item['results'][0]['raw_content'])
    
    return content_strings

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Takes a root url and maps out links to a specificed depth then scrapes the web content from each link")
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
        url = input("Please enter a root url that you wish to map from: ")

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
        print(f"Web content successfully saved at {path}")

    except Exception as e:
        print(f"There was an error saving your file.\nError: {e}")
        content_len = len("".join(content_strings))
        user_request = input(f"Content is {content_len} characters long, do you wish to print to console to copy/paste instead? Type Yes or No: ")

        if user_request.lower() in ['yes', 'y', 'yeah', 'ya']:
            print(content_strings)
    
