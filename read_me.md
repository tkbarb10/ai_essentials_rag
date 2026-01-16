# RAG Assistant: Building a Complete Pipeline from Web Scraping to Chatbot

## TL;DR / Abstract
- Add more about why it matters

This project outlines the steps to get to get a RAG application up and running from any stage in the process, whether that be just an idea in your head, or you already have an organized vector database and just need an LLM interface to plug it into.  Below I'll give an overview of the point of this project and how it evolved

## Overview

RAG is a method by which we split up documents into vector embeddings for storage in a vector database in order to provide context to a language model at run time to enrich its responses.  In this project we'll walk through each of the steps from the beginning.  We'll start with a simple method to scrape the web to get the raw material for your RAG pipeline.  Then we'll run through a couple LLM-powered methods to clean and organize the raw content.  Depending on the size of corpus you find, this could be a simple showcase of one of the benefits over RAG instead of putting everything in a prompt.  Then we'll walk through how to store this content into a vector database before topping it off with a demonstration with the stored text "Blueprints for Text Analytics in Python" using a Gradio interface.

The reason for that particular text is because I'm currently enrolled in a Masters for Data Science program at University of San Diego and taking a class called Applied Large Language Models for Data Science.  The original syllabus has called for two texts, the one above and another.  But unbeknownst to me, they revamped the syllabus and are now only requiring the other text.  Since I already had the Blueprints one, instead of letting it sit on my hard drive, I figured it would be put to better use in this project where Iu could still learn from it without worrying about reading the whole thing.

## Prerequisites
- Python 3.12.9 or higher
- Required API keys
  - LLM  
    This repo uses Langchain to wrap models, giving you flexibility in the model you choose.  It's currently set up using Groq as a provider, but can be easily switched to any  provider/model that Langchain supports
  - Tavily  
    Tavily is the web search application used for web search.  This is needed to enable the RAG Assistant to have the ability to search the web.  It is also used for scraping the web for content for your vector database
  - HuggingFace  
    A HF log in is needed for the embedding models
- Basic understanding of: LLMs, vector databases, web APIs

## Project Overview & Architecture

High level overview of the steps in the pipeline.  It was designed to be modular so each step can be utilized in sequence or individually.

![RAG Pipeline Diagram](assets/rag_pipeline.svg)

### Stage 1: Ingestion

In the `ingestion/` directory you'll find three scripts: `scrape.py`, `clean.py`, `prep.py`.  All three can be utilized through the CLI or within a notebook

 - `scrape.py`  
    This script uses the Tavily web API to map and extract content from a root URL.  You first provide a root url that you want to start with, and the `.map()` method will extract every url it can find from that url to within a certain depth.  You can provide instructions to the mapper, change the maximum depth it'll search to (default is 5), or adjust any other parameter than the `.map()` accepts.  This will output a link of urls that can then be iterated through to extract their raw content via the `.extract()` method.  The result is a list of raw strings scraped from each url in the list

- `clean.py`  
    The resulting list of content will be messy.  There will be links to random images, html tags, lots of dead space, etc.  Instead of having to figure out every edge case to clean this up, we can leverage the power of LLM's.  This script first creates a list of message payloads for each string and then iterates through making calls to an LLM prompting it to clean up the raw string and returning just the content we're interested in.  This can be a costly process so messages that go beyond your rate limits are skipped.  Your chosen model is pinged first to get the rate limit for your account.  The final result is a single string with headers to denote individual sites.

- `prep.py`  
    After cleaning the strings, there is bound to be a lot of redundant and disorganized content.  So this step utilizes a language model to sort through the provided information, remove redundant and useless content, and organize the remaining into categories that would be useful for storage into a vector database.  These categories can be provided by you, or can be left to the LLM to decide.  The output will be a single string organized in Markdown format (important for the text splitting process)

### Stage 2: Vector Store

There are two scripts here to `initialize` the vector store and `insert` RAG material into it.  Like before, these scripts can be utilized through the CLI or imported into your notebook

- `initialize.py`  
    Using a Langchain wrapper for Chroma DB, this script loads a Hugging Face embedding model and instantiates a vector store with a name and location of your choosing.  If you already have a vector store you'd like to you, just pass the path and name of the store and it will be activated

- `insert.py`  
    This script takes the previous one a step further and also accepts a path to the documents you wish to upload, splits them, and adds to the store.  The splitting process first uses Langchains `MarkdownHeaderTextSplitter` to split the text on headers and subheaders.  The reason for this is because the method automatically recognizes these headers and adds them as metadata.  If your content isn't in Markdown format, don't worry. It'll pass through the markdown splitter into the next step which uses `RecursiveCharacterTextSplitter` which will then chunk the text recursively within the markdown headers or without.  The default chunk is 750 tokens with an overlap between chunks of 150.

### Stage 3: RAG Assistant

There are two classes for this step.  `rag_assistant` binds the previous steps with an LLM to query and can be utilized through the CLI or imported into a notebook.  `gradio_interface` allows you to interact with the RAG Assistant through a `Gradio` app.

- `rag_assistant.py`  
    This script contains a class called RagAssistant.  This combines all the previous steps and has a `topic` and `prompt_template` argument.  The `topic` is a string you can insert describing what your vector store contains (e.g "Blueprints for Text Analytics in Python textbook") and `prompt_template` is the prompt for the 'personality' you want the model to have.  The default is `educational_assistant`.  Once set up, you can ask the model any question you like and it'll respond using the context provided by the vector store

- `gradio_interface.py` -> `app.py`
    This class wraps the RAG Assistant in an interface to adapt it to Gradio.  It can be launched with the `app.py` script through the CLI 

## Installation & Setup

If you don't already have it, install `uv` package then clone the repo
```python
pip install uv

git clone https://github.com/tkbarb10/ai_essentials_rag.git
cd ai_essentials_rag
```

Run `uv venv` to set up your env then `uv sync` to install dependencies

Set up your `.env` file.  You'll need at least one model API key

To run every script except the gradio interface you can access it as a module

```python
python -m directory.script
```

To run the Gradio app

```python
python app.py
```

## Configuration & Customization

- YAML prompt system
    Prompts can be found in the prompts directory.  Feel free to add and adjust as you see fit for your use case

    - `components.yaml`
        Reusable components that can be used across prompt templates such as tone, reasoning strategy, and available tools
    
    - `ingestion_prompts.yaml`
        The prompts used in scraping prepping the data from the web
    
    - `rag_prompts.yaml`
        This file contains the prompt templates that can be swapped in and out depending on the purpose of your rag pipeline.  The two currently available are `qa_assistant` (ie corporate chatbot) and `educational_assistant`

## Logging

Logging is set up at each step to track errors and other metadata (such as LLM reasoning to aid with prompt tuning).  Logs get saved to the [Logs File](outputs/logs/) directory and are saved under different file names depending on which component was logging data

## Troubleshooting

Common errors are issues with rate limits and file paths.  Ensure you're API keys and model choices are correct and check the `paths.py` file in `config/`.

Most likely though the biggest issues you'll find are with the vector store and the LLM responses.  This isn't a rules based process and it may take some iteration to tune the vector store search and queries to your liking.  Play around with how the documents are stored by changing the chunk size and overlap.  Maybe you'll find that adding richer metadata to return to the model will improve response.  Change the vector store search parameters.  The current default is a **similarity** search with **k = 3**, but maybe you'll have better results by increasing **k**.  Or perhaps changing the search type to [max marginal relevance](https://reference.langchain.com/python/langchain_core/vectorstores/#langchain_core.vectorstores.base.VectorStore.max_marginal_relevance_search) will yield better results for you.  Have fun with it!

## Future Directions/Limitations
The goal behind this repo set up was to be able to extend it for multi-agent orchestration for the Ready Tensor Agentic AI in Production certification.  To that end there are some limitations we hope to adjust in the next iteration

- Adding more prompt templates and improving upon existing ones.  
- Extending the number of tools the model has access to
- Currently this project only handles `.txt` and `.md` files so increasing the number of file types that are supported for splitting and ingesting
- The `Gradio` interface at the moment is pretty bare bones, so I hope to improve the interface and modernize it more to make it a legit ui
- The biggest change I'm going to make is adding different RAG strategies.  This is currently a basic RAG pipeline where we query the vector store and pass the context on to the model. Adding different strategies such as graph rag or adaptive rag would widen the potential use cases

## Contributing

I'd like to thank our AI overlords for their help and service in making projects like this possible. 

## License

[License](LICENSE)

## References & Resources

"Blueprints for Text Analytics Using Python by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler (O'Reilly, 2021), 978-1-492-07408-3."
