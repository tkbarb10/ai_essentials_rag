# RAG Assistant: Building a Complete Pipeline from Web Scraping to Chatbot

![Python Version](https://img.shields.io/badge/python-3.12.9+-blue.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 📋 TL;DR

A complete end-to-end RAG (Retrieval-Augmented Generation) pipeline that scrapes and processes web content, stores it in a vector database, and provides an interactive Gradio chat interface. Built with modularity in mind—use the entire pipeline or pick individual components for your specific needs. Perfect for building domain-specific AI assistants without the hassle of manual data curation.

---

## 📚 Table of Contents

- [What is RAG?](#what-is-rag)
- [Why This Project?](#why-this-project)
- [Features](#features)
- [Installation & Quick Start](#installation--quick-start)
- [Project Structure](#project-structure)
- [Detailed Architecture](#detailed-architecture)
  - [Stage 1: Ingestion](#stage-1-ingestion)
  - [Stage 2: Vector Store](#stage-2-vector-store)
  - [Stage 3: RAG Assistant](#stage-3-rag-assistant)
- [Usage Examples](#usage-examples)
- [Configuration & Customization](#configuration--customization)
- [Logging](#logging)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Future Directions & Limitations](#future-directions--limitations)
- [Contributing](#contributing)
- [License](#license)
- [References & Resources](#references--resources)

---

## 🤖 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a method where documents are split into vector embeddings and stored in a vector database. At runtime, relevant context is retrieved and provided to a language model to enrich its responses with domain-specific knowledge, enabling accurate answers beyond the model's training data.

---

## 💡 Why This Project?

I'm currently enrolled in a Masters program for Data Science at University of San Diego, taking a class called Applied Large Language Models for Data Science. The original syllabus called for two texts, including *"Blueprints for Text Analytics in Python"*. However, the syllabus was revamped and now only requires the other text.

Since I already had the Blueprints book, instead of letting it collect digital dust on my hard drive, I figured it would be put to better use in this project. This way, I could still learn from it without worrying about reading the whole thing and demonstrate a practical RAG implementation in the process.

---

## ✨ Features

- **🌐 Web Scraping**: Automated content extraction from websites using Tavily API
- **🧹 LLM-Powered Cleaning**: Intelligent content cleanup and organization using language models
- **📊 Vector Storage**: Efficient document storage using Chroma DB with HuggingFace embeddings
- **💬 Interactive Chat Interface**: User-friendly Gradio UI for querying your knowledge base
- **🔧 Modular Design**: Use individual components or the complete pipeline
- **📝 Flexible Prompts**: YAML-based prompt templates for easy customization
- **📈 Comprehensive Logging**: Track errors, metadata, and LLM reasoning for debugging
- **🔄 Multiple RAG Strategies**: Customizable search and retrieval approaches

---

## 🚀 Installation & Quick Start

### Prerequisites

- **Python 3.12.9** or higher
- Basic understanding of: LLMs, vector databases, web APIs

### Required API Keys

You'll need API keys from the following providers:

| Provider | Purpose | Get Your Key |
|----------|---------|--------------|
| **Groq** | LLM provider (via Langchain). Can be swapped for any Langchain-supported provider/model | [Get Groq API Key](https://console.groq.com/keys) |
| **Tavily** | Web search and scraping for content extraction and RAG assistant web search capability | [Get Tavily API Key](https://app.tavily.com/home) |
| **HuggingFace** | Embedding models for vector storage | [Get HF Token](https://huggingface.co/settings/tokens) |

### Installation Steps

**1. Install UV package manager** (if you don't have it):
```bash
pip install uv
```

**2. Clone the repository**:
```bash
git clone https://github.com/tkbarb10/ai_essentials_rag.git
cd ai_essentials_rag
```

**3. Set up your virtual environment**:
```bash
uv venv
uv sync
```

**4. Configure your environment variables**:

Create a `.env` file in the root directory with your API keys:

```env
# LLM Provider (Groq example)
GROQ_API_KEY=your_groq_api_key_here

# Web Search & Scraping
TAVILY_API_KEY=your_tavily_api_key_here

# Embeddings
HUGGINGFACE_TOKEN=your_hf_token_here
```

**5. Verify installation**:

Run the Gradio app to test your setup:
```bash
python app.py
```

The interface should launch in your browser at `http://localhost:7860`

> **✅ Success!** If you see the Gradio interface, you're ready to start building your RAG pipeline.

---

## 📁 Project Structure

```
ai_essentials_rag/
├── app.py                          # Gradio app entry point
├── .env                            # API keys and configuration (create this)
├── pyproject.toml                  # Project dependencies
├── README.md                       # This file
│
├── config/                         # Configuration files
│   ├── settings.yaml               # Centralized settings (models, chunking, app config)
│   ├── paths.py                    # File path configurations
│   ├── types.py                    # Type definitions (ComponentsDict, etc.)
│   └── load_env.py                 # Load API keys and settings exports
│
├── ingestion/                      # Stage 1: Data ingestion
│   ├── scrape.py                   # Web scraping with Tavily
│   ├── clean.py                    # LLM-powered content cleaning
│   └── prep.py                     # Content organization & formatting
│
├── vector_store/                   # Stage 2: Vector database
│   ├── initialize.py               # Create/load vector store
│   └── insert.py                   # Document splitting & insertion
│
├── rag_assistant/                  # Stage 3: RAG interface
│   ├── rag_assistant.py            # Core RAG assistant class
│   └── gradio_interface.py         # Gradio UI wrapper
│
├── prompts/                        # Prompt templates
│   ├── components.yaml             # Reusable prompt components
│   ├── ingestion_prompts.yaml      # Data processing prompts
│   └── rag_prompts.yaml            # RAG assistant personalities
│
├── outputs/                        # Generated outputs
│   ├── logs/                       # Application logs
│   ├── scraped_content.txt        # Raw scraped data
│   └── processed_content.txt      # Cleaned & organized data
│
├── data/                           # Input data
│   └── your_file/                 
│
└── assets/                         # Project assets
    └── rag_pipeline.svg            # Architecture diagram
```

---

## 🏗️ Detailed Architecture

High-level overview of the pipeline stages. The system is designed to be **modular**.  Each stage can be used sequentially or independently.

![RAG Pipeline Diagram](assets/rag_pipeline.svg)

### Stage 1: Ingestion

Located in the `ingestion/` directory. All three scripts can be used via CLI or imported into a notebook.

#### 🌐 `scrape.py`

Uses the **Tavily API** to map and extract content from websites.

**How it works:**
1. Provide a root URL to start from
2. The `.map()` method extracts every URL found from that page up to a specified depth (default: 5 levels)
3. The `.extract()` method iterates through the URL list and retrieves raw content

**Key Features:**
- Configurable search depth
- Custom mapping instructions
- Outputs list of raw content strings from each URL
- Documentation for how to use the method and other arguments to pass [tavily.map](https://docs.tavily.com/documentation/api-reference/endpoint/map)

**Input:** Root URL  
**Output:** List of raw HTML/text strings

---

#### 🧹 `clean.py`

Leverages an LLM to declutter the scraped content.

**Why use an LLM?** Raw scraped content contains HTML tags, broken formatting, random image links, and dead space. Instead of handling every edge case manually, we let the LLM deal with extracting only the useful content.

**How it works:**
1. Creates message payloads for each raw string
2. Iterates through and prompts the LLM to clean each string
3. Respects rate limits (checks your account limits first and skips messages that exceed them)
4. Combines cleaned content into a single string with headers denoting individual sites

**Input:** List of raw content strings  
**Output:** Single cleaned string with site headers

---

#### 📊 `prep.py`

Uses an **LLM to organize and deduplicate** cleaned content for optimal vector storage.

**How it works:**
1. Analyzes the cleaned content to identify redundant or useless information
2. Removes duplicates and irrelevant content
3. Organizes remaining content into categories (you can specify categories or let the LLM decide)
4. Formats output in **Markdown** (important for the text splitting process in Stage 2)

**Input:** Cleaned content string  
**Output:** Organized Markdown-formatted document

---

### Stage 2: Vector Store

Located in the `vector_store/` directory. Scripts can be used via CLI or imported into notebooks.

#### 🗄️ `initialize.py`

Creates or loads a **Chroma DB vector store** using Langchain wrappers.

**How it works:**
1. Loads a HuggingFace embedding model. See this link for [model_kwargs](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer) and [encode_kwargs](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode) to customize
2. Instantiates a vector store with your chosen name and location
3. If a store already exists at the path, it loads that instead of creating a new one

**Input:** Store name, location, embedding model  
**Output:** Initialized Chroma DB vector store

> **💡 Note:** How to configure the [search space](https://docs.trychroma.com/docs/collections/configure#hnsw-index-configuration) for your collection

---

#### 📥 `insert.py`

Processes documents and adds them to your vector store.

**How it works:**

**Two-stage splitting process:**

1. **MarkdownHeaderTextSplitter**
   - Splits text on headers and subheaders
   - Automatically adds headers as metadata
   - Preserves document structure

2. **RecursiveCharacterTextSplitter**
   - Chunks text within markdown sections (or entire doc if no markdown)
   - Default: 750 tokens per chunk
   - 150 token overlap between chunks for context continuity

> **💡 Note:** If your content isn't in Markdown format, it passes through the first splitter harmlessly and gets chunked by the recursive splitter.

**Input:** Document path, vector store  
**Output:** Documents split and stored in vector database

---

### Stage 3: RAG Assistant

Located in the `rag_assistant/` directory. Combines previous stages with an LLM for querying.

#### 🤖 `rag_assistant.py`

The **RagAssistant class** brings everything together.

**Key Parameters:**
- **`topic`**: Description of what your vector store contains  
  *Example:* `"Blueprints for Text Analytics in Python textbook"`
  
- **`prompt_template`**: The 'personality' you want the assistant to have  
  *Default:* `educational_assistant`  
  *Available:* `educational_assistant`, `qa_assistant`

**How it works:**
1. Accepts user questions
2. Queries the vector store for relevant context
3. Combines context with the prompt template
4. Returns LLM-generated response based on retrieved information

**Input:** User question  
**Output:** Context-aware LLM response

---

#### 🖥️ `gradio_interface.py` → `app.py`

Wraps the RAG Assistant in a **Gradio web interface** for easy interaction.

**Features:**
- Clean chat interface
- Conversation history
- Easy deployment

**Launch:**
```bash
python app.py
```

The app will be accessible at `http://localhost:7860`

---

## 💻 Usage Examples

### Running Scripts as Modules

All scripts except the Gradio interface can be run as modules:

```bash
python -m directory.script
```

### Example 1: Scraping Web Content

**Via CLI:**
```bash
python -m ingestion.scrape
```

**In Python:**
```python
from ingestion.scrape import raw_web_content

# Scrape content from a website
urls = raw_web_content(
    root_url="https://example.com",
    max_depth=3,
    instructions="Focus on documentation pages"
)
```

### Example 2: Cleaning Scraped Content

**Via CLI:**
```bash
python -m ingestion.clean
```

**In Python:**
```python
from ingestion.clean import cleaned_content

# Clean raw content with LLM
cleaned = cleaned_content(
    raw_content_list=urls
    prompt=scrape_prompt
)
```

### Example 3: Preparing for Vector Storage

**Via CLI:**
```bash
python -m ingestion.prep
```

**In Python:**
```python
from ingestion.prep import prepare_web_content

# Organize and format content
prepared = prepare_web_content(
    cleaned_content=cleaned,
    categories=["Installation", "Usage", "API Reference"]
)
```

### Example 4: Creating a Vector Store

**In Python:**
```python
from vector_store.initialize import create_vector_store

# Create a new vector store
vector_store = create_vector_store(
    persist_path="./data/vector_stores"
    collection_name="my_knowledge_base",
    embedding_model=embedding_model
)
```

### Example 5: Inserting Documents

**In Python:**
```python
from vector_store.insert import insert_documents

# Add documents to vector store
insert_documents(
    document_path="./outputs/processed_content/organized_content.md",
    vector_store=vector_store,
    chunk_size=750,
    chunk_overlap=150
)
```

### Example 6: Querying the RAG Assistant

**In Python:**
```python
from rag_assistant.rag_assistant import RagAssistant

# Initialize assistant
assistant = RagAssistant(
    topic="Blueprints for Text Analytics in Python textbook",
    prompt_template="educational_assistant"
)

# Ask questions
response = assistant.query("What is tokenization?")
print(response)
```

### Example 7: Launching the Gradio Interface

**Via CLI:**
```bash
python app.py
```

Then open your browser to `http://localhost:7860` and start chatting!

---

## ⚙️ Configuration & Customization

### Centralized Settings

All core application settings are centralized in a single configuration file located at `config/settings.yaml`. This makes it easy to customize the behavior of the entire pipeline without modifying code.

**Location:** `config/settings.yaml`

#### Settings Overview

| Section | Purpose | Key Settings |
|---------|---------|--------------|
| `MODEL_CONFIG` | LLM configuration | `model`, `model_provider`, `temperature`, `max_retries` |
| `EMBEDDING_MODEL` | Vector embedding model | HuggingFace model name |
| `TEXT_SPLIT` | Document chunking | `chunk_size`, `chunk_overlap`, `headers_to_split_on` |
| `VECTOR_STORE` | Chroma DB defaults | `default_persist_path`, `default_collection_name`, `collection_metadata` |
| `RAG` | RAG assistant defaults | `default_n_results`, `default_prompt_template` |
| `APP` | Gradio app configuration | `topic`, `collection_name`, `components`, `gradio` settings |

#### Example Configuration

```yaml
MODEL_CONFIG:
  model: openai/gpt-oss-20b
  model_provider: groq
  temperature: 0.5
  reasoning_effort: medium
  max_retries: 2

EMBEDDING_MODEL: google/embeddinggemma-300m

TEXT_SPLIT:
  chunk_size: 1000
  chunk_overlap: 150
  headers_to_split_on:
    - ["#", "Header 1"]
    - ["##", "Header 2"]
    - ["###", "Header 3"]

VECTOR_STORE:
  default_persist_path: "./chroma/rag_material"
  default_collection_name: "default_collection"
  collection_metadata:
    hnsw:space: "cosine"

RAG:
  default_n_results: 3
  default_prompt_template: "educational_assistant"

APP:
  topic: "Blueprint Text Analytics in Python textbook"
  collection_name: "blueprint_text_analytics"
  components:
    tones: "conversational"
    reasoning_strategies: "Self-Ask"
    tools: true
  gradio:
    title: "Blueprints for Text Analytics in Python Textbook"
    description: "Ask questions about NLP solutions for real world problems"
    server_name: "127.0.0.1"
    server_port: 7860
    share: true
    examples:
      - "How can I build a simple preprocessing pipeline for text data?"
      - "What are n-grams and how are they relevant to machine learning?"
```

#### How to Use

Settings are automatically loaded and exported from `config/load_env.py`. Import the settings you need:

```python
from config.load_env import MODEL_CONFIG, EMBEDDING_MODEL, TEXT_SPLIT, VECTOR_STORE, RAG, APP

# Use settings directly
print(f"Using model: {MODEL_CONFIG['model']}")
print(f"Chunk size: {TEXT_SPLIT['chunk_size']}")
```

#### Common Customizations

**Change the LLM model:**
```yaml
MODEL_CONFIG:
  model: meta-llama/llama-3-70b
  model_provider: groq
  temperature: 0.7
```

**Adjust chunking for longer documents:**
```yaml
TEXT_SPLIT:
  chunk_size: 1500
  chunk_overlap: 200
```

**Configure the Gradio app for a different domain:**
```yaml
APP:
  topic: "Your Custom Knowledge Base"
  collection_name: "your_collection"
  components:
    tones: "professional"
    reasoning_strategies: "CoT"
    tools: false
  gradio:
    title: "Your Custom Assistant"
    description: "Ask questions about your domain"
```

> **💡 Tip:** Changes to `settings.yaml` take effect the next time you run any script or restart the Gradio app. No code changes required!

---

### YAML Prompt System

All prompts are located in the `prompts/` directory and use YAML format for easy customization.

#### 📝 `components.yaml`

Contains **reusable components** that can be mixed and matched across prompt templates:

- **Tone**: Professional, casual, educational, etc.
- **Reasoning strategy**: Chain-of-thought, step-by-step, etc.
- **Available tools**: Web search

**Example structure:**
```yaml
tones:
  educational: "Explain concepts clearly with examples..."
  professional: "Maintain formal business communication..."

reasoning_strategies:
  step_by_step: "Break down your response into clear steps..."
  chain_of_thought: "Show your reasoning process..."
```

---

#### 🌐 `ingestion_prompts.yaml`

Prompts used for **scraping and preparing data** from the web:

- Content cleaning instructions
- Organization strategies
- Category generation prompts

**Customization:** Adjust these to change how content is processed and organized for your specific use case.

---

#### 🤖 `rag_prompts.yaml`

Prompt templates that define your **RAG assistant's personality**:

**Available Templates:**

1. **`educational_assistant`**  
   - Patient, detailed explanations
   - Uses examples and analogies
   - Encourages learning and understanding

2. **`qa_assistant`**  
   - Concise, direct answers
   - Professional corporate chatbot style
   - Focuses on quick, accurate responses

**Creating Custom Templates:**

```yaml
custom_assistant:
  system: |
    You are a [role] specializing in [domain].
    Your goal is to [objective].
    
  tone: professional
  reasoning: step_by_step
  tools:
    - web_search
    - document_retrieval
```

Simply add your custom template to the file and reference it when initializing the RAG Assistant:

```python
assistant = RagAssistant(
    topic="Your topic",
    prompt_template="custom_assistant"
)
```

---

### Vector Store Configuration

Customize how documents are stored and retrieved:

**Chunking Parameters:**
- `chunk_size`: Number of tokens per chunk (default: 750)
- `chunk_overlap`: Token overlap between chunks (default: 150)

**Search Parameters:**
- Search type: `similarity` (default), `mmr` (max marginal relevance)
- `k`: Number of documents to retrieve (default: 3)

**Example:**
```python
from vector_store.insert import insert_documents

insert_documents(
    document_path="./my_docs.md",
    vector_store=store,
    chunk_size=1000,  # Larger chunks
    chunk_overlap=200  # More overlap
)
```

---

## 📊 Logging

Comprehensive logging is set up at each stage to track errors, metadata, and LLM reasoning.

### Log Locations

All logs are saved to the `outputs/logs/` directory with different files based on component:

```
outputs/logs/
├── prompt_builder.log  # Errors with building the prompt, and outputs final prompt
├── rag_assistant.log   # Query processing and responses
└── gradio_logs.log      # Errors associated with loading and using the app
```

### What Gets Logged

- **Ingestion**: URLs scraped, cleaning progress, rate limit hits
- **Vector Store**: Documents inserted, chunk counts, retrieval queries
- **RAG Assistant**: User queries, retrieved context, LLM reasoning chains
- **Errors**: Stack traces, API failures, configuration issues

### Log Levels

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (e.g., approaching rate limits)
- `ERROR`: Error messages that don't stop execution
- `CRITICAL`: Critical errors that halt the process

> **💡 Tip:** Check the LLM reasoning logs when tuning prompts. They show the model's 'thought' process and can help identify prompt improvements.

---

## 🧪 Tests

The `tests/` directory contains both unit and integration tests using **pytest**.

**Run all tests:**
```bash
pytest
```

**Run only unit tests:**
```bash
pytest tests/test_unit_*.py
```

**Run only integration tests:**
```bash
pytest tests/test_integration_*.py
```

**Run with verbose output:**
```bash
pytest -v
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### ❌ API Key Errors

**Problem:** `Authentication failed` or `Invalid API key`

**Solutions:**
1. Verify your `.env` file is in the root directory
2. Check that API keys are correctly formatted (no extra spaces)
3. Confirm keys are active on the provider's dashboard
4. Try regenerating the API key

---

#### ❌ Rate Limit Errors

**Problem:** `Rate limit exceeded` during cleaning/processing

**Solutions:**
1. The `clean.py` script automatically checks rate limits and skips messages that exceed them
2. Wait for your rate limit to reset (usually hourly)
3. Consider upgrading your API plan
4. Process content in smaller batches

---

#### ❌ File Path Issues

**Problem:** `FileNotFoundError` or `Path not found`

**Solutions:**
1. Check the `config/paths.py` file for correct path configurations
2. Ensure you're running scripts from the project root directory
3. Verify that output directories exist (create them if needed)
4. Use absolute paths if relative paths are causing issues

---

#### ❌ Vector Store Not Returning Results

**Problem:** RAG assistant returns generic responses or "I don't know"

**Solutions:**

**1. Check if documents were actually inserted:**
```python
# Verify document count
print(f"Documents in store: {vector_store._collection.count()}")
```

**2. Adjust search parameters:**
```python
# Increase number of retrieved documents
retriever = vector_store.as_retriever(
    search_type="similarity",
    k=5  # Try increasing from default of 3
)

# Or try max marginal relevance search
retriever = vector_store.as_retriever(
    search_type="mmr",
    k=3,
    fetch_k=10  # Fetch more candidates before filtering
)
```

**3. Tune chunking strategy:**
- **Smaller chunks** (400-500 tokens): Better for precise, specific queries
- **Larger chunks** (1000-1500 tokens): Better for broader context (like if you have a textbook stored)
- **More overlap** (200-300 tokens): Better context continuity

**4. Add richer metadata:**
Modify the splitting process to include more metadata fields (document source, section titles, dates, etc.)

**5. Review retrieved context:**
```python
# Debug what's being retrieved
docs = vector_store.similarity_search(query, k=3)
for i, doc in enumerate(docs):
    print(f"Doc {i}: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}\n")
```

---

#### ❌ Poor Response Quality

**Problem:** RAG assistant gives irrelevant or low-quality answers

**Solutions:**
1. **Improve your prompts**: Edit templates in `prompts/rag_prompts.yaml`
2. **Check retrieved context**: Use the debug method above to see what context is being passed to the LLM
3. **Adjust embedding model**: Try different HuggingFace embedding models
4. **Review source documents**: Ensure the content in your vector store is high-quality and relevant
5. **Experiment with search types**: Switch between `similarity` and `mmr` search

---

#### ❌ Gradio Interface Won't Launch

**Problem:** `Address already in use` or interface doesn't open

**Solutions:**
```bash
# Specify a different port
python app.py --port 7861

# Or kill existing process on port 7860
lsof -ti:7860 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :7860   # Windows (then kill PID)
```

---

### Still Having Issues?

1. Check the relevant log file in `outputs/logs/`
2. Enable DEBUG logging for more detailed output
3. Verify all dependencies are installed: `uv sync`
4. Ensure Python version is 3.12.9 or higher
5. Try running a minimal example to isolate the issue

> **💡 Pro Tip:** Most issues with RAG systems come down to **vector store tuning**. Don't be afraid to experiment with chunk sizes, overlap, search parameters, and metadata. It's not a rules-based process—iteration is key!

---

## 🚀 Future Directions & Limitations

This project was designed to be extensible for multi-agent orchestration and the Ready Tensor Agentic AI in Production certification. Here are planned improvements and current limitations:

### 🎯 Planned Improvements

#### **Enhanced Prompt System**
- [ ] Add more prompt templates (technical writer, code reviewer, research assistant)
- [ ] Improve existing templates based on user feedback

#### **Expanded Tool Access**
- [ ] Web search integration enhancements
- [ ] Calculator and computation tools
- [ ] Code execution sandbox
- [ ] External API integrations

#### **Broader File Support**
- [ ] PDF processing (currently only `.txt` and `.md`)
- [ ] DOCX and other document formats

#### **Logging Process**
- [ ] Add returned context with query to logging
- [ ] Post processing method for logs to make that data useful

#### **UI Modernization**
- [ ] Enhanced Gradio interface with better styling
- [ ] Conversation history and export
- [ ] Multi-user support
- [ ] Mobile-responsive design

#### **Advanced RAG Strategies**
This is currently a **basic RAG pipeline** (query → retrieve → generate). Future versions will implement:

- **Graph RAG**: Knowledge graph-based retrieval for complex relationships
- **Adaptive RAG**: Dynamic retrieval strategies based on query complexity
- **Hybrid Search**: Combining vector similarity with keyword search
- **Multi-hop Reasoning**: Following chains of reasoning across documents
- **Query Decomposition**: Breaking complex queries into sub-queries
- **Self-RAG**: Model evaluates its own retrieval relevance

### ⚠️ Current Limitations

- **File Format Support**: Limited to text and markdown files
- **RAG Strategy**: Single basic retrieval approach
- **UI**: Minimal Gradio interface
- **Scalability**: Not optimized for very large document collections (>100k documents)
- **Multimodal**: No support for images, audio, or video in RAG context

### 🔮 Long-term Vision

Transform this into a **production-ready agentic AI system** with:
- Orchestrated multi-agent workflows
- Enterprise-scale document processing
- Real-time knowledge updates
- Advanced monitoring and analytics

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new features, documentation improvements, or RAG strategy implementations, I'd love to collaborate.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Special Thanks

I'd like to thank our AI overlords for their help and service in making projects like this possible.

---

## 📄 License

This project is licensed under the Apcahe 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References & Resources

### Primary Reference
"Blueprints for Text Analytics Using Python" by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler (O'Reilly, 2021), 978-1-492-07408-3.

### Model Usage Rights

#### Embedding Model
- **Model**: [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- **Title**: EmbeddingGemma: Powerful and Lightweight Text Representations
- **Authors**: Schechter Vera, Henrique* and Dua, Sahil* and Zhang, Biao and Salz, Daniel and Mullins, Ryan and Raghuram Panyam, Sindhu and Smoot, Sara and Naim, Iftekhar and Zou, Joe and Chen, Feiyang and Cer, Daniel and Lisak, Alice and Choi, Min and Gonzalez, Lucas and Sanseviero, Omar and Cameron, Glenn and Ballantyne, Ian and Black, Kat and Chen, Kaifeng and Wang, Weiyi and Li, Zhe and Martins, Gus and Lee, Jinhyuk and Sherwood, Mark and Ji, Juyeong and Wu, Renjie and Zheng, Jingxiao and Singh, Jyotinder and Sharma, Abheesht and Sreepat, Divya and Jain, Aashi and Elarabawy, Adham and Co, AJ and Doumanoglou, Andreas and Samari, Babak and Hora, Ben and Potetz, Brian and Kim, Dahun and Alfonseca, Enrique and Moiseev, Fedor and Han, Feng and Palma Gomez, Frank and Hernández Ábrego, Gustavo and Zhang, Hesen and Hui, Hui and Han, Jay and Gill, Karan and Chen, Ke and Chen, Koert and Shanbhogue, Madhuri and Boratko, Michael and Suganthan, Paul and Duddu, Sai Meher Karthik and Mariserla, Sandeep and Ariafar, Setareh and Zhang, Shanfeng and Zhang, Shijie and Baumgartner, Simon and Goenka, Sonam and Qiu, Steve and Dabral, Tanmaya and Walker, Trevor and Rao, Vikram and Khawaja, Waleed and Zhou, Wenlei and Ren, Xiaoqi and Xia, Ye and Chen, Yichang and Chen, Yi-Ting and Dong, Zhe and Ding, Zhongli and Visin, Francesco and Liu, Gaël and Zhang, Jiageng and Kenealy, Kathleen and Casbon, Michelle and Kumar, Ravin and Mesnard, Thomas and Gleicher, Zach and Brick, Cormac and Lacombe, Olivier and Roberts, Adam and Sung, Yunhsuan and Hoffmann, Raphael and Warkentin, Tris and Joulin, Armand and Duerig, Tom and Seyedhosseini, Mojtaba
- **Publisher**: Google Deepmind
- **Original Paper**: https://arxiv.org/abs/2509.20354
- **Usage Rights**: Permits commercial use, modification, distribution, and private use. Provided "as-is" with no warranties.

#### LLM Inference
- **Provider**: [Groq](https://groq.com/)
- **Models Used**: `gpt-oss-20b`, `gpt-oss-120b`
- **Terms of Service**: [Groq AI Policy](https://console.groq.com/docs/legal/ai-policy)
- **Key Terms**: Users are responsible for all decisions made based on AI outputs and must verify accuracy for consequential decisions. Prohibits illegal/harmful activities, misinformation, and high-risk automated decisions without human oversight.

### Tools & Frameworks
- [Langchain](https://python.langchain.com/) - LLM application framework
- [Chroma DB](https://www.trychroma.com/) - Vector database
- [Gradio](https://www.gradio.app/) - ML web interfaces
- [Tavily](https://tavily.com/) - Web search API
- [HuggingFace](https://huggingface.co/) - Embedding models

### Further Reading
- [RAG Explained](https://app.readytensor.ai/lessons/introduction-to-rag-retrieval-augumented-generation-systems-aaidc-week2-lecture-4-3Ht58iNXuvS7)
- [Langchain RAG Tutorials](https://python.langchain.com/docs/use_cases/question_answering/)
---

**Questions or feedback?** Open an issue or reach out on [GitHub](https://github.com/tkbarb10/ai_essentials_rag)!

---

## 📬 Contact

For questions, suggestions, or collaboration inquiries, feel free to reach out:

- **Email**: tkbarb12@gmail.com
- **GitHub**: [tkbarb10](https://github.com/tkbarb10)
