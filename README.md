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
│   ├── paths.py                    # File path configurations
│   └── load_env.py                 # Load your API keys and model configurations
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

High-level overview of the pipeline stages. The system is designed to be **modular**—each stage can be used sequentially or independently.

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

**Input:** Root URL  
**Output:** List of raw HTML/text strings

---

#### 🧹 `clean.py`

Leverages **LLM power** to clean messy scraped content.

**Why use an LLM?** Raw scraped content contains HTML tags, broken formatting, random image links, and dead space. Instead of handling every edge case manually, we let the LLM intelligently extract only the useful content.

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
1. Loads a HuggingFace embedding model
2. Instantiates a vector store with your chosen name and location
3. If a store already exists at the path, it loads that instead of creating a new one

**Input:** Store name, location, embedding model  
**Output:** Initialized Chroma DB vector store

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
from ingestion.scrape import scrape_website

# Scrape content from a website
urls = scrape_website(
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
from ingestion.clean import clean_content

# Clean raw content with LLM
cleaned = clean_content(
    raw_content_list=urls,
    model="groq/llama-3-70b"
)
```

### Example 3: Preparing for Vector Storage

**Via CLI:**
```bash
python -m ingestion.prep
```

**In Python:**
```python
from ingestion.prep import prepare_content

# Organize and format content
prepared = prepare_content(
    cleaned_content=cleaned,
    categories=["Installation", "Usage", "API Reference"]
)
```

### Example 4: Creating a Vector Store

**In Python:**
```python
from vector_store.initialize import initialize_vector_store

# Create a new vector store
vector_store = initialize_vector_store(
    store_name="my_knowledge_base",
    store_path="./data/vector_stores"
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

### YAML Prompt System

All prompts are located in the `prompts/` directory and use YAML format for easy customization.

#### 📝 `components.yaml`

Contains **reusable components** that can be mixed and matched across prompt templates:

- **Tone**: Professional, casual, educational, etc.
- **Reasoning strategy**: Chain-of-thought, step-by-step, etc.
- **Available tools**: Web search, document retrieval, etc.

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
├── ingestion.log       # Web scraping, cleaning, prep
├── vector_store.log    # Database operations
├── rag_assistant.log   # Query processing and responses
└── errors.log          # Critical errors across all components
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

> **💡 Tip:** Check the LLM reasoning logs when tuning prompts—they show the model's thought process and can help identify prompt improvements.

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
- **Larger chunks** (1000-1500 tokens): Better for broader context
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
- [ ] Create prompt template builder/validator

#### **Expanded Tool Access**
- [ ] Web search integration enhancements
- [ ] Calculator and computation tools
- [ ] Code execution sandbox
- [ ] External API integrations

#### **Broader File Support**
- [ ] PDF processing (currently only `.txt` and `.md`)
- [ ] DOCX and other document formats
- [ ] Image and multimodal content
- [ ] Audio transcription and processing

#### **UI Modernization**
- [ ] Enhanced Gradio interface with better styling
- [ ] Conversation history and export
- [ ] Multi-user support
- [ ] Mobile-responsive design
- [ ] Dark mode

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
- **Multi-language**: Primarily English-optimized

### 🔮 Long-term Vision

Transform this into a **production-ready agentic AI system** with:
- Orchestrated multi-agent workflows
- Enterprise-scale document processing
- Real-time knowledge updates
- Advanced monitoring and analytics
- Deployment templates (Docker, K8s)

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

I'd like to thank our AI overlords for their help and service in making projects like this possible. 🤖

---

## 📄 License

This project is licensed under the Apcahe 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References & Resources

### Primary Reference
"Blueprints for Text Analytics Using Python" by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler (O'Reilly, 2021), 978-1-492-07408-3.

### Tools & Frameworks
- [Langchain](https://python.langchain.com/) - LLM application framework
- [Chroma DB](https://www.trychroma.com/) - Vector database
- [Gradio](https://www.gradio.app/) - ML web interfaces
- [Tavily](https://tavily.com/) - Web search API
- [HuggingFace](https://huggingface.co/) - Embedding models

### Further Reading
- [RAG Explained](https://arxiv.org/abs/2005.11401) - Original RAG paper
- [Langchain RAG Tutorials](https://python.langchain.com/docs/use_cases/question_answering/)
- [Building Production RAG Systems](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

**Questions or feedback?** Open an issue or reach out on [GitHub](https://github.com/tkbarb10/ai_essentials_rag)!
