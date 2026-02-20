# LangChain Ujeebu Integration

[![PyPI version](https://badge.fury.io/py/langchain-ujeebu.svg)](https://badge.fury.io/py/langchain-ujeebu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official LangChain integration for [Ujeebu Extract API](https://ujeebu.com/docs/extract) - Extract clean, structured content from news articles and blog posts for use with Large Language Models (LLMs) and AI applications.

## Features

- **Easy Integration**: Seamlessly integrate Ujeebu Extract API with LangChain agents and chains
- **Document Loaders**: Load articles as LangChain Documents for use with vector stores and retrievers
- **Agent Tools**: Use Ujeebu Extract as a tool in LangChain agents
- **Rich Metadata**: Extract article text, HTML, author, publication date, images, and more
- **Quick Mode**: Optional fast extraction mode (30-60% faster)
- **Type Safe**: Full type hints and Pydantic validation

## What is Ujeebu Extract?

Ujeebu Extract converts news and blog articles into clean, structured JSON data. It extracts:

- Clean article text and HTML
- Author and publication date
- Title and summary
- Images and media
- RSS feeds
- Site metadata

Perfect for RAG (Retrieval-Augmented Generation) applications, content analysis, and LLM training data.

## Installation

```bash
pip install langchain-ujeebu
```

### Requirements

- Python 3.8 or higher
- LangChain 0.1.0 or higher
- An Ujeebu API key ([Get one here](https://ujeebu.com/signup))

## Quick Start

### Set up your API key

```bash
export UJEEBU_API_KEY="your-api-key"
```

Or set it programmatically:

```python
import os
os.environ["UJEEBU_API_KEY"] = "your-api-key"
```

### Using as an Agent Tool

```python
from langchain_ujeebu import UjeebuExtractTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Initialize the tool
ujeebu_tool = UjeebuExtractTool()

# Create an agent
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools=[ujeebu_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Use the agent
response = agent.invoke({
    "input": "Extract the article from https://example.com/article and summarize it"
})
print(response)
```

### Using the Document Loader

```python
from langchain_ujeebu import UjeebuLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load articles
loader = UjeebuLoader(
    urls=[
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3"
    ]
)
documents = loader.load()

# Create a vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Query the documents
results = vectorstore.similarity_search("What are the main topics?")
```

## Usage Examples

### Basic Article Extraction

```python
from langchain_ujeebu import UjeebuExtractTool

tool = UjeebuExtractTool()
result = tool._run(
    url="https://example.com/article",
    text=True,
    author=True,
    pub_date=True
)
print(result)
```

### Extract with Images

```python
from langchain_ujeebu import UjeebuExtractTool

tool = UjeebuExtractTool()
result = tool._run(
    url="https://example.com/article",
    images=True  # Extract article images
)
```

### Quick Mode for Faster Extraction

```python
from langchain_ujeebu import UjeebuLoader

loader = UjeebuLoader(
    urls=["https://example.com/article"],
    quick_mode=True  # 30-60% faster, slightly less accurate
)
documents = loader.load()
```

### Load with HTML Content

```python
from langchain_ujeebu import UjeebuLoader

loader = UjeebuLoader(
    urls=["https://example.com/article"],
    extract_html=True,  # Include HTML content
    extract_images=True  # Include images
)
documents = loader.load()

# Access metadata
doc = documents[0]
print(f"Title: {doc.metadata['title']}")
print(f"Author: {doc.metadata['author']}")
print(f"Images: {doc.metadata['images']}")
```

### Build a QA System

```python
from langchain_ujeebu import UjeebuLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Load articles
loader = UjeebuLoader(
    urls=[
        "https://example.com/article1",
        "https://example.com/article2"
    ]
)
documents = loader.load()

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
result = qa_chain.invoke({"query": "What are the main points?"})
print(result["result"])
```

## API Reference

### UjeebuExtractTool

A LangChain tool for extracting article content.

**Parameters:**
- `api_key` (str, optional): Ujeebu API key. Defaults to `UJEEBU_API_KEY` environment variable.

**Tool Parameters:**
- `url` (str, required): URL of the article to extract
- `text` (bool): Extract article text (default: True)
- `html` (bool): Extract article HTML (default: False)
- `author` (bool): Extract article author (default: True)
- `pub_date` (bool): Extract publication date (default: True)
- `images` (bool): Extract images (default: False)
- `quick_mode` (bool): Use quick mode for faster extraction (default: False)

### UjeebuLoader

A LangChain document loader for articles.

**Parameters:**
- `urls` (List[str], required): List of article URLs to load
- `api_key` (str, optional): Ujeebu API key
- `extract_text` (bool): Extract article text (default: True)
- `extract_html` (bool): Extract article HTML (default: False)
- `extract_author` (bool): Extract author (default: True)
- `extract_pub_date` (bool): Extract publication date (default: True)
- `extract_images` (bool): Extract images (default: False)
- `quick_mode` (bool): Use quick mode (default: False)

**Methods:**
- `load()`: Load all documents
- `lazy_load()`: Lazy load documents (same as load for this implementation)

**Document Metadata:**
- `source`: Original URL
- `url`: Resolved URL
- `canonical_url`: Canonical URL
- `title`: Article title
- `author`: Article author
- `pub_date`: Publication date
- `language`: Article language
- `site_name`: Site name
- `summary`: Article summary
- `image`: Main image URL
- `images`: List of all image URLs (if extract_images=True)

## Advanced Usage

### Custom API Endpoint

```python
from langchain_ujeebu import UjeebuLoader

loader = UjeebuLoader(
    urls=["https://example.com/article"],
    base_url="https://custom-api.ujeebu.com/extract"
)
```

### Error Handling

```python
from langchain_ujeebu import UjeebuLoader

loader = UjeebuLoader(urls=["https://example.com/article"])

try:
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
except ValueError as e:
    print(f"API key error: {e}")
except Exception as e:
    print(f"Error loading documents: {e}")
```

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=langchain_ujeebu --cov-report=html

# Run type checking
mypy langchain_ujeebu

# Run linting
flake8 langchain_ujeebu
black langchain_ujeebu
```

## Examples

Check out the [examples](./examples) directory for more usage examples:

- [agent_example.py](./examples/agent_example.py) - Using Ujeebu with LangChain agents
- [document_loader_example.py](./examples/document_loader_example.py) - Using the document loader with vector stores

## Pricing

Ujeebu Extract API pricing is based on usage. Check the [pricing page](https://ujeebu.com/pricing) for details.

## Support

- **Documentation**: [https://ujeebu.com/docs/extract](https://ujeebu.com/docs/extract)
- **API Reference**: [https://ujeebu.com/docs](https://ujeebu.com/docs)
- **Support**: [support@ujeebu.com](mailto:support@ujeebu.com)
- **GitHub Issues**: [Report a bug](https://github.com/ujeebu/langchain-ujeebu/issues)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - Build applications with LLMs through composability
- [Ujeebu API](https://ujeebu.com) - Web scraping and content extraction API

## Changelog

### 0.1.0 (2024-12-30)

- Initial release
- UjeebuExtractTool for LangChain agents
- UjeebuLoader document loader
- Full test coverage
- Comprehensive documentation
