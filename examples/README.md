# Ujeebu LangChain Integration Examples

This directory contains comprehensive examples demonstrating how to use the Ujeebu LangChain integration for various web scraping and content extraction tasks.

## Prerequisites

Before running the examples, make sure you have:

1. **Ujeebu API Key**: Get your free API key at [https://ujeebu.com/signup](https://ujeebu.com/signup)
2. **OpenAI API Key** (for agent/RAG examples): Get it at [https://platform.openai.com](https://platform.openai.com)

Set your environment variables:

```bash
export UJEEBU_API_KEY="your-ujeebu-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # For LLM examples
```

## Installation

```bash
# Install the package
pip install langchain-ujeebu

# For agent examples (recommended)
pip install langchain-openai langgraph

# For RAG examples
pip install langchain-openai faiss-cpu

# For data pipeline examples
pip install pandas
```

## Examples Overview

### 1. `extract_article_example.py` - Article Extraction
Demonstrates using `UjeebuExtractTool` to extract clean, structured content from news articles and blog posts.

**Features covered:**
- Basic article extraction
- Extraction with custom options (images, quick mode)
- HTML content extraction
- Batch extraction
- Using with LangChain agents

```bash
python extract_article_example.py
```

### 2. `scrape_website_example.py` - Web Scraping
Shows how to use `UjeebuScrapeTool` for general web scraping with JavaScript rendering support.

**Features covered:**
- Basic web page scraping
- Structured data extraction with CSS selectors
- E-commerce product scraping
- Waiting for dynamic content
- Premium proxy usage
- Infinite scroll handling

```bash
python scrape_website_example.py
```

### 3. `document_loader_example.py` - Document Loading
Demonstrates using `UjeebuLoader` to load web content as LangChain Documents.

**Features covered:**
- Loading single/multiple articles
- Extracting with metadata (author, date, images)
- Text splitting for vector stores
- Creating FAISS vector stores
- Building Q&A chains

```bash
python document_loader_example.py
```

### 4. `agent_example.py` - LangChain Agents
Shows how to build AI agents that can autonomously research the web.

**Features covered:**
- Basic agents with extract tool
- Agents with scrape tool
- Multi-tool agents
- Article comparison
- Task-specific research
- ReAct reasoning agents

```bash
python agent_example.py
```

### 5. `rag_system_example.py` - RAG Systems
Demonstrates building Retrieval-Augmented Generation systems with web content.

**Features covered:**
- Building RAG from multiple articles
- Topic-focused RAG systems
- Incremental document addition
- Custom prompts for domain-specific answers

```bash
python rag_system_example.py
```

### 6. `research_agent_example.py` - Research Agents
Advanced example of building research agents with memory and structured output.

**Features covered:**
- Research agents with conversation memory
- Fact-checking workflows
- Multi-source comparison
- Structured data extraction
- Interactive research sessions

```bash
python research_agent_example.py
```

### 7. `data_extraction_pipeline.py` - Data Pipelines
Shows how to build data extraction pipelines for various use cases.

**Features covered:**
- Book catalog scraping with pagination
- Quote collection with author metadata
- News aggregation from multiple sources
- Price monitoring
- Export to pandas DataFrames

```bash
python data_extraction_pipeline.py
```

## Quick Start

Here's a minimal example to get started:

```python
import os
from langchain_ujeebu import UjeebuExtractTool, UjeebuScrapeTool, UjeebuLoader

# Set API key
os.environ["UJEEBU_API_KEY"] = "your-api-key"

# Extract article content
extract_tool = UjeebuExtractTool()
article = extract_tool._run(url="https://example.com/article")
print(article)

# Scrape a website
scrape_tool = UjeebuScrapeTool()
html = scrape_tool._run(url="https://example.com", js_rendering=True)
print(html)

# Load documents for RAG
loader = UjeebuLoader(urls=["https://example.com/article"])
documents = loader.load()
print(documents)
```

## Extraction Rules Reference

When using `UjeebuScrapeTool` with extraction rules, use this format:

```python
import json

extraction_rules = {
    # Simple text extraction (single element)
    "title": {
        "selector": "h1",
        "type": "text"
    },

    # Multiple text elements
    "paragraphs": {
        "selector": "p",
        "type": "text",
        "multiple": True
    },

    # Extract href attribute (for links)
    "link": {
        "selector": "a.main-link",
        "type": "link"
    },

    # Extract image src
    "image": {
        "selector": "img.hero",
        "type": "image"
    },

    # Extract custom attribute
    "data_id": {
        "selector": "[data-id]",
        "type": "attr",
        "attribute": "data-id"
    },

    # Nested object extraction (multiple items with children)
    "products": {
        "selector": ".product-card",
        "type": "obj",
        "multiple": True,
        "children": {
            "name": {
                "selector": ".title",
                "type": "text"
            },
            "price": {
                "selector": ".price",
                "type": "text"
            },
            "link": {
                "selector": "a",
                "type": "link"
            },
            "image": {
                "selector": "img",
                "type": "image"
            }
        }
    },

    # Use & to reference the parent element itself
    "links": {
        "selector": "a.article-link",
        "type": "obj",
        "multiple": True,
        "children": {
            "text": {
                "selector": "&",  # Gets text from the <a> element itself
                "type": "text"
            },
            "url": {
                "selector": "&",  # Gets href from the <a> element itself
                "type": "link"
            }
        }
    }
}

result = scrape_tool._run(
    url="https://example.com",
    extract_rules=json.dumps(extraction_rules)
)
```

### Extraction Rule Types

| Type | Description |
|------|-------------|
| `text` | Extract text content of the element |
| `link` | Extract href attribute (for links) |
| `image` | Extract src attribute (for images) |
| `attr` | Extract custom attribute (use with `attribute` field) |
| `obj` | Extract nested object (use with `children` field) |

### Options

| Option | Description |
|--------|-------------|
| `selector` | CSS selector to find elements. Use `&` to reference the parent element |
| `type` | Type of extraction (see above) |
| `multiple` | Set to `true` to extract all matching elements as a list |
| `children` | Nested extraction rules for `type: "obj"` |
| `attribute` | Attribute name for `type: "attr"` |

## Troubleshooting

### Rate Limiting
If you're hitting rate limits, add delays between requests:
```python
import time
for url in urls:
    result = tool._run(url=url)
    time.sleep(1)  # 1 second delay
```

### Timeout Errors
Increase the timeout for slow pages:
```python
tool = UjeebuScrapeTool(timeout=180)  # 3 minutes
```

### Anti-Bot Protection
Use premium proxies for difficult sites:
```python
result = scrape_tool._run(url=url, premium_proxy=True)
```

### JavaScript-Heavy Sites
Wait for specific elements to load:
```python
result = scrape_tool._run(url=url, wait_for=".content-loaded")
```

## Support

- **Documentation**: [https://ujeebu.com/docs](https://ujeebu.com/docs)
- **API Reference**: [https://ujeebu.com/docs/api](https://ujeebu.com/docs/api)
- **Issues**: [GitHub Issues](https://github.com/prodacity/ujeebu-forwarder/issues)

