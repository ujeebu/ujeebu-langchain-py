#!/usr/bin/env python3
"""
Example: Web Scraping with Ujeebu Scrape API

This example demonstrates how to use the UjeebuScrapeTool to scrape
web pages with JavaScript rendering, extract structured data, and more.

Prerequisites:
    - pip install langchain-ujeebu langchain-openai
    - Set UJEEBU_API_KEY environment variable
    - Set OPENAI_API_KEY environment variable (for LLM examples)

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
import json
from langchain_ujeebu import UjeebuScrapeTool

# Ensure API key is set
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    print("Get your API key at: https://ujeebu.com/signup")
    exit(1)


def basic_scraping():
    """Basic web page scraping example."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Web Page Scraping")
    print("=" * 60)

    # Initialize the scrape tool
    scrape_tool = UjeebuScrapeTool()

    # Scrape a web page with JavaScript rendering
    url = "https://news.ycombinator.com"
    result = scrape_tool._run(url=url, js_rendering=True)

    print(f"\nScraped content from: {url}")
    print("-" * 40)
    print(result[:2000] + "..." if len(result) > 2000 else result)


def scrape_with_extraction_rules():
    """Extract structured data using CSS selectors."""
    print("\n" + "=" * 60)
    print("Example 2: Extract Structured Data with CSS Selectors")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    # Define extraction rules to get specific data
    # This extracts titles and links from Hacker News
    extraction_rules = {
        "titles": {
            "selector": ".titleline > a",
            "type": "obj",
            "multiple": True,
            "children": {
                "text": {"selector": "&", "type": "text"},
                "href": {"selector": "&", "type": "link"},
            },
        },
        "points": {"selector": ".score", "type": "text", "multiple": True},
    }

    url = "https://news.ycombinator.com"
    result = scrape_tool._run(
        url=url, js_rendering=True, extract_rules=json.dumps(extraction_rules)
    )

    print(f"\nExtracted data from: {url}")
    print("-" * 40)
    print(result)


def scrape_ecommerce_product():
    """Scrape product information from an e-commerce site."""
    print("\n" + "=" * 60)
    print("Example 3: Scrape E-commerce Product Data")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    # Extract product details from a book product page
    extraction_rules = {
        "product_title": {
            "selector": "h1",
            "type": "text",
        },
        "price": {"selector": ".price_color", "type": "text"},
        "description": {
            "selector": "#product_description ~ p",
            "type": "text",
        },
        "availability": {
            "selector": ".availability",
            "type": "text",
        },
    }

    # Example with a sample product page
    url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
    result = scrape_tool._run(
        url=url, js_rendering=True, extract_rules=json.dumps(extraction_rules)
    )

    print(f"\nExtracted product data from: {url}")
    print("-" * 40)
    print(result)


def scrape_with_wait_for():
    """Wait for dynamic content to load before scraping."""
    print("\n" + "=" * 60)
    print("Example 4: Wait for Dynamic Content")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    # Wait for a specific element to appear before scraping
    url = "https://quotes.toscrape.com/js/"
    result = scrape_tool._run(
        url=url,
        js_rendering=True,
        wait_for=".quote",  # Wait for quotes to load
        response_type="text",
    )

    print(f"\nScraped after waiting for .quote: {url}")
    print("-" * 40)
    print(result[:1500] + "..." if len(result) > 1500 else result)


def scrape_with_premium_proxy():
    """Use premium proxies for difficult-to-scrape sites."""
    print("\n" + "=" * 60)
    print("Example 5: Scraping with Premium Proxy")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    # Some sites require premium residential proxies
    url = "https://quotes.toscrape.com/"
    result = scrape_tool._run(
        url=url, js_rendering=True, premium_proxy=True  # Use premium residential proxy
    )

    print(f"\nScraped with premium proxy: {url}")
    print("-" * 40)
    print(result[:500])


def scrape_infinite_scroll():
    """Scrape data from an infinite scroll page."""
    print("\n" + "=" * 60)
    print("Example 6: Scrape Infinite Scroll Page")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    extraction_rules = {
        "quotes": {"selector": ".quote .text", "type": "text", "multiple": True},
        "authors": {"selector": ".quote .author", "type": "text", "multiple": True},
    }

    # Scrape quotes from an infinite scroll demo page
    url = "https://quotes.toscrape.com/scroll"
    result = scrape_tool._run(
        url=url,
        js_rendering=True,
        wait_for=".quote",
        extract_rules=json.dumps(extraction_rules),
    )

    print(f"\nExtracted from infinite scroll: {url}")
    print("-" * 40)
    print(result)


def use_with_langchain_agent():
    """Use scrape tool with a LangChain agent for research."""
    print("\n" + "=" * 60)
    print("Example 7: Using with LangChain Agent for Research")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping agent example - OPENAI_API_KEY not set")
        return

    from langchain_openai import ChatOpenAI
    from langchain_ujeebu import UjeebuExtractTool

    # Create agent with both Extract and Scrape tools
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    extract_tool = UjeebuExtractTool()
    scrape_tool = UjeebuScrapeTool()
    tools = [extract_tool, scrape_tool]

    query = """
        I need to research the top stories on Hacker News today.
        First, scrape the front page of news.ycombinator.com to get the top 3 story titles and links.
        Then summarize what topics are trending.
    """

    # Try LangGraph (recommended) or fallback to legacy
    try:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(llm, tools)

        result = agent.invoke({"messages": [("human", query)]})

        # Get the last AI message
        response = None
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                response = msg.content
                break
        response = response or str(result)

    except ImportError:
        # Fallback to legacy agent
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful research assistant."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = executor.invoke({"input": query})
        response = result.get("output", str(result))

    print("\nAgent Response:")
    print("-" * 40)
    print(response)


if __name__ == "__main__":
    print("Ujeebu Scrape Tool Examples")
    print("===========================")

    # Run examples
    basic_scraping()
    scrape_with_extraction_rules()
    scrape_ecommerce_product()
    scrape_with_wait_for()
    scrape_with_premium_proxy()
    scrape_infinite_scroll()

    # Optional: Agent example
    use_with_langchain_agent()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
