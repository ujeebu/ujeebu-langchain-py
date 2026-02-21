#!/usr/bin/env python3
"""
Example: LangChain Agent with Ujeebu Tools

This example demonstrates how to build AI agents that can:
- Extract article content for analysis
- Scrape websites with JavaScript rendering
- Perform web research tasks autonomously

Prerequisites:
    - pip install langchain-ujeebu langchain-openai langgraph
    - Set UJEEBU_API_KEY environment variable
    - Set OPENAI_API_KEY environment variable

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
from langchain_ujeebu import UjeebuExtractTool, UjeebuScrapeTool
from langchain_openai import ChatOpenAI

# Check for required API keys
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    print("Get your API key at: https://ujeebu.com/signup")
    exit(1)

if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)


def create_agent(tools, llm):
    """Create a LangGraph agent with the given tools."""
    try:
        # Try LangGraph (recommended for new projects)
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(llm, tools)
    except ImportError:
        # Fallback to legacy agent (for older installations)
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that can use tools to answer questions.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)


def invoke_agent(agent, query):
    """Invoke agent with proper input format."""
    try:
        # LangGraph format
        result = agent.invoke({"messages": [("human", query)]})
        if isinstance(result, dict) and "messages" in result:
            # Get the last AI message
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content:
                    return msg.content
        return str(result)
    except Exception:
        # Legacy format
        result = agent.invoke({"input": query})
        return result.get("output", str(result))


def basic_agent_with_extract():
    """Create a basic agent with article extraction capability."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Agent with Extract Tool")
    print("=" * 60)

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Create the Ujeebu Extract tool
    extract_tool = UjeebuExtractTool()

    # Create agent
    agent = create_agent([extract_tool], llm)

    # Ask the agent to extract and summarize
    print("\n🤖 Asking agent to extract and summarize an article...")

    response = invoke_agent(
        agent,
        """
        Extract the article from https://ujeebu.com/blog/enhancing-lead-generation-with-web-data-scraping-and-content-extraction-2/
        and provide a 3-sentence summary of the main points.
    """,
    )

    print("\n📝 Agent Response:")
    print("-" * 40)
    print(response)


def agent_with_scrape_tool():
    """Create an agent with web scraping capability."""
    print("\n" + "=" * 60)
    print("Example 2: Agent with Scrape Tool")
    print("=" * 60)

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    scrape_tool = UjeebuScrapeTool()

    agent = create_agent([scrape_tool], llm)

    print("\n🤖 Asking agent to scrape Hacker News...")

    response = invoke_agent(
        agent,
        """
        Scrape the front page of news.ycombinator.com and tell me
        what are the top 5 trending topics today.
    """,
    )

    print("\n📝 Agent Response:")
    print("-" * 40)
    print(response)


def agent_with_both_tools():
    """Create an agent with both Extract and Scrape tools."""
    print("\n" + "=" * 60)
    print("Example 3: Agent with Multiple Tools")
    print("=" * 60)

    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Both tools available
    extract_tool = UjeebuExtractTool()
    scrape_tool = UjeebuScrapeTool()

    agent = create_agent([extract_tool, scrape_tool], llm)

    print("\n🤖 Asking agent to research using both tools...")

    response = invoke_agent(
        agent,
        """
        I want to understand what's happening in tech news today.

        1. First, scrape the Hacker News homepage to see what's trending
        2. If you find an interesting article, extract its full content
        3. Give me a brief summary of what you found
    """,
    )

    print("\n📝 Agent Response:")
    print("-" * 40)
    print(response)


def agent_for_comparison():
    """Use agent to compare information from multiple articles."""
    print("\n" + "=" * 60)
    print("Example 4: Article Comparison Agent")
    print("=" * 60)

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    extract_tool = UjeebuExtractTool()

    agent = create_agent([extract_tool], llm)

    print("\n🤖 Asking agent to compare articles...")

    response = invoke_agent(
        agent,
        """
        Extract these two articles and compare their perspectives:

        1. https://ujeebu.com/blog/enhancing-lead-generation-with-web-data-scraping-and-content-extraction-2/
        2. https://ujeebu.com/blog/how-to-scrape-google-search-results/

        What are the similarities and differences in their coverage?
    """,
    )

    print("\n📝 Agent Response:")
    print("-" * 40)
    print(response)


def agent_with_specific_task():
    """Agent for a specific research task."""
    print("\n" + "=" * 60)
    print("Example 5: Task-Specific Agent")
    print("=" * 60)

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    extract_tool = UjeebuExtractTool()

    agent = create_agent([extract_tool], llm)

    print("\n🤖 Asking agent to find specific information...")

    response = invoke_agent(
        agent,
        """
        Extract the article from https://ujeebu.com/blog/enhancing-lead-generation-with-web-data-scraping-and-content-extraction-2/
        and answer these questions:

        1. Who wrote this article?
        2. When was it published?
        3. What companies are mentioned?
        4. What is the main conclusion?
    """,
    )

    print("\n📝 Agent Response:")
    print("-" * 40)
    print(response)


def direct_tool_usage():
    """Demonstrate direct tool usage without an agent."""
    print("\n" + "=" * 60)
    print("Example 6: Direct Tool Usage (No Agent)")
    print("=" * 60)

    # Initialize tools
    extract_tool = UjeebuExtractTool()
    scrape_tool = UjeebuScrapeTool()

    # Direct extraction
    print("\n📰 Extracting article directly...")
    article = extract_tool._run(url="https://ujeebu.com/blog/how-to-scrape-google-search-results/")
    print(f"Extracted content (first 500 chars):\n{article[:500]}...")

    # Direct scraping with extraction rules
    print("\n🔍 Scraping with extraction rules...")
    import json

    rules = json.dumps(
        {"quotes": {"selector": ".quote .text", "type": "text", "multiple": True}}
    )
    scraped = scrape_tool._run(url="https://quotes.toscrape.com/", extract_rules=rules)
    print(f"Scraped data:\n{scraped[:500]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Agent Examples with Ujeebu Tools")
    print("=" * 60)

    # Direct tool usage (always works, doesn't require langgraph)
    direct_tool_usage()

    # Agent examples (require langgraph or langchain with OpenAI)
    basic_agent_with_extract()
    agent_with_scrape_tool()
    agent_with_both_tools()
    agent_for_comparison()
    agent_with_specific_task()

    print("\n" + "=" * 60)
    print("Agent Examples completed!")
    print("=" * 60)
