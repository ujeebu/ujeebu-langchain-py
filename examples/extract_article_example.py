#!/usr/bin/env python3
"""
Example: Extracting Article Content with Ujeebu Extract API

This example demonstrates how to use the UjeebuExtractTool to extract
clean, structured content from news articles and blog posts.

Prerequisites:
    - pip install langchain-ujeebu langchain-openai
    - Set UJEEBU_API_KEY environment variable
    - Set OPENAI_API_KEY environment variable (for LLM examples)

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
from langchain_ujeebu import UjeebuExtractTool

# Ensure API key is set
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    print("Get your API key at: https://ujeebu.com/signup")
    exit(1)


def basic_extraction():
    """Basic article extraction example."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Article Extraction")
    print("=" * 60)

    # Initialize the extract tool
    extract_tool = UjeebuExtractTool()

    # Extract article content
    url = "https://www.bbc.com/news/technology-67881954"
    result = extract_tool._run(url=url)

    print(f"\nExtracted content from: {url}")
    print("-" * 40)
    print(result[:2000] + "..." if len(result) > 2000 else result)


def extraction_with_options():
    """Article extraction with custom options."""
    print("\n" + "=" * 60)
    print("Example 2: Extraction with Custom Options")
    print("=" * 60)

    extract_tool = UjeebuExtractTool()

    # Extract with images and quick mode
    url = "https://techcrunch.com/2024/01/15/openai-gpt-store/"
    result = extract_tool._run(
        url=url,
        text=True,
        images=True,
        quick_mode=True,  # 30-60% faster, slightly less accurate
    )

    print(f"\nExtracted with images from: {url}")
    print("-" * 40)
    print(result[:2000] + "..." if len(result) > 2000 else result)


def extraction_with_html():
    """Extract article with HTML content preserved."""
    print("\n" + "=" * 60)
    print("Example 3: Extraction with HTML Content")
    print("=" * 60)

    extract_tool = UjeebuExtractTool()

    url = "https://www.theverge.com/2024/1/16/24040562/google-layoffs-2024"
    result = extract_tool._run(
        url=url,
        text=False,  # Skip plain text
        html=True,  # Get HTML content
    )

    print(f"\nExtracted HTML from: {url}")
    print("-" * 40)
    print(result[:1500] + "..." if len(result) > 1500 else result)


def use_with_langchain_agent():
    """Use extract tool with a LangChain agent."""
    print("\n" + "=" * 60)
    print("Example 4: Using with LangChain Agent")
    print("=" * 60)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping agent example - OPENAI_API_KEY not set")
        return

    from langchain_openai import ChatOpenAI

    # Initialize LLM and tool
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    extract_tool = UjeebuExtractTool()

    # Create agent using LangGraph (recommended) or fallback to legacy
    try:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(llm, [extract_tool])

        # LangGraph invocation
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        "Extract the article from https://www.bbc.com/news/technology-67881954 and give me a 3-sentence summary.",
                    )
                ]
            }
        )

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
                ("system", "You are a helpful assistant."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, [extract_tool], prompt)
        executor = AgentExecutor(agent=agent, tools=[extract_tool], verbose=True)
        result = executor.invoke(
            {
                "input": "Extract the article from https://www.bbc.com/news/technology-67881954 and give me a 3-sentence summary."
            }
        )
        response = result.get("output", str(result))

    print("\nAgent Response:")
    print("-" * 40)
    print(response)


def batch_extraction():
    """Extract multiple articles in sequence."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Article Extraction")
    print("=" * 60)

    extract_tool = UjeebuExtractTool()

    urls = [
        "https://www.bbc.com/news/technology-67881954",
        "https://www.theverge.com/2024/1/16/24040562/google-layoffs-2024",
        "https://techcrunch.com/2024/01/15/openai-gpt-store/",
    ]

    for i, url in enumerate(urls, 1):
        print(f"\n--- Article {i}/{len(urls)} ---")
        result = extract_tool._run(url=url, quick_mode=True)

        # Just print the title and summary
        lines = result.split("\n")
        for line in lines[:5]:
            if line.strip():
                print(line)


if __name__ == "__main__":
    print("Ujeebu Extract Tool Examples")
    print("============================")

    # Run examples
    basic_extraction()
    extraction_with_options()
    extraction_with_html()
    batch_extraction()

    # Optional: Agent example (requires OpenAI API key)
    use_with_langchain_agent()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
