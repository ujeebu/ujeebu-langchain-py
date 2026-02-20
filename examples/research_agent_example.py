#!/usr/bin/env python3
"""
Example: Research Agent with Ujeebu Tools

This example demonstrates how to build a research agent that can:
- Extract article content for analysis
- Scrape websites for data
- Answer questions based on web research

Prerequisites:
    - pip install langchain-ujeebu langchain-openai langgraph
    - Set UJEEBU_API_KEY environment variable
    - Set OPENAI_API_KEY environment variable

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
import json
from langchain_ujeebu import UjeebuExtractTool, UjeebuScrapeTool
from langchain_openai import ChatOpenAI

# Check for required API keys
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    exit(1)

if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)


def create_agent(tools, llm):
    """Create a LangGraph agent with the given tools."""
    try:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(llm, tools), "langgraph"
    except ImportError:
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
        return AgentExecutor(agent=agent, tools=tools, verbose=True), "legacy"


def invoke_agent(agent, query, agent_type="langgraph"):
    """Invoke agent with proper input format."""
    if agent_type == "langgraph":
        result = agent.invoke({"messages": [("human", query)]})
        if isinstance(result, dict) and "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content:
                    return msg.content
        return str(result)
    else:
        result = agent.invoke({"input": query})
        return result.get("output", str(result))


def research_task_single_article():
    """Research a single article and answer questions."""
    print("\n" + "=" * 60)
    print("Research Task: Single Article Analysis")
    print("=" * 60)

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    extract_tool = UjeebuExtractTool()

    agent, agent_type = create_agent([extract_tool], llm)

    query = """
    Please extract the article from https://www.bbc.com/news/technology-67881954
    and provide:
    1. A brief summary (2-3 sentences)
    2. The main topic
    3. Key points mentioned
    """

    response = invoke_agent(agent, query, agent_type)

    print("\n📝 Research Results:")
    print("-" * 40)
    print(response)


def research_task_compare_sources():
    """Compare information from multiple sources."""
    print("\n" + "=" * 60)
    print("Research Task: Compare Multiple Sources")
    print("=" * 60)

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    extract_tool = UjeebuExtractTool()
    scrape_tool = UjeebuScrapeTool()

    agent, agent_type = create_agent([extract_tool, scrape_tool], llm)

    query = """
    I need to research the current state of AI development.

    1. First, check what's trending on Hacker News (news.ycombinator.com)
    2. Then extract an AI-related article if you find one
    3. Summarize the key trends you observe
    """

    response = invoke_agent(agent, query, agent_type)

    print("\n📝 Research Results:")
    print("-" * 40)
    print(response)


def research_task_fact_checking():
    """Use the agent for fact-checking."""
    print("\n" + "=" * 60)
    print("Research Task: Fact Checking")
    print("=" * 60)

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    scrape_tool = UjeebuScrapeTool()

    agent, agent_type = create_agent([scrape_tool], llm)

    query = """
    I heard that there have been major tech layoffs recently.
    Can you verify this by:
    1. Scraping a tech news homepage
    2. Finding any articles about layoffs
    3. Summarizing what you find
    """

    response = invoke_agent(agent, query, agent_type)

    print("\n📝 Fact Check Results:")
    print("-" * 40)
    print(response)


def research_with_structured_output():
    """Research with structured JSON output."""
    print("\n" + "=" * 60)
    print("Research Task: Structured Output")
    print("=" * 60)

    # Use extraction rules for structured data
    scrape_tool = UjeebuScrapeTool()

    extraction_rules = json.dumps(
        {
            "titles": {
                "selector": ".titleline > a",
                "type": "obj",
                "multiple": True,
                "children": {
                    "text": {"selector": "&", "type": "text"},
                    "url": {"selector": "&", "type": "link"},
                },
            },
            "scores": {"selector": ".score", "type": "text", "multiple": True},
        }
    )

    # Scrape with extraction rules
    result = scrape_tool._run(
        url="https://news.ycombinator.com", extract_rules=extraction_rules
    )

    print("\n📊 Structured Data from Hacker News:")
    print("-" * 40)

    try:
        data = json.loads(result)
        titles = data.get("titles", [])[:5]
        scores = data.get("scores", [])[:5]

        for i, (title, score) in enumerate(zip(titles, scores), 1):
            if isinstance(title, dict):
                print(f"{i}. {title.get('text', 'N/A')} ({score})")
            else:
                print(f"{i}. {title} ({score})")
    except json.JSONDecodeError:
        print(result)


def direct_research_without_agent():
    """Demonstrate direct tool usage for research without an agent."""
    print("\n" + "=" * 60)
    print("Direct Research: No Agent Required")
    print("=" * 60)

    extract_tool = UjeebuExtractTool()
    scrape_tool = UjeebuScrapeTool()

    # Step 1: Scrape headlines
    print("\n📰 Step 1: Scraping Hacker News headlines...")
    extraction_rules = json.dumps(
        {"stories": {"selector": ".titleline > a", "type": "text", "multiple": True}}
    )

    headlines = scrape_tool._run(
        url="https://news.ycombinator.com", extract_rules=extraction_rules
    )

    try:
        data = json.loads(headlines)
        stories = data.get("stories", [])[:5]
        print("Top 5 stories:")
        for i, story in enumerate(stories, 1):
            print(f"  {i}. {story}")
    except:
        print(headlines[:500])

    # Step 2: Extract a specific article
    print("\n📄 Step 2: Extracting article content...")
    article = extract_tool._run(url="https://quotes.toscrape.com/")
    print(f"Extracted content preview:\n{article[:400]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("Research Agent Examples with Ujeebu + LangChain")
    print("=" * 60)

    # Direct research (always works, no agent needed)
    direct_research_without_agent()
    research_with_structured_output()

    # Agent-based research
    research_task_single_article()

    # More complex research (uncomment to run)
    # research_task_compare_sources()
    # research_task_fact_checking()

    print("\n" + "=" * 60)
    print("Research Agent Examples completed!")
    print("=" * 60)
