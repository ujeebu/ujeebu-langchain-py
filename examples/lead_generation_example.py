#!/usr/bin/env python3
"""
Example: Lead Generation with Ujeebu + OpenAI

This example demonstrates how to use UjeebuScrapeTool to scrape web pages
and then use ChatOpenAI to extract contact information (emails, phones,
names, companies) from the scraped content.

Prerequisites:
    - pip install langchain-ujeebu langchain-openai pandas
    - Set UJEEBU_API_KEY environment variable
    - Set OPENAI_API_KEY environment variable

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
import re
import json
from langchain_ujeebu import UjeebuScrapeTool
from langchain_openai import ChatOpenAI

# Ensure API keys are set
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    print("Get your API key at: https://ujeebu.com/signup")
    exit(1)

if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)


def parse_json_response(text):
    """Parse JSON from an LLM response, stripping markdown code fences if present."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` wrappers
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


CONTACT_EXTRACTION_PROMPT = """\
You are a contact information extraction assistant. Analyze the following \
web page content and extract any contact information you can find.

Return a JSON array of contacts. Each contact should have these fields \
(use null for missing values):
- "name": person's full name
- "email": email address
- "phone": phone number
- "role": job title or role
- "company": company or organization name

If no contacts are found, return an empty array: []

IMPORTANT: Return ONLY valid JSON, no other text.

Web page content:
{content}"""


LEAD_SCORING_PROMPT = """\
You are a lead qualification assistant. Based on the following web page \
content and extracted contacts, score each lead from 1-10 and provide a \
brief reason.

Scoring criteria:
- Company size indicators (more employees = higher score)
- Industry relevance (tech, SaaS, e-commerce score higher)
- Contact completeness (email + phone + name = higher score)
- Decision-maker role (CEO, CTO, VP = higher score)

Return a JSON array where each item has:
- "name": contact name
- "score": integer 1-10
- "reason": brief explanation of the score

IMPORTANT: Return ONLY valid JSON, no other text.

Page content:
{content}

Extracted contacts:
{contacts}"""


def scrape_and_extract_contacts():
    """Scrape a single page and extract contact information using OpenAI."""
    print("\n" + "=" * 60)
    print("Example 1: Single Page Contact Extraction")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    url = "https://www.eff.org/about/contact"
    print(f"\n   Scraping: {url}")

    page_content = scrape_tool._run(
        url=url,
        js_rendering=True,
        response_type="text",
    )

    if page_content.startswith("Error"):
        print(f"   Scraping failed: {page_content}")
        return []

    print(f"   Scraped {len(page_content)} characters")

    # Use OpenAI to extract contacts
    print("   Extracting contacts with OpenAI...")
    prompt = CONTACT_EXTRACTION_PROMPT.format(content=page_content[:8000])
    response = llm.invoke(prompt)

    try:
        contacts = parse_json_response(response.content)
        print(f"   Found {len(contacts)} contacts")

        for contact in contacts:
            print(f"\n   Contact:")
            print(f"      Name:    {contact.get('name') or 'N/A'}")
            print(f"      Email:   {contact.get('email') or 'N/A'}")
            print(f"      Phone:   {contact.get('phone') or 'N/A'}")
            print(f"      Role:    {contact.get('role') or 'N/A'}")
            print(f"      Company: {contact.get('company') or 'N/A'}")

        if not contacts:
            print("   No contacts found on this page")

        return contacts

    except json.JSONDecodeError:
        print(f"   Could not parse LLM response: {response.content[:200]}")
        return []


def batch_lead_generation():
    """Scrape multiple pages and extract contacts from each."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Lead Generation")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # Target URLs (public pages with contact information)
    urls = [
        "https://www.eff.org/about/contact",
        "https://ahrefs.com/about",
        "https://basecamp.com/about",
    ]

    all_contacts = []

    for url in urls:
        print(f"\n   Scraping: {url}")

        page_content = scrape_tool._run(
            url=url,
            js_rendering=True,
            response_type="text",
        )

        if page_content.startswith("Error"):
            print(f"   Scraping failed: {page_content}")
            continue

        print(f"   Scraped {len(page_content)} characters")
        print("   Extracting contacts...")

        prompt = CONTACT_EXTRACTION_PROMPT.format(content=page_content[:8000])
        response = llm.invoke(prompt)

        try:
            contacts = parse_json_response(response.content)
            # Tag each contact with its source URL
            for contact in contacts:
                contact["source_url"] = url
            all_contacts.extend(contacts)
            print(f"   Found {len(contacts)} contacts")
        except json.JSONDecodeError:
            print(f"   Could not parse LLM response")

    print(f"\n   Total contacts found: {len(all_contacts)}")

    # Export to CSV via pandas
    if all_contacts:
        try:
            import pandas as pd

            df = pd.DataFrame(all_contacts)
            filename = "leads_export.csv"
            df.to_csv(filename, index=False)
            print(f"\n   Exported {len(df)} leads to {filename}")
            print(f"   Columns: {list(df.columns)}")
            print(f"\n   Preview:")
            print(df.head().to_string())
        except ImportError:
            print("   pandas not installed - skipping CSV export")
            print("   Install with: pip install pandas")

    return all_contacts


def lead_generation_with_scoring():
    """Scrape pages, extract contacts, and score/qualify leads."""
    print("\n" + "=" * 60)
    print("Example 3: Lead Generation with Scoring")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    url = "https://ahrefs.com/about"
    print(f"\n   Scraping: {url}")

    page_content = scrape_tool._run(
        url=url,
        js_rendering=True,
        response_type="text",
    )

    if page_content.startswith("Error"):
        print(f"   Scraping failed: {page_content}")
        return []

    print(f"   Scraped {len(page_content)} characters")

    # Step 1: Extract contacts
    print("   Step 1: Extracting contacts...")
    extract_prompt = CONTACT_EXTRACTION_PROMPT.format(content=page_content[:8000])
    extract_response = llm.invoke(extract_prompt)

    try:
        contacts = parse_json_response(extract_response.content)
    except json.JSONDecodeError:
        print(f"   Could not parse extraction response")
        contacts = []

    print(f"   Found {len(contacts)} contacts")

    if not contacts:
        print("   No contacts found to score")
        return []

    # Step 2: Score leads
    print("   Step 2: Scoring leads...")
    score_prompt = LEAD_SCORING_PROMPT.format(
        content=page_content[:6000],
        contacts=json.dumps(contacts, indent=2),
    )
    score_response = llm.invoke(score_prompt)

    try:
        scored_leads = parse_json_response(score_response.content)
        print(f"   Scored {len(scored_leads)} leads")

        for lead in scored_leads:
            score = lead.get("score", 0)
            bar = "#" * score + "-" * (10 - score)
            print(f"\n   Lead: {lead.get('name', 'N/A')}")
            print(f"      Score:  [{bar}] {score}/10")
            print(f"      Reason: {lead.get('reason', 'N/A')}")

        return scored_leads

    except json.JSONDecodeError:
        print(f"   Could not parse scoring response")
        return []


if __name__ == "__main__":
    print("=" * 60)
    print("Lead Generation Examples")
    print("=" * 60)

    contacts = scrape_and_extract_contacts()
    leads = batch_lead_generation()
    scored = lead_generation_with_scoring()

    print("\n" + "=" * 60)
    print("Lead Generation Examples completed!")
    print("=" * 60)
