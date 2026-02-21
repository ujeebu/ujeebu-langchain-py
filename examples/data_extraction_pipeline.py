#!/usr/bin/env python3
"""
Example: Data Extraction Pipelines with Ujeebu

This example demonstrates how to build data extraction pipelines for:
- E-commerce product scraping
- News aggregation
- Social media monitoring
- Price comparison

Prerequisites:
    - pip install langchain-ujeebu pandas
    - Set UJEEBU_API_KEY environment variable

Get your Ujeebu API key at: https://ujeebu.com/signup
"""

import os
import json
from langchain_ujeebu import UjeebuScrapeTool, UjeebuLoader

# Ensure API key is set
if not os.getenv("UJEEBU_API_KEY"):
    print("Please set UJEEBU_API_KEY environment variable")
    print("Get your API key at: https://ujeebu.com/signup")
    exit(1)


def scrape_books_catalog():
    """Scrape book catalog with structured data extraction."""
    print("\n" + "=" * 60)
    print("Pipeline 1: Book Catalog Scraping")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    # Define extraction rules for book catalog
    extraction_rules = {
        "books": {
            "selector": "article.product_pod",
            "type": "obj",
            "multiple": True,
            "children": {
                "title": {"selector": "h3 a", "type": "attr", "attribute": "title"},
                "price": {"selector": ".price_color", "type": "text"},
                "availability": {"selector": ".availability", "type": "text"},
                "rating": {
                    "selector": ".star-rating",
                    "type": "attr",
                    "attribute": "class",
                },
                "link": {"selector": "h3 a", "type": "link"},
            },
        }
    }

    # Scrape multiple pages
    all_books = []
    for page in range(1, 4):  # Scrape first 3 pages
        url = f"https://books.toscrape.com/catalogue/page-{page}.html"
        print(f"\n   Scraping page {page}...")

        result = scrape_tool._run(
            url=url,
            js_rendering=False,  # Static site, no JS needed
            extract_rules=json.dumps(extraction_rules),
        )

        try:
            data = json.loads(result)
            books = data.get("books", [])
            all_books.extend(books)
            print(f"   Found {len(books)} books on page {page}")
        except json.JSONDecodeError:
            print(f"   Error parsing page {page}")

    print(f"\n📚 Total books scraped: {len(all_books)}")

    # Display sample
    print("\n   Sample books:")
    for book in all_books[:5]:
        print(f"   - {book.get('title', 'N/A')[:40]}... | {book.get('price', 'N/A')}")

    return all_books


def scrape_quotes_with_pagination():
    """Scrape quotes with author information."""
    print("\n" + "=" * 60)
    print("Pipeline 2: Quotes Scraping with Authors")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    extraction_rules = {
        "quotes": {
            "selector": ".quote",
            "type": "obj",
            "multiple": True,
            "children": {
                "text": {"selector": ".text", "type": "text"},
                "author": {"selector": ".author", "type": "text"},
                "tags": {"selector": ".tag", "type": "text", "multiple": True},
            },
        },
        "next_page": {"selector": ".next a", "type": "link"},
    }

    all_quotes = []
    url = "https://quotes.toscrape.com/"
    pages_scraped = 0
    max_pages = 3

    while url and pages_scraped < max_pages:
        print(f"\n   Scraping: {url}")

        result = scrape_tool._run(
            url=url, js_rendering=False, extract_rules=json.dumps(extraction_rules)
        )

        try:
            data = json.loads(result)
            quotes = data.get("quotes", [])
            all_quotes.extend(quotes)
            print(f"   Found {len(quotes)} quotes")

            # Get next page URL
            next_page = data.get("next_page")
            if next_page:
                if next_page.startswith("http"):
                    url = next_page
                else:
                    url = f"https://quotes.toscrape.com{next_page}"
            else:
                url = None

            pages_scraped += 1
        except json.JSONDecodeError:
            print("   Error parsing page")
            break

    print(f"\n📜 Total quotes scraped: {len(all_quotes)}")

    # Display sample with tags
    print("\n   Sample quotes:")
    for quote in all_quotes[:3]:
        text = quote.get("text", "")[:60]
        author = quote.get("author", "Unknown")
        tags = ", ".join(quote.get("tags", [])[:3])
        print(f'   "{text}..." - {author}')
        print(f"      Tags: {tags}")

    return all_quotes


def news_aggregation_pipeline():
    """Aggregate news from multiple sources."""
    print("\n" + "=" * 60)
    print("Pipeline 3: News Aggregation")
    print("=" * 60)

    # Use UjeebuLoader for article extraction
    news_sources = [
        # BBC Technology
        "https://www.bbc.com/news/technology",
    ]

    scrape_tool = UjeebuScrapeTool()

    # First, get article links from the category page
    extraction_rules = {
        "articles": {
            "selector": "a[href*='/news/articles/']",
            "type": "obj",
            "multiple": True,
            "children": {
                "title": {"selector": "&", "type": "text"},
                "url": {"selector": "&", "type": "link"},
            },
        }
    }

    print("\n   Fetching article links from BBC Technology...")
    result = scrape_tool._run(
        url="https://www.bbc.com/news/technology",
        js_rendering=True,
        extract_rules=json.dumps(extraction_rules),
    )

    try:
        data = json.loads(result)
        articles = data.get("articles", [])

        # Filter and clean article links
        article_urls = []
        seen = set()
        for article in articles:
            url = article.get("url", "")
            title = article.get("title", "").strip()
            if url and title and url not in seen and len(title) > 10:
                if not url.startswith("http"):
                    url = f"https://www.bbc.com{url}"
                article_urls.append({"title": title, "url": url})
                seen.add(url)

        print(f"   Found {len(article_urls)} unique articles")

        # Display found articles
        print("\n   Articles found:")
        for i, article in enumerate(article_urls[:5], 1):
            print(f"   {i}. {article['title'][:50]}...")

    except json.JSONDecodeError:
        print("   Error parsing category page")

    return article_urls[:5] if "article_urls" in dir() else []


def price_monitoring_pipeline():
    """Monitor prices across product pages."""
    print("\n" + "=" * 60)
    print("Pipeline 4: Price Monitoring")
    print("=" * 60)

    scrape_tool = UjeebuScrapeTool()

    # Products to monitor (using books.toscrape.com as example)
    products = [
        {
            "name": "A Light in the Attic",
            "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        },
        {
            "name": "Tipping the Velvet",
            "url": "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        },
        {
            "name": "Soumission",
            "url": "https://books.toscrape.com/catalogue/soumission_998/index.html",
        },
    ]

    extraction_rules = {
        "title": {"selector": "h1", "type": "text"},
        "price": {"selector": ".price_color", "type": "text"},
        "availability": {"selector": ".availability", "type": "text"},
        "description": {"selector": "#product_description ~ p", "type": "text"},
    }

    price_data = []

    for product in products:
        print(f"\n   Checking: {product['name']}...")

        result = scrape_tool._run(
            url=product["url"],
            js_rendering=False,
            extract_rules=json.dumps(extraction_rules),
        )

        try:
            data = json.loads(result)
            data["url"] = product["url"]
            data["monitored_name"] = product["name"]
            price_data.append(data)

            price = data.get("price", "N/A")
            availability = data.get("availability", "Unknown").strip()
            print(f"   Price: {price} | {availability}")

        except json.JSONDecodeError:
            print(f"   Error getting price for {product['name']}")

    print(f"\n💰 Price monitoring complete for {len(price_data)} products")
    return price_data


def export_to_dataframe(data, name):
    """Export scraped data to pandas DataFrame."""
    try:
        import pandas as pd

        df = pd.DataFrame(data)
        print(f"\n   DataFrame created with {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")

        # Display first few rows
        print(f"\n   Preview:")
        print(df.head().to_string())

        # Optionally save to CSV
        filename = f"{name}.csv"
        df.to_csv(filename, index=False)
        print(f"\n   Saved to {filename}")

        return df
    except ImportError:
        print("   pandas not installed - skipping DataFrame export")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Data Extraction Pipeline Examples")
    print("=" * 60)

    # Run pipelines
    books = scrape_books_catalog()
    quotes = scrape_quotes_with_pagination()
    news = news_aggregation_pipeline()
    prices = price_monitoring_pipeline()

    # Export to DataFrames (optional)
    print("\n" + "=" * 60)
    print("Exporting Data")
    print("=" * 60)

    if books:
        export_to_dataframe(books, "books_catalog")

    if quotes:
        export_to_dataframe(quotes, "quotes_collection")

    print("\n" + "=" * 60)
    print("Pipeline Examples completed!")
    print("=" * 60)
