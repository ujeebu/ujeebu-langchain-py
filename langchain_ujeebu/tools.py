"""Ujeebu tools for LangChain agents."""

import json
import logging
import os
from typing import Optional, Type, Any, Dict

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from ujeebu_python import UjeebuClient
import requests

logger = logging.getLogger(__name__)


class UjeebuExtractInput(BaseModel):
    """Input schema for UjeebuExtractTool."""

    url: str = Field(..., description="URL of the article to extract content from")
    text: bool = Field(default=True, description="Extract article text content")
    html: bool = Field(default=False, description="Extract article HTML content")
    author: bool = Field(default=True, description="Extract article author")
    pub_date: bool = Field(default=True, description="Extract article publication date")
    images: bool = Field(default=False, description="Extract article images")
    quick_mode: bool = Field(
        default=False,
        description=(
            "Use quick mode for faster extraction"
            " (30-60% faster, slightly less accurate)"
        ),
    )


class UjeebuExtractTool(BaseTool):
    """Extract clean, structured content from articles.

    Uses the Ujeebu Extract API.

    This tool extracts article content including:
    - Main text content
    - HTML content
    - Author information
    - Publication date
    - Title and summary
    - Images and media

    To use this tool, you must set the UJEEBU_API_KEY environment variable.
    Get your API key at: https://ujeebu.com/signup
    """

    name: str = "ujeebu_extract"
    description: str = (
        "Extract clean, structured content from news articles and blog posts. "
        "Useful for retrieving article text, metadata, author, publication date, "
        "and other structured information from web pages. "
        "Input should be a valid article URL."
    )
    args_schema: Type[BaseModel] = UjeebuExtractInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120
    _client: UjeebuClient = None  # type: ignore[assignment]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        **kwargs: Any,
    ):
        """Initialize the Ujeebu Extract tool.

        Args:
            api_key: Ujeebu API key. If not provided,
                will use UJEEBU_API_KEY env variable.
            base_url: Base URL for the Ujeebu API
                (optional, defaults to https://api.ujeebu.com).
            timeout: Request timeout in seconds (default: 120).
            **kwargs: Additional arguments to pass to BaseTool.
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("UJEEBU_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "Ujeebu API key must be provided either through "
                "api_key parameter or UJEEBU_API_KEY environment "
                "variable. Get your API key at "
                "https://ujeebu.com/signup"
            )
        # Initialize the Ujeebu SDK client
        self._client = UjeebuClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _run(
        self,
        url: str,
        text: bool = True,
        html: bool = False,
        author: bool = True,
        pub_date: bool = True,
        images: bool = False,
        quick_mode: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Extract article content from the given URL.

        Args:
            url: URL of the article to extract
            text: Whether to extract article text
            html: Whether to extract article HTML
            author: Whether to extract article author
            pub_date: Whether to extract publication date
            images: Whether to extract images
            quick_mode: Whether to use quick mode for faster extraction
            run_manager: Callback manager for tool run

        Returns:
            Formatted string containing the extracted article content
        """
        params: Dict[str, Any] = {
            "text": int(text),
            "html": int(html),
            "author": int(author),
            "pub_date": int(pub_date),
            "images": int(images),
            "quick_mode": int(quick_mode),
        }

        try:
            response = self._client.extract(url, params)
            response.raise_for_status()

            data = response.json()
            article = data.get("article", {})

            # Format the response in a readable way
            result_parts = []

            if article.get("title"):
                result_parts.append(f"Title: {article['title']}")

            if article.get("author"):
                result_parts.append(f"Author: {article['author']}")

            if article.get("pub_date"):
                result_parts.append(f"Published: {article['pub_date']}")

            if article.get("site_name"):
                result_parts.append(f"Site: {article['site_name']}")

            if article.get("summary"):
                result_parts.append(f"\nSummary: {article['summary']}")

            if text and article.get("text"):
                result_parts.append(f"\nContent:\n{article['text']}")

            if html and article.get("html"):
                result_parts.append(f"\nHTML:\n{article['html']}")

            if images and article.get("images"):
                result_parts.append(f"\nImages: {', '.join(article['images'][:5])}")

            return "\n".join(result_parts)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Invalid API key for Ujeebu Extract")
                return "Error: Invalid API key. Please check your UJEEBU_API_KEY."
            elif e.response.status_code == 404:
                logger.warning(f"Article not found at URL: {url}")
                return f"Error: Article not found at URL: {url}"
            elif e.response.status_code == 408:
                logger.warning(f"Request timeout for URL: {url}")
                return "Error: Request timeout. Try increasing the timeout."
            else:
                logger.error(f"Error extracting article from {url}: {str(e)}")
                return f"Error extracting article: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error extracting article from {url}: {str(e)}")
            return f"Error: {str(e)}"

    async def _arun(
        self,
        url: str,
        text: bool = True,
        html: bool = False,
        author: bool = True,
        pub_date: bool = True,
        images: bool = False,
        quick_mode: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of _run method."""
        # For now, we'll use the sync version
        # In production, you'd want to use an async HTTP client like aiohttp
        return self._run(
            url=url,
            text=text,
            html=html,
            author=author,
            pub_date=pub_date,
            images=images,
            quick_mode=quick_mode,
            run_manager=run_manager,
        )


class UjeebuScrapeInput(BaseModel):
    """Input schema for UjeebuScrapeTool."""

    url: str = Field(..., description="URL of the web page to scrape")
    js_rendering: bool = Field(
        default=True, description="Enable JavaScript rendering for dynamic content"
    )
    premium_proxy: bool = Field(
        default=False,
        description="Use premium residential proxies for better success rate",
    )
    wait_for: Optional[str] = Field(
        default=None, description="CSS selector to wait for before scraping"
    )
    response_type: str = Field(
        default="html",
        description="Response type: 'html', 'text', 'screenshot', or 'pdf'",
    )
    extract_rules: Optional[str] = Field(
        default=None,
        description=(
            'JSON string of extraction rules. Format: {"field": {"selector": ".css", '
            '"type": "text|obj|link|image|attr", "multiple": true, "children": {...}}}'
        ),
    )


class UjeebuScrapeTool(BaseTool):
    """Tool for scraping web pages using Ujeebu Scrape API.

    This tool can:
    - Render JavaScript-heavy pages
    - Extract HTML or text content
    - Take screenshots or generate PDFs
    - Extract structured data using CSS/XPath selectors
    - Handle anti-bot protection with premium proxies

    To use this tool, you must set the UJEEBU_API_KEY environment variable.
    Get your API key at: https://ujeebu.com/signup
    """

    name: str = "ujeebu_scrape"
    description: str = (
        "Scrape web pages with JavaScript rendering support. "
        "Can extract HTML content, take screenshots, generate PDFs, "
        "or extract structured data using CSS selectors. "
        "Useful for scraping dynamic websites, SPAs, "
        "and pages with anti-bot protection. "
        "Input should be a valid URL."
    )
    args_schema: Type[BaseModel] = UjeebuScrapeInput
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120
    _client: UjeebuClient = None  # type: ignore[assignment]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        **kwargs: Any,
    ):
        """Initialize the Ujeebu Scrape tool.

        Args:
            api_key: Ujeebu API key. If not provided,
                will use UJEEBU_API_KEY env variable.
            base_url: Base URL for the Ujeebu API
                (optional, defaults to https://api.ujeebu.com).
            timeout: Request timeout in seconds (default: 120).
            **kwargs: Additional arguments to pass to BaseTool.
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("UJEEBU_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "Ujeebu API key must be provided either through "
                "api_key parameter or UJEEBU_API_KEY environment "
                "variable. Get your API key at "
                "https://ujeebu.com/signup"
            )
        # Initialize the Ujeebu SDK client
        self._client = UjeebuClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _run(
        self,
        url: str,
        js_rendering: bool = True,
        premium_proxy: bool = False,
        wait_for: Optional[str] = None,
        response_type: str = "html",
        extract_rules: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Scrape content from the given URL.

        Args:
            url: URL of the web page to scrape
            js_rendering: Whether to enable JavaScript rendering
            premium_proxy: Whether to use premium proxies
            wait_for: CSS selector to wait for
            response_type: Type of response (html, text, screenshot, pdf)
            extract_rules: JSON string of extraction rules
            run_manager: Callback manager for tool run

        Returns:
            Scraped content or extracted data as a string
        """
        params: Dict[str, Any] = {
            "js": int(js_rendering),
            "json": 1,
        }

        if premium_proxy:
            params["proxy_type"] = "premium"

        if wait_for:
            params["wait_for"] = wait_for

        if response_type in ("screenshot", "pdf"):
            params["response_type"] = response_type

        try:
            # Handle extraction rules if provided
            if extract_rules:
                try:
                    rules = json.loads(extract_rules)
                    response = self._client.scrape_with_rules(url, rules, params)
                except json.JSONDecodeError:
                    return "Error: extract_rules must be a valid JSON string"
            else:
                response = self._client.scrape(url, params)

            response.raise_for_status()
            data = response.json()

            # Format the response based on type
            if extract_rules:
                # Return extracted data as formatted JSON
                extracted = data.get("result", {})
                return json.dumps(extracted, indent=2)
            elif response_type == "screenshot":
                screenshot_url = data.get("screenshot")
                if screenshot_url:
                    return f"Screenshot saved: {screenshot_url}"
                return "Screenshot captured (base64 in response)"
            elif response_type == "pdf":
                pdf_url = data.get("pdf")
                if pdf_url:
                    return f"PDF generated: {pdf_url}"
                return "PDF generated (base64 in response)"
            else:
                # Return HTML or text content
                raw_html: str = data.get("html", "")
                if response_type == "text":
                    # Strip HTML tags for text-only response
                    import re

                    clean = re.sub(r"<[^>]+>", "", raw_html)
                    return clean[:10000] if len(clean) > 10000 else clean
                return raw_html[:10000] if len(raw_html) > 10000 else raw_html

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Invalid API key for Ujeebu Scrape")
                return "Error: Invalid API key. Please check your UJEEBU_API_KEY."
            elif e.response.status_code == 404:
                logger.warning(f"Page not found at URL: {url}")
                return f"Error: Page not found at URL: {url}"
            elif e.response.status_code == 408:
                logger.warning(f"Request timeout for URL: {url}")
                return "Error: Request timeout. Try increasing the timeout."
            else:
                logger.error(f"Error scraping page {url}: {str(e)}")
                return f"Error scraping page: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error scraping page {url}: {str(e)}")
            return f"Error: {str(e)}"

    async def _arun(
        self,
        url: str,
        js_rendering: bool = True,
        premium_proxy: bool = False,
        wait_for: Optional[str] = None,
        response_type: str = "html",
        extract_rules: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of _run method."""
        return self._run(
            url=url,
            js_rendering=js_rendering,
            premium_proxy=premium_proxy,
            wait_for=wait_for,
            response_type=response_type,
            extract_rules=extract_rules,
            run_manager=run_manager,
        )
