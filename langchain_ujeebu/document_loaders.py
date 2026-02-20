"""Ujeebu document loaders for LangChain."""

import logging
import os
from typing import Iterator, List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from ujeebu_python import UjeebuClient

logger = logging.getLogger(__name__)


class UjeebuLoader(BaseLoader):
    """Load articles using Ujeebu Extract API.

    This loader fetches article content and creates LangChain Document objects
    that can be used with vector stores, retrievers, and other LangChain components.

    Examples:
        Basic usage:

        ```python
        from langchain_ujeebu import UjeebuLoader

        # Load a single article
        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key="your-api-key"
        )
        documents = loader.load()
        ```

        Load multiple articles:

        ```python
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ]
        loader = UjeebuLoader(urls=urls)
        documents = loader.load()
        ```

        Use with vector store:

        ```python
        from langchain_ujeebu import UjeebuLoader
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings

        loader = UjeebuLoader(urls=["https://example.com/article"])
        documents = loader.load()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        ```
    """

    def __init__(
        self,
        urls: List[str],
        api_key: Optional[str] = None,
        extract_text: bool = True,
        extract_html: bool = False,
        extract_author: bool = True,
        extract_pub_date: bool = True,
        extract_images: bool = False,
        quick_mode: bool = False,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        """Initialize the Ujeebu document loader.

        Args:
            urls: List of article URLs to load
            api_key: Ujeebu API key. If not provided,
                will use UJEEBU_API_KEY env variable
            extract_text: Whether to extract article text content
            extract_html: Whether to extract article HTML content
            extract_author: Whether to extract article author
            extract_pub_date: Whether to extract publication date
            extract_images: Whether to extract images
            quick_mode: Whether to use quick mode
                (30-60% faster, slightly less accurate)
            base_url: Base URL for the Ujeebu API (optional,
                defaults to https://api.ujeebu.com)
            timeout: Request timeout in seconds (default: 120)
        """
        self.urls = urls
        self.api_key = api_key or os.getenv("UJEEBU_API_KEY")
        self.extract_text = extract_text
        self.extract_html = extract_html
        self.extract_author = extract_author
        self.extract_pub_date = extract_pub_date
        self.extract_images = extract_images
        self.quick_mode = quick_mode

        if not self.api_key:
            raise ValueError(
                "Ujeebu API key must be provided either through api_key parameter "
                "or UJEEBU_API_KEY environment variable. "
                "Get your API key at https://ujeebu.com/signup"
            )

        # Initialize the Ujeebu SDK client
        self._client = UjeebuClient(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _extract_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract a single article from the given URL.

        Args:
            url: URL of the article to extract

        Returns:
            Dictionary containing article data, or None if extraction failed
        """
        params: Dict[str, Any] = {
            "text": self.extract_text,
            "html": self.extract_html,
            "author": self.extract_author,
            "pub_date": self.extract_pub_date,
            "images": self.extract_images,
            "quick_mode": self.quick_mode,
        }

        try:
            response = self._client.extract(url, params)
            response.raise_for_status()

            data: Dict[str, Any] = response.json()
            article: Optional[Dict[str, Any]] = data.get("article")
            return article

        except Exception as e:
            logger.error(f"Error extracting article from {url}: {str(e)}")
            return None

    def _create_document(self, article: Dict[str, Any], url: str) -> Document:
        """Create a Document from an extracted article.

        Args:
            article: Dictionary containing article data
            url: Original URL of the article

        Returns:
            Document object with content and metadata
        """
        # Prepare the content
        content_parts = []

        if self.extract_text and article.get("text"):
            content_parts.append(article["text"])

        if self.extract_html and article.get("html"):
            content_parts.append(f"HTML: {article['html']}")

        content = "\n\n".join(content_parts)

        # Prepare metadata
        metadata = {
            "source": url,
            "url": article.get("url", url),
            "canonical_url": article.get("canonical_url", url),
        }

        if article.get("title"):
            metadata["title"] = article["title"]

        if self.extract_author and article.get("author"):
            metadata["author"] = article["author"]

        if self.extract_pub_date and article.get("pub_date"):
            metadata["pub_date"] = article["pub_date"]

        if article.get("language"):
            metadata["language"] = article["language"]

        if article.get("site_name"):
            metadata["site_name"] = article["site_name"]

        if article.get("summary"):
            metadata["summary"] = article["summary"]

        if article.get("image"):
            metadata["image"] = article["image"]

        if self.extract_images and article.get("images"):
            metadata["images"] = article["images"]

        return Document(page_content=content, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents one at a time.

        This is the preferred method for loading documents as it uses
        less memory by yielding documents one at a time.

        Yields:
            Document objects, one for each successfully extracted article
        """
        for url in self.urls:
            article = self._extract_article(url)
            if article:
                yield self._create_document(article, url)

    def load(self) -> List[Document]:
        """Load articles from the provided URLs.

        Returns:
            List of Document objects, one for each successfully extracted article
        """
        return list(self.lazy_load())
