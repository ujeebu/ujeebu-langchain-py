"""Integration tests for Ujeebu tools (requires API key).

These tests require a valid UJEEBU_API_KEY environment variable to run.
They make real API calls and verify end-to-end functionality.
"""

import os
import pytest

# Skip all tests if no API key is set
pytestmark = pytest.mark.skipif(
    not os.getenv("UJEEBU_API_KEY"),
    reason="UJEEBU_API_KEY environment variable not set",
)


def assert_no_error(result: str) -> None:
    """Assert that the result does not contain an error message."""
    assert isinstance(result, str), "Result should be a string"
    assert not result.startswith("Error:"), f"Result contains error: {result}"
    assert not result.startswith(
        "Error extracting"
    ), f"Result contains extraction error: {result}"
    assert not result.startswith(
        "Error scraping"
    ), f"Result contains scraping error: {result}"


def is_error_result(result: str) -> bool:
    """Check if the result is an error message."""
    if not isinstance(result, str):
        return False
    return (
        result.startswith("Error:")
        or result.startswith("Error extracting")
        or result.startswith("Error scraping")
        or "Error:" in result[:50]
    )


def assert_has_content(result: str, min_length: int = 50) -> None:
    """Assert that the result has meaningful content."""
    assert_no_error(result)
    assert (
        len(result) >= min_length
    ), f"Result too short ({len(result)} chars): {result[:100]}"


class TestUjeebuExtractToolIntegration:
    """Integration tests for UjeebuExtractTool with real API calls."""

    @pytest.fixture
    def tool(self):
        """Create tool instance with real API key."""
        from langchain_ujeebu import UjeebuExtractTool

        return UjeebuExtractTool()

    def test_extract_real_article(self, tool):
        """Test extracting content from a real article."""
        url = (
            "https://ujeebu.com/blog/"
            "web-scraping-an-essential-tool-for-business-intelligence/"
        )
        result = tool._run(url=url)
        assert_has_content(result, min_length=100)
        # Should contain title or content sections
        assert any(
            keyword in result for keyword in ["Title:", "Content:", "Summary:"]
        ), f"Result missing expected sections: {result[:200]}"

    def test_extract_with_text_option(self, tool):
        """Test extraction with text option enabled."""
        url = (
            "https://ujeebu.com/blog/"
            "web-scraping-an-essential-tool-for-business-intelligence/"
        )
        result = tool._run(
            url=url,
            text=True,
            html=False,
        )
        assert_has_content(result, min_length=100)

    def test_extract_handles_invalid_url(self, tool):
        """Test that invalid URLs are handled gracefully with error message."""
        result = tool._run(url="https://this-url-does-not-exist-12345.com/article")
        assert isinstance(result, str)
        # Should contain error message for invalid URL
        assert (
            is_error_result(result) or len(result) == 0
        ), f"Expected error or empty result for invalid URL, got: {result[:100]}"

    def test_extract_returns_structured_content(self, tool):
        """Test that extraction returns structured content with expected fields."""
        url = (
            "https://ujeebu.com/blog/"
            "web-scraping-an-essential-tool-for-business-intelligence/"
        )
        result = tool._run(url=url)
        assert_no_error(result)
        # Should be formatted with labels
        assert ":" in result, "Result should contain key-value formatted content"


class TestUjeebuScrapeToolIntegration:
    """Integration tests for UjeebuScrapeTool with real API calls."""

    @pytest.fixture
    def tool(self):
        """Create tool instance with real API key."""
        from langchain_ujeebu import UjeebuScrapeTool

        return UjeebuScrapeTool()

    def test_scrape_real_page(self, tool):
        """Test scraping a real web page."""
        result = tool._run(url="https://ujeebu.com")
        assert_has_content(result, min_length=100)
        # Should contain HTML content
        assert "<" in result and ">" in result, "Result should contain HTML tags"

    def test_scrape_with_js_rendering(self, tool):
        """Test scraping with JavaScript rendering enabled."""
        result = tool._run(url="https://ujeebu.com", js_rendering=True)
        assert_has_content(result, min_length=100)

    def test_scrape_handles_invalid_url(self, tool):
        """Test that invalid URLs are handled gracefully with error message."""
        result = tool._run(url="https://this-url-does-not-exist-12345.com")
        assert isinstance(result, str)
        # Should contain error message for invalid URL
        assert (
            is_error_result(result) or len(result) == 0
        ), f"Expected error or empty result for invalid URL, got: {result[:100]}"

    def test_scrape_returns_html_by_default(self, tool):
        """Test that scraping returns HTML content by default."""
        result = tool._run(url="https://ujeebu.com")
        assert_has_content(result, min_length=100)
        # Should contain common HTML elements
        assert any(
            tag in result.lower()
            for tag in ["<html", "<head", "<body", "<div", "<!doctype"]
        ), f"Result should contain HTML elements: {result[:200]}"


class TestUjeebuLoaderIntegration:
    """Integration tests for UjeebuLoader with real API calls."""

    @pytest.fixture
    def loader(self):
        """Create loader instance with real API key."""
        from langchain_ujeebu import UjeebuLoader

        url = (
            "https://ujeebu.com/blog/"
            "web-scraping-an-essential-tool-for-business-intelligence/"
        )
        return UjeebuLoader(urls=[url])

    def test_load_real_article(self, loader):
        """Test loading a real article."""
        from langchain_core.documents import Document

        documents = loader.load()
        assert isinstance(documents, list)
        assert len(documents) > 0, "Should load at least one document"

        doc = documents[0]
        assert isinstance(doc, Document)
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")

        # Verify metadata is present (API may return empty text for some articles)
        assert "source" in doc.metadata, "Document should have source in metadata"

        # Verify we got meaningful metadata from the extraction
        has_content = len(doc.page_content) > 0
        has_title = "title" in doc.metadata and doc.metadata["title"]
        has_summary = "summary" in doc.metadata and doc.metadata["summary"]

        assert has_content or has_title or has_summary, (
            "Document should have content or meaningful metadata. "
            f"Got: content={len(doc.page_content)}, "
            f"metadata={doc.metadata}"
        )

    def test_lazy_load_real_article(self, loader):
        """Test lazy loading a real article."""
        from langchain_core.documents import Document

        docs_loaded = 0
        for doc in loader.lazy_load():
            assert isinstance(doc, Document)
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")

            # Verify we got something from the extraction
            has_content = len(doc.page_content) > 0
            has_title = "title" in doc.metadata and doc.metadata["title"]

            assert has_content or has_title, (
                "Document should have content or title. "
                f"Got: content={len(doc.page_content)}, "
                f"metadata={doc.metadata}"
            )

            docs_loaded += 1
            break  # Just test first document

        assert docs_loaded > 0, "Should have loaded at least one document"

    def test_load_multiple_articles(self):
        """Test loading multiple articles."""
        from langchain_ujeebu import UjeebuLoader
        from langchain_core.documents import Document

        urls = [
            (
                "https://ujeebu.com/blog/"
                "web-scraping-an-essential-tool-"
                "for-business-intelligence/"
            ),
            (
                "https://ujeebu.com/blog/"
                "bypassing-anti-bot-systems-"
                "comprehensive-guide/"
            ),
        ]
        loader = UjeebuLoader(urls=urls)

        documents = loader.load()
        assert isinstance(documents, list)
        assert len(documents) >= 1, "Should load at least one document"

        for doc in documents:
            assert isinstance(doc, Document)
            assert "source" in doc.metadata, "Each document should have source metadata"

            # Check that we got something from the extraction
            has_content = len(doc.page_content) > 0
            has_title = "title" in doc.metadata and doc.metadata["title"]

            assert has_content or has_title, (
                "Each document should have content or title. "
                f"Got: content={len(doc.page_content)}, "
                f"metadata keys={list(doc.metadata.keys())}"
            )
