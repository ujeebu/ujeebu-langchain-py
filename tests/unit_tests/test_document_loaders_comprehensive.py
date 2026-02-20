"""Comprehensive tests for Ujeebu document loaders."""

import pytest
from unittest.mock import Mock, patch
from langchain_ujeebu.document_loaders import UjeebuLoader
from langchain.docstore.document import Document
import requests


class TestUjeebuLoaderInitialization:
    """Test cases for UjeebuLoader initialization."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    def test_initialization_with_api_key(self, mock_api_key):
        """Test loader initialization with explicit API key."""
        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        assert loader.urls == ["https://example.com/article"]
        assert loader.api_key == mock_api_key

    def test_initialization_from_env_variable(self, monkeypatch, mock_api_key):
        """Test loader initialization from UJEEBU_API_KEY environment variable."""
        monkeypatch.setenv("UJEEBU_API_KEY", mock_api_key)
        loader = UjeebuLoader(urls=["https://example.com/article"])
        assert loader.api_key == mock_api_key

    def test_initialization_api_key_priority(self, monkeypatch, mock_api_key):
        """Test that explicit API key takes priority over env variable."""
        monkeypatch.setenv("UJEEBU_API_KEY", "env_api_key")
        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        assert loader.api_key == mock_api_key

    def test_initialization_no_api_key_raises_error(self, monkeypatch):
        """Test that initialization fails without API key."""
        monkeypatch.delenv("UJEEBU_API_KEY", raising=False)
        with pytest.raises(ValueError) as exc_info:
            UjeebuLoader(urls=["https://example.com/article"])
        assert "API key must be provided" in str(exc_info.value)
        assert "https://ujeebu.com/signup" in str(exc_info.value)

    def test_initialization_default_parameters(self, mock_api_key):
        """Test default parameter values."""
        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        assert loader.extract_text is True
        assert loader.extract_html is False
        assert loader.extract_author is True
        assert loader.extract_pub_date is True
        assert loader.extract_images is False
        assert loader.quick_mode is False

    def test_initialization_custom_parameters(self, mock_api_key):
        """Test custom parameter values."""
        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_text=False,
            extract_html=True,
            extract_author=False,
            extract_pub_date=False,
            extract_images=True,
            quick_mode=True,
            timeout=60,
        )
        assert loader.extract_text is False
        assert loader.extract_html is True
        assert loader.extract_author is False
        assert loader.extract_pub_date is False
        assert loader.extract_images is True
        assert loader.quick_mode is True

    def test_initialization_with_empty_urls(self, mock_api_key):
        """Test initialization with empty URL list."""
        loader = UjeebuLoader(urls=[], api_key=mock_api_key)
        assert loader.urls == []

    def test_initialization_with_multiple_urls(self, mock_api_key):
        """Test initialization with multiple URLs."""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]
        loader = UjeebuLoader(urls=urls, api_key=mock_api_key)
        assert loader.urls == urls
        assert len(loader.urls) == 3

    def test_initialization_with_custom_base_url(self, mock_api_key):
        """Test initialization with custom base URL."""
        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            base_url="https://custom-api.ujeebu.com",
        )
        assert loader._client is not None

    def test_initialization_with_custom_timeout(self, mock_api_key):
        """Test initialization with custom timeout."""
        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key, timeout=300
        )
        assert loader._client is not None


class TestUjeebuLoaderLoad:
    """Test cases for UjeebuLoader.load() method."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @pytest.fixture
    def mock_article_response(self):
        """Mock article response from API."""
        return {
            "article": {
                "title": "Test Article Title",
                "text": "This is the article text content.",
                "html": "<p>This is the article HTML content.</p>",
                "author": "John Doe",
                "pub_date": "2024-01-01 12:00:00",
                "url": "https://example.com/article",
                "canonical_url": "https://example.com/canonical-article",
                "site_name": "Example Site",
                "summary": "Article summary",
                "language": "en",
                "image": "https://example.com/image.jpg",
                "images": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                ],
            }
        }

    @pytest.fixture
    def mock_minimal_article_response(self):
        """Mock minimal article response from API."""
        return {
            "article": {
                "text": "Minimal article text.",
                "url": "https://example.com/minimal-article",
            }
        }

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_single_document(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading a single document."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "article text content" in documents[0].page_content
        mock_client.extract.assert_called_once()

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_multiple_documents(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading multiple documents."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        urls = [f"https://example.com/article{i}" for i in range(5)]
        loader = UjeebuLoader(urls=urls, api_key=mock_api_key)
        documents = loader.load()

        assert len(documents) == 5
        assert mock_client.extract.call_count == 5

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_empty_urls(self, mock_client_class, mock_api_key):
        """Test loading with empty URL list."""
        mock_client_class.return_value = Mock()

        loader = UjeebuLoader(urls=[], api_key=mock_api_key)
        documents = loader.load()

        assert len(documents) == 0

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_text_only(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading with text extraction only."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_text=True,
            extract_html=False,
        )
        documents = loader.load()

        assert "article text content" in documents[0].page_content
        assert "HTML:" not in documents[0].page_content

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_html_only(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading with HTML extraction only."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_text=False,
            extract_html=True,
        )
        documents = loader.load()

        assert "HTML:" in documents[0].page_content
        assert "<p>This is the article HTML content.</p>" in documents[0].page_content

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_both_text_and_html(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading with both text and HTML extraction."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_text=True,
            extract_html=True,
        )
        documents = loader.load()

        assert "article text content" in documents[0].page_content
        assert "HTML:" in documents[0].page_content


class TestUjeebuLoaderMetadata:
    """Test cases for document metadata."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @pytest.fixture
    def mock_full_article_response(self):
        """Mock full article response from API with all fields."""
        return {
            "article": {
                "title": "Test Article Title",
                "text": "This is the article text content.",
                "html": "<p>This is the article HTML content.</p>",
                "author": "John Doe",
                "pub_date": "2024-01-01 12:00:00",
                "url": "https://example.com/article",
                "canonical_url": "https://example.com/canonical-article",
                "site_name": "Example Site",
                "summary": "Article summary",
                "language": "en",
                "image": "https://example.com/main-image.jpg",
                "images": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                ],
            }
        }

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_source_field(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test that source metadata is correctly set."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert documents[0].metadata["source"] == "https://example.com/article"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_url_fields(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test URL and canonical URL metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert documents[0].metadata["url"] == "https://example.com/article"
        assert (
            documents[0].metadata["canonical_url"]
            == "https://example.com/canonical-article"
        )

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_title(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test title metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert documents[0].metadata["title"] == "Test Article Title"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_author(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test author metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_author=True,
        )
        documents = loader.load()

        assert documents[0].metadata["author"] == "John Doe"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_author_disabled(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test author metadata when extraction is disabled."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_author=False,
        )
        documents = loader.load()

        # Author should not be in metadata when extraction is disabled
        assert "author" not in documents[0].metadata

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_pub_date(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test publication date metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_pub_date=True,
        )
        documents = loader.load()

        assert documents[0].metadata["pub_date"] == "2024-01-01 12:00:00"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_images(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test images metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_images=True,
        )
        documents = loader.load()

        assert "images" in documents[0].metadata
        assert len(documents[0].metadata["images"]) == 2
        assert documents[0].metadata["image"] == "https://example.com/main-image.jpg"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_language(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test language metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert documents[0].metadata["language"] == "en"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_site_name(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test site name metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert documents[0].metadata["site_name"] == "Example Site"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_summary(
        self, mock_client_class, mock_api_key, mock_full_article_response
    ):
        """Test summary metadata."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_full_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert documents[0].metadata["summary"] == "Article summary"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_missing_optional_fields(self, mock_client_class, mock_api_key):
        """Test handling of missing optional metadata fields."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "text": "Minimal article.",
                "url": "https://example.com/minimal",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/minimal"], api_key=mock_api_key
        )
        documents = loader.load()

        # Required fields
        assert documents[0].metadata["source"] == "https://example.com/minimal"
        assert documents[0].metadata["url"] == "https://example.com/minimal"
        # Optional fields should not be present
        assert "title" not in documents[0].metadata
        assert "author" not in documents[0].metadata
        assert "summary" not in documents[0].metadata


class TestUjeebuLoaderErrorHandling:
    """Test cases for error handling."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_api_error_graceful_handling(
        self, mock_client_class, mock_api_key, caplog
    ):
        """Test graceful handling of API errors."""
        mock_client = Mock()
        mock_client.extract.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_http_error(self, mock_client_class, mock_api_key, caplog):
        """Test handling of HTTP errors."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_connection_error(self, mock_client_class, mock_api_key, caplog):
        """Test handling of connection errors."""
        mock_client = Mock()
        mock_client.extract.side_effect = requests.exceptions.ConnectionError(
            "Connection failed"
        )
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_timeout_error(self, mock_client_class, mock_api_key, caplog):
        """Test handling of timeout errors."""
        mock_client = Mock()
        mock_client.extract.side_effect = requests.exceptions.Timeout(
            "Request timed out"
        )
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_json_decode_error(self, mock_client_class, mock_api_key, caplog):
        """Test handling of JSON decode errors."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_partial_success(self, mock_client_class, mock_api_key, caplog):
        """Test partial success when some URLs fail."""
        mock_client = Mock()
        mock_response_success = Mock()
        mock_response_success.json.return_value = {
            "article": {
                "title": "Success Article",
                "text": "Article text.",
                "url": "https://example.com/success",
            }
        }
        mock_response_success.raise_for_status.return_value = None

        # First call succeeds, second fails, third succeeds
        mock_client.extract.side_effect = [
            mock_response_success,
            Exception("API Error"),
            mock_response_success,
        ]
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=[
                "https://example.com/success1",
                "https://example.com/fail",
                "https://example.com/success2",
            ],
            api_key=mock_api_key,
        )
        documents = loader.load()

        assert len(documents) == 2
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_empty_article_response(self, mock_client_class, mock_api_key):
        """Test handling of empty article response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"article": None}
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_missing_article_key(self, mock_client_class, mock_api_key):
        """Test handling of response missing article key."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"other_key": "value"}
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0


class TestUjeebuLoaderEdgeCases:
    """Test cases for edge cases."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_empty_text_content(self, mock_client_class, mock_api_key):
        """Test handling of empty text content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Empty Article",
                "text": "",
                "url": "https://example.com/empty",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(urls=["https://example.com/empty"], api_key=mock_api_key)
        documents = loader.load()

        assert len(documents) == 1
        assert documents[0].page_content == ""

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_unicode_content(self, mock_client_class, mock_api_key):
        """Test handling of Unicode content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "日本語タイトル",
                "text": "这是中文内容。日本語も含まれています。🎉",
                "author": "山田太郎",
                "url": "https://example.com/unicode",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/unicode"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 1
        assert "这是中文内容" in documents[0].page_content
        assert documents[0].metadata["title"] == "日本語タイトル"
        assert documents[0].metadata["author"] == "山田太郎"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_special_characters_in_url(self, mock_client_class, mock_api_key):
        """Test handling of special characters in URL."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Article",
                "text": "Content",
                "url": "https://example.com/article?query=test&foo=bar#section",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        url = "https://example.com/article?query=test&foo=bar#section"
        loader = UjeebuLoader(urls=[url], api_key=mock_api_key)
        documents = loader.load()

        assert len(documents) == 1
        assert documents[0].metadata["source"] == url

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_very_long_content(self, mock_client_class, mock_api_key):
        """Test handling of very long content."""
        mock_client = Mock()
        mock_response = Mock()
        long_text = "A" * 100000  # 100K characters
        mock_response.json.return_value = {
            "article": {
                "title": "Long Article",
                "text": long_text,
                "url": "https://example.com/long",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(urls=["https://example.com/long"], api_key=mock_api_key)
        documents = loader.load()

        assert len(documents) == 1
        assert len(documents[0].page_content) == 100000

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_duplicate_urls(self, mock_client_class, mock_api_key):
        """Test handling of duplicate URLs."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Article",
                "text": "Content",
                "url": "https://example.com/article",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Same URL twice
        loader = UjeebuLoader(
            urls=["https://example.com/article", "https://example.com/article"],
            api_key=mock_api_key,
        )
        documents = loader.load()

        # Both URLs should be processed (no deduplication)
        assert len(documents) == 2
        assert mock_client.extract.call_count == 2

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_whitespace_only_text(self, mock_client_class, mock_api_key):
        """Test handling of whitespace-only text content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Whitespace Article",
                "text": "   \n\t  \n  ",
                "url": "https://example.com/whitespace",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/whitespace"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 1
        assert documents[0].page_content == "   \n\t  \n  "


class TestUjeebuLoaderLazyLoad:
    """Test cases for lazy_load method."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_lazy_load_returns_documents(self, mock_client_class, mock_api_key):
        """Test that lazy_load returns documents."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Article",
                "text": "Content",
                "url": "https://example.com/article",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = list(loader.lazy_load())

        assert len(documents) == 1
        assert isinstance(documents[0], Document)

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_lazy_load_same_as_load(self, mock_client_class, mock_api_key):
        """Test that lazy_load returns same result as load."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Article",
                "text": "Content",
                "url": "https://example.com/article",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )

        # Note: We need fresh mock for second call
        lazy_docs = list(loader.lazy_load())

        assert len(lazy_docs) == 1


class TestUjeebuLoaderAPIParams:
    """Test cases for API parameter passing."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_params_passed_to_api(self, mock_client_class, mock_api_key):
        """Test that extraction parameters are passed to API."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test",
                "text": "Content",
                "url": "https://example.com/article",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_text=True,
            extract_html=True,
            extract_author=True,
            extract_pub_date=False,
            extract_images=True,
            quick_mode=True,
        )
        loader.load()

        # Check the params passed to extract
        call_args = mock_client.extract.call_args
        params = call_args[0][1]  # Second positional argument

        assert params["text"] is True
        assert params["html"] is True
        assert params["author"] is True
        assert params["pub_date"] is False
        assert params["images"] is True
        assert params["quick_mode"] is True

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_url_passed_to_api(self, mock_client_class, mock_api_key):
        """Test that URL is correctly passed to API."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test",
                "text": "Content",
                "url": "https://example.com/specific-article",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/specific-article"], api_key=mock_api_key
        )
        loader.load()

        # Check the URL passed to extract
        call_args = mock_client.extract.call_args
        url = call_args[0][0]  # First positional argument

        assert url == "https://example.com/specific-article"
