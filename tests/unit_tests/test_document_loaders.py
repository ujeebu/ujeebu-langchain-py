"""Tests for Ujeebu document loaders."""

import pytest
from unittest.mock import Mock, patch
from langchain_ujeebu.document_loaders import UjeebuLoader


class TestUjeebuLoader:
    """Test cases for UjeebuLoader."""

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
                "canonical_url": "https://example.com/article",
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

    def test_loader_initialization(self, mock_api_key):
        """Test loader initialization with API key."""
        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        assert loader.urls == ["https://example.com/article"]
        assert loader.api_key == mock_api_key
        assert loader.extract_text is True
        assert loader.extract_html is False

    def test_loader_initialization_from_env(self, monkeypatch, mock_api_key):
        """Test loader initialization from environment variable."""
        monkeypatch.setenv("UJEEBU_API_KEY", mock_api_key)
        loader = UjeebuLoader(urls=["https://example.com/article"])
        assert loader.api_key == mock_api_key

    def test_loader_initialization_no_api_key(self, monkeypatch):
        """Test loader initialization fails without API key."""
        monkeypatch.delenv("UJEEBU_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key must be provided"):
            UjeebuLoader(urls=["https://example.com/article"])

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
        doc = documents[0]
        assert "article text content" in doc.page_content
        assert doc.metadata["title"] == "Test Article Title"
        assert doc.metadata["author"] == "John Doe"
        assert doc.metadata["source"] == "https://example.com/article"
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

        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]
        loader = UjeebuLoader(urls=urls, api_key=mock_api_key)
        documents = loader.load()

        assert len(documents) == 3
        assert mock_client.extract.call_count == 3

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_html(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading documents with HTML content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_html=True,
        )
        documents = loader.load()

        assert len(documents) == 1
        assert "HTML:" in documents[0].page_content
        assert "<p>This is the article HTML content.</p>" in documents[0].page_content

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_images(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading documents with images."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_images=True,
        )
        documents = loader.load()

        assert len(documents) == 1
        assert "images" in documents[0].metadata
        assert len(documents[0].metadata["images"]) == 2

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_quick_mode(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test loading documents with quick mode enabled."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key, quick_mode=True
        )
        documents = loader.load()

        assert len(documents) == 1
        # Verify quick_mode was passed in params
        call_args = mock_client.extract.call_args
        assert call_args[0][1]["quick_mode"] == 1

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_with_error(self, mock_client_class, mock_api_key, caplog):
        """Test loading documents with API error."""
        mock_client = Mock()
        mock_client.extract.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        documents = loader.load()

        assert len(documents) == 0
        # Error is now logged, not printed - check log records
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_load_partial_success(
        self, mock_client_class, mock_api_key, mock_article_response, caplog
    ):
        """Test loading documents with partial success."""
        mock_client = Mock()
        # First call succeeds, second fails
        mock_response_success = Mock()
        mock_response_success.json.return_value = mock_article_response
        mock_response_success.raise_for_status.return_value = None

        mock_client.extract.side_effect = [
            mock_response_success,
            Exception("API Error"),
        ]
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article1", "https://example.com/article2"],
            api_key=mock_api_key,
        )
        documents = loader.load()

        assert len(documents) == 1
        assert documents[0].metadata["title"] == "Test Article Title"
        # Error is now logged, not printed
        assert "Error extracting article" in caplog.text

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_lazy_load(self, mock_client_class, mock_api_key, mock_article_response):
        """Test lazy loading documents."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"], api_key=mock_api_key
        )
        # lazy_load now returns a generator, convert to list to test
        documents = list(loader.lazy_load())

        assert len(documents) == 1
        assert documents[0].metadata["title"] == "Test Article Title"

    @patch("langchain_ujeebu.document_loaders.UjeebuClient")
    def test_metadata_fields(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test all metadata fields are correctly set."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        loader = UjeebuLoader(
            urls=["https://example.com/article"],
            api_key=mock_api_key,
            extract_images=True,
        )
        documents = loader.load()

        metadata = documents[0].metadata
        assert metadata["title"] == "Test Article Title"
        assert metadata["author"] == "John Doe"
        assert metadata["pub_date"] == "2024-01-01 12:00:00"
        assert metadata["site_name"] == "Example Site"
        assert metadata["language"] == "en"
        assert metadata["summary"] == "Article summary"
        assert metadata["image"] == "https://example.com/image.jpg"
        assert "images" in metadata
