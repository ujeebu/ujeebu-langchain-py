"""Standard unit tests for Ujeebu document loader following LangChain guidelines."""

import pytest
from unittest.mock import patch, Mock


class TestUjeebuLoaderStandard:
    """Standard unit tests for UjeebuLoader.

    These tests verify the loader meets LangChain's standard loader requirements.
    """

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @pytest.fixture
    def loader(self, mock_api_key):
        """Create loader instance for testing."""
        with patch("langchain_ujeebu.document_loaders.UjeebuClient"):
            from langchain_ujeebu import UjeebuLoader

            return UjeebuLoader(
                urls=["https://example.com/article"], api_key=mock_api_key
            )

    def test_loader_inherits_from_base_loader(self, loader):
        """Loader must inherit from BaseLoader."""
        from langchain_core.document_loaders import BaseLoader

        assert isinstance(loader, BaseLoader)

    def test_loader_has_load_method(self, loader):
        """Loader must implement load method."""
        assert hasattr(loader, "load")
        assert callable(loader.load)

    def test_loader_has_lazy_load_method(self, loader):
        """Loader must implement lazy_load method."""
        assert hasattr(loader, "lazy_load")
        assert callable(loader.lazy_load)

    def test_lazy_load_returns_iterator(self, loader):
        """lazy_load must return an iterator, not a list."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Article",
                "text": "Test content",
            }
        }
        mock_response.raise_for_status.return_value = None
        loader._client.extract.return_value = mock_response

        result = loader.lazy_load()
        # Should be a generator/iterator, not a list
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_returns_list_of_documents(self, loader):
        """load must return a list of Document objects."""
        from langchain_core.documents import Document

        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Article",
                "text": "Test content",
            }
        }
        mock_response.raise_for_status.return_value = None
        loader._client.extract.return_value = mock_response

        result = loader.load()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Document)

    def test_document_has_page_content(self, loader):
        """Documents must have page_content attribute."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Article",
                "text": "Test content",
            }
        }
        mock_response.raise_for_status.return_value = None
        loader._client.extract.return_value = mock_response

        documents = loader.load()
        assert len(documents) > 0
        assert hasattr(documents[0], "page_content")
        assert isinstance(documents[0].page_content, str)

    def test_document_has_metadata(self, loader):
        """Documents must have metadata attribute."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Article",
                "text": "Test content",
                "url": "https://example.com/article",
            }
        }
        mock_response.raise_for_status.return_value = None
        loader._client.extract.return_value = mock_response

        documents = loader.load()
        assert len(documents) > 0
        assert hasattr(documents[0], "metadata")
        assert isinstance(documents[0].metadata, dict)

    def test_document_metadata_has_source(self, loader):
        """Document metadata must include source field."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Article",
                "text": "Test content",
            }
        }
        mock_response.raise_for_status.return_value = None
        loader._client.extract.return_value = mock_response

        documents = loader.load()
        assert len(documents) > 0
        assert "source" in documents[0].metadata

    def test_loader_handles_multiple_urls(self, mock_api_key):
        """Loader must handle multiple URLs."""
        with patch(
            "langchain_ujeebu.document_loaders.UjeebuClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "article": {
                    "title": "Test Article",
                    "text": "Test content",
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_client.extract.return_value = mock_response
            mock_client_class.return_value = mock_client

            from langchain_ujeebu import UjeebuLoader

            loader = UjeebuLoader(
                urls=[
                    "https://example.com/article1",
                    "https://example.com/article2",
                    "https://example.com/article3",
                ],
                api_key=mock_api_key,
            )

            documents = loader.load()
            assert len(documents) == 3
            assert mock_client.extract.call_count == 3

    def test_loader_handles_failed_extractions(self, mock_api_key):
        """Loader must gracefully handle failed extractions."""
        with patch(
            "langchain_ujeebu.document_loaders.UjeebuClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.extract.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_client

            from langchain_ujeebu import UjeebuLoader

            loader = UjeebuLoader(
                urls=["https://example.com/article"], api_key=mock_api_key
            )

            # Should not raise, should return empty list
            documents = loader.load()
            assert isinstance(documents, list)
            assert len(documents) == 0
