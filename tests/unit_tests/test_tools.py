"""Tests for Ujeebu tools."""

import pytest
from unittest.mock import Mock, patch
from langchain_ujeebu.tools import UjeebuExtractTool, UjeebuExtractInput


class TestUjeebuExtractTool:
    """Test cases for UjeebuExtractTool."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key fixture."""
        return "test_api_key_12345"

    @pytest.fixture
    def tool(self, mock_api_key):
        """Create a UjeebuExtractTool instance."""
        with patch("langchain_ujeebu.tools.UjeebuClient"):
            return UjeebuExtractTool(api_key=mock_api_key)

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

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_tool_initialization(self, mock_client_class, mock_api_key):
        """Test tool initialization with API key."""
        tool = UjeebuExtractTool(api_key=mock_api_key)
        assert tool.name == "ujeebu_extract"
        assert tool.api_key == mock_api_key
        assert "extract" in tool.description.lower()
        mock_client_class.assert_called_once()

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_tool_initialization_from_env(
        self, mock_client_class, monkeypatch, mock_api_key
    ):
        """Test tool initialization from environment variable."""
        monkeypatch.setenv("UJEEBU_API_KEY", mock_api_key)
        tool = UjeebuExtractTool()
        assert tool.api_key == mock_api_key

    def test_tool_initialization_no_api_key(self, monkeypatch):
        """Test tool initialization fails without API key."""
        monkeypatch.delenv("UJEEBU_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key must be provided"):
            UjeebuExtractTool()

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_run_success(self, mock_client_class, mock_api_key, mock_article_response):
        """Test successful article extraction."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = UjeebuExtractTool(api_key=mock_api_key)
        result = tool._run(url="https://example.com/article")

        assert "Test Article Title" in result
        assert "John Doe" in result
        assert "2024-01-01" in result
        assert "This is the article text content" in result
        mock_client.extract.assert_called_once()

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_run_with_html(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test article extraction with HTML content."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = UjeebuExtractTool(api_key=mock_api_key)
        result = tool._run(url="https://example.com/article", html=True)

        assert "<p>This is the article HTML content.</p>" in result
        mock_client.extract.assert_called_once()

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_run_with_images(
        self, mock_client_class, mock_api_key, mock_article_response
    ):
        """Test article extraction with images."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = mock_article_response
        mock_response.raise_for_status.return_value = None
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = UjeebuExtractTool(api_key=mock_api_key)
        result = tool._run(url="https://example.com/article", images=True)

        assert "image1.jpg" in result
        mock_client.extract.assert_called_once()

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_run_401_error(self, mock_client_class, mock_api_key):
        """Test handling of 401 authentication error."""
        import requests as req

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 401
        http_error = req.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = UjeebuExtractTool(api_key=mock_api_key)
        result = tool._run(url="https://example.com/article")

        assert "Invalid API key" in result or "Error" in result

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_run_404_error(self, mock_client_class, mock_api_key):
        """Test handling of 404 not found error."""
        import requests as req

        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        http_error = req.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_client.extract.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = UjeebuExtractTool(api_key=mock_api_key)
        result = tool._run(url="https://example.com/article")

        assert "Error" in result

    @patch("langchain_ujeebu.tools.UjeebuClient")
    def test_run_timeout_error(self, mock_client_class, mock_api_key):
        """Test handling of timeout error."""
        mock_client = Mock()
        mock_client.extract.side_effect = Exception("Timeout")
        mock_client_class.return_value = mock_client

        tool = UjeebuExtractTool(api_key=mock_api_key)
        result = tool._run(url="https://example.com/article")

        assert "Error" in result

    def test_input_schema(self):
        """Test the input schema."""
        schema = UjeebuExtractInput(url="https://example.com/article")
        assert schema.url == "https://example.com/article"
        assert schema.text is True
        assert schema.html is False
        assert schema.author is True
        assert schema.quick_mode is False

    @pytest.mark.asyncio
    @patch("langchain_ujeebu.tools.UjeebuClient")
    async def test_arun(self, mock_client_class, mock_api_key):
        """Test async run method."""
        tool = UjeebuExtractTool(api_key=mock_api_key)
        with patch.object(tool, "_run", return_value="Test result") as mock_run:
            result = await tool._arun(url="https://example.com/article")
            assert result == "Test result"
            mock_run.assert_called_once()
