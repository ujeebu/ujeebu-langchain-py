"""Standard unit tests for Ujeebu tools following LangChain guidelines."""

import pytest
from unittest.mock import patch, Mock


class TestUjeebuExtractToolStandard:
    """Standard unit tests for UjeebuExtractTool.

    These tests verify the tool meets LangChain's standard tool requirements.
    """

    @pytest.fixture
    def tool(self):
        """Create tool instance for testing."""
        with patch("langchain_ujeebu.tools.UjeebuClient"):
            from langchain_ujeebu import UjeebuExtractTool

            return UjeebuExtractTool(api_key="test_key")

    def test_tool_has_name(self, tool):
        """Tool must have a name property."""
        assert tool.name is not None
        assert isinstance(tool.name, str)
        assert len(tool.name) > 0
        assert tool.name == "ujeebu_extract"

    def test_tool_has_description(self, tool):
        """Tool must have a description property."""
        assert tool.description is not None
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_tool_has_args_schema(self, tool):
        """Tool must have an args_schema property."""
        assert tool.args_schema is not None

    def test_tool_args_schema_is_pydantic_model(self, tool):
        """Tool args_schema must be a valid Pydantic model."""
        from pydantic import BaseModel

        assert issubclass(tool.args_schema, BaseModel)

    def test_tool_args_schema_has_url_field(self, tool):
        """Tool args_schema must have a url field for extraction."""
        schema = tool.args_schema
        fields = schema.model_fields
        assert "url" in fields
        assert fields["url"].is_required()

    def test_tool_inherits_from_base_tool(self, tool):
        """Tool must inherit from BaseTool."""
        from langchain_core.tools import BaseTool

        assert isinstance(tool, BaseTool)

    def test_tool_has_run_method(self, tool):
        """Tool must implement _run method."""
        assert hasattr(tool, "_run")
        assert callable(tool._run)

    def test_tool_has_arun_method(self, tool):
        """Tool must implement _arun method for async support."""
        assert hasattr(tool, "_arun")
        assert callable(tool._arun)

    def test_tool_invoke_returns_string(self, tool):
        """Tool _run should return a string."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "article": {
                "title": "Test Title",
                "text": "Test content",
                "author": "Test Author",
            }
        }
        mock_response.raise_for_status.return_value = None
        tool._client.extract.return_value = mock_response

        result = tool._run(url="https://example.com/article")
        assert isinstance(result, str)


class TestUjeebuScrapeToolStandard:
    """Standard unit tests for UjeebuScrapeTool.

    These tests verify the tool meets LangChain's standard tool requirements.
    """

    @pytest.fixture
    def tool(self):
        """Create tool instance for testing."""
        with patch("langchain_ujeebu.tools.UjeebuClient"):
            from langchain_ujeebu import UjeebuScrapeTool

            return UjeebuScrapeTool(api_key="test_key")

    def test_tool_has_name(self, tool):
        """Tool must have a name property."""
        assert tool.name is not None
        assert isinstance(tool.name, str)
        assert len(tool.name) > 0
        assert tool.name == "ujeebu_scrape"

    def test_tool_has_description(self, tool):
        """Tool must have a description property."""
        assert tool.description is not None
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_tool_has_args_schema(self, tool):
        """Tool must have an args_schema property."""
        assert tool.args_schema is not None

    def test_tool_args_schema_is_pydantic_model(self, tool):
        """Tool args_schema must be a valid Pydantic model."""
        from pydantic import BaseModel

        assert issubclass(tool.args_schema, BaseModel)

    def test_tool_args_schema_has_url_field(self, tool):
        """Tool args_schema must have a url field for scraping."""
        schema = tool.args_schema
        fields = schema.model_fields
        assert "url" in fields
        assert fields["url"].is_required()

    def test_tool_inherits_from_base_tool(self, tool):
        """Tool must inherit from BaseTool."""
        from langchain_core.tools import BaseTool

        assert isinstance(tool, BaseTool)

    def test_tool_has_run_method(self, tool):
        """Tool must implement _run method."""
        assert hasattr(tool, "_run")
        assert callable(tool._run)

    def test_tool_has_arun_method(self, tool):
        """Tool must implement _arun method for async support."""
        assert hasattr(tool, "_arun")
        assert callable(tool._arun)

    def test_tool_invoke_returns_string(self, tool):
        """Tool _run should return a string."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "html": "<html><body>Test content</body></html>"
        }
        mock_response.raise_for_status.return_value = None
        tool._client.scrape.return_value = mock_response

        result = tool._run(url="https://example.com")
        assert isinstance(result, str)
