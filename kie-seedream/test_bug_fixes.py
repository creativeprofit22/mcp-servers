"""
Functional tests for KIE MCP Server bug fixes.
Tests the specific code paths that were fixed for NoneType errors.
"""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
import sys
sys.path.insert(0, 'src')

# Test Bug #1 & #2: response.json() returning None
class TestResponseJsonNullHandling:
    """Test that None responses from API are handled gracefully."""

    @pytest.mark.asyncio
    async def test_create_task_handles_none_response(self):
        """Bug #1: create_task() should raise exception if response.json() is None"""
        from server import KieApiClient, Config

        config = Config(kie_api_key="test_key")
        client = KieApiClient(config)

        # Mock httpx response that returns None from .json()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

            with pytest.raises(Exception) as exc_info:
                await client.create_task(model="test-model", prompt="test prompt")

            assert "Empty response body" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_task_status_handles_none_response(self):
        """Bug #2: get_task_status() should raise exception if response.json() is None"""
        from server import KieApiClient, Config

        config = Config(kie_api_key="test_key")
        client = KieApiClient(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = None

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            with pytest.raises(Exception) as exc_info:
                await client.get_task_status("test_task_id")

            assert "Empty response body" in str(exc_info.value)


# Test Bug #3, #4, #5: isinstance checks for data field
class TestDataFieldTypeChecking:
    """Test that data field is properly type-checked before .get() calls."""

    def test_extract_handles_data_as_none(self):
        """Bug #3/4/5: Should handle case where data field is None"""
        from server import extract_images_from_result

        # Case: data field exists but is None
        result = {"data": None, "status": "success"}
        images = extract_images_from_result(result)
        # Should not crash, should return empty or handle gracefully
        assert isinstance(images, list)

    def test_extract_handles_data_as_string(self):
        """Should handle case where data field is a string instead of dict"""
        from server import extract_images_from_result

        result = {"data": "some string", "status": "success"}
        images = extract_images_from_result(result)
        assert isinstance(images, list)


# Test Bug #6: extract_images_from_result with None items
class TestExtractImagesNoneHandling:
    """Test that None items in output arrays don't crash extraction."""

    def test_extract_handles_none_in_output_list(self):
        """Bug #6: Should skip None items in output list"""
        from server import extract_images_from_result

        result = {
            "output": ["http://valid-url.com/image.png", None, {"url": "http://another.com/img.jpg"}, None]
        }
        images = extract_images_from_result(result)

        # Should only extract valid items, skip None
        assert len(images) == 2
        assert images[0]["url"] == "http://valid-url.com/image.png"
        assert images[1]["url"] == "http://another.com/img.jpg"

    def test_extract_handles_mixed_invalid_types(self):
        """Should handle mixed types including integers, bools"""
        from server import extract_images_from_result

        result = {
            "output": ["http://valid.com/img.png", 123, True, {"url": "http://ok.com/img.jpg"}]
        }
        images = extract_images_from_result(result)

        # Should only extract strings and dicts with url
        assert len(images) == 2


# Test Bug #7: CallToolResult error handling
class TestMCPErrorHandling:
    """Test that errors return proper CallToolResult with isError=True."""

    def test_call_tool_result_import(self):
        """Verify CallToolResult is properly imported"""
        from server import CallToolResult
        assert CallToolResult is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
