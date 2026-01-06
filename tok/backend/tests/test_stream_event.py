"""Tests for StreamEvent dataclass.

Tests the SSE serialization logic.
"""

import json
import pytest
from app.services.claude_service import StreamEvent


class TestStreamEventToSse:
    """Tests for StreamEvent.to_sse method.

    Execution branches:
    1. Minimal event (only type)
    2. Event with content
    3. Event with tool fields
    4. None values are excluded from output
    """

    def test_serializes_minimal_event_with_only_type(self):
        # Arrange
        event = StreamEvent(type="complete")

        # Act
        result = event.to_sse()

        # Assert
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        payload = json.loads(result[6:-2])  # Strip "data: " and "\n\n"
        assert payload == {"type": "complete"}

    def test_serializes_text_event_with_content(self):
        # Arrange
        event = StreamEvent(type="text", content="Hello world", model="claude-3")

        # Act
        result = event.to_sse()

        # Assert
        payload = json.loads(result[6:-2])
        assert payload == {
            "type": "text",
            "content": "Hello world",
            "model": "claude-3",
        }

    def test_serializes_tool_use_event_with_tool_fields(self):
        # Arrange
        event = StreamEvent(
            type="tool_use",
            tool_name="Bash",
            tool_input={"command": "ls -la"},
        )

        # Act
        result = event.to_sse()

        # Assert
        payload = json.loads(result[6:-2])
        assert payload == {
            "type": "tool_use",
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
        }

    def test_excludes_none_values_from_serialized_output(self):
        # Arrange
        event = StreamEvent(
            type="error",
            content="Something failed",
            is_error=True,
            # All other fields are None by default
        )

        # Act
        result = event.to_sse()

        # Assert
        payload = json.loads(result[6:-2])
        assert payload == {
            "type": "error",
            "content": "Something failed",
            "is_error": True,
        }
        assert "tool_name" not in payload
        assert "session_id" not in payload

    def test_includes_false_boolean_in_output(self):
        # Arrange
        event = StreamEvent(type="tool_result", tool_result="success", is_error=False)

        # Act
        result = event.to_sse()

        # Assert
        payload = json.loads(result[6:-2])
        # is_error=False should be included (it's not None)
        assert payload == {
            "type": "tool_result",
            "tool_result": "success",
            "is_error": False,
        }
