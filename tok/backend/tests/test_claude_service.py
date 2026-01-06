"""Tests for ClaudeService.

External dependency (ClaudeSDKClient) is mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.claude_service import ClaudeService


def setup_mock_claude_client(messages_to_yield):
    """Setup mock ClaudeSDKClient that yields specified messages.

    Args:
        messages_to_yield: List of mock message objects to yield from receive_response
    """
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.query = AsyncMock()

    async def mock_receive():
        for msg in messages_to_yield:
            yield msg

    mock_client.receive_response = mock_receive
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client


class TestClaudeServiceInit:
    """Tests for ClaudeService initialization.

    Execution branches:
    1. Default tools when none provided
    2. Custom tools when provided
    """

    def test_uses_default_tools_when_none_provided(self):
        # Arrange & Act
        service = ClaudeService()

        # Assert
        assert service.allowed_tools == ClaudeService.DEFAULT_TOOLS

    def test_uses_provided_tools_when_specified(self):
        # Arrange
        custom_tools = ["Read", "Bash"]

        # Act
        service = ClaudeService(allowed_tools=custom_tools)

        # Assert
        assert service.allowed_tools == ["Read", "Bash"]


class TestClaudeServiceStreamResponse:
    """Tests for ClaudeService.stream_response method.

    Execution branches:
    1. Yields error event when exception occurs
    """

    @pytest.mark.asyncio
    async def test_yields_error_event_when_client_raises_exception(self):
        # Arrange
        service = ClaudeService()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Act
        events = []
        with patch("app.services.claude_service.ClaudeSDKClient", return_value=mock_client):
            async for event in service.stream_response("test"):
                events.append(event)

        # Assert
        assert len(events) == 1
        assert events[0].type == "error"
        assert events[0].content == "Connection failed"
        assert events[0].is_error is True


class TestClaudeServiceBuildOptions:
    """Tests for ClaudeService._build_options method.

    Execution branches:
    1. No optional params
    2. With session_id (resume)
    3. With cwd
    4. With system_prompt
    """

    def test_builds_options_with_default_tools_and_permission_mode(self):
        # Arrange
        service = ClaudeService()

        # Act
        with patch("app.services.claude_service.ClaudeAgentOptions") as MockOptions:
            MockOptions.return_value = MagicMock()
            service._build_options()

        # Assert
        MockOptions.assert_called_once()
        call_kwargs = MockOptions.call_args[1]
        assert call_kwargs["allowed_tools"] == ClaudeService.DEFAULT_TOOLS
        assert call_kwargs["permission_mode"] == "acceptEdits"

    def test_sets_resume_when_session_id_provided(self):
        # Arrange
        service = ClaudeService()

        # Act
        with patch("app.services.claude_service.ClaudeAgentOptions") as MockOptions:
            mock_options = MagicMock()
            MockOptions.return_value = mock_options
            service._build_options(session_id="session-123")

        # Assert
        assert mock_options.resume == "session-123"

    def test_sets_cwd_when_provided(self):
        # Arrange
        service = ClaudeService()

        # Act
        with patch("app.services.claude_service.ClaudeAgentOptions") as MockOptions:
            mock_options = MagicMock()
            MockOptions.return_value = mock_options
            service._build_options(cwd="/tmp/work")

        # Assert
        assert mock_options.cwd == "/tmp/work"

    def test_sets_system_prompt_when_configured(self):
        # Arrange
        service = ClaudeService(system_prompt="Be helpful")

        # Act
        with patch("app.services.claude_service.ClaudeAgentOptions") as MockOptions:
            mock_options = MagicMock()
            MockOptions.return_value = mock_options
            service._build_options()

        # Assert
        assert mock_options.system_prompt == "Be helpful"
