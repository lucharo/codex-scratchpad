"""Tests for context service.

Tests the pure business logic of building conversation context.
No mocking needed - these are pure functions operating on dataclasses.
"""

from app.services.context import (
    BranchOriginContext,
    MessageContext,
    build_branch_context,
    build_linear_context,
    build_prompt_from_context,
)


class TestBuildPromptFromContext:
    """Tests for build_prompt_from_context function.

    Execution branches:
    1. Empty message list
    2. User message formatting
    3. Assistant message formatting
    4. System message formatting (passthrough)
    5. Mixed messages in sequence
    """

    def test_returns_empty_string_when_messages_list_is_empty(self):
        # Arrange
        messages = []

        # Act
        result = build_prompt_from_context(messages)

        # Assert
        assert result == ""

    def test_formats_user_message_with_user_prefix(self):
        # Arrange
        messages = [MessageContext(role="user", content="Hello world")]

        # Act
        result = build_prompt_from_context(messages)

        # Assert
        assert result == "User: Hello world"

    def test_formats_assistant_message_with_assistant_prefix(self):
        # Arrange
        messages = [MessageContext(role="assistant", content="Hi there")]

        # Act
        result = build_prompt_from_context(messages)

        # Assert
        assert result == "Assistant: Hi there"

    def test_passes_system_message_content_through_without_prefix(self):
        # Arrange
        messages = [MessageContext(role="system", content="[BRANCH POINT]")]

        # Act
        result = build_prompt_from_context(messages)

        # Assert
        assert result == "[BRANCH POINT]"

    def test_joins_multiple_messages_with_double_newlines(self):
        # Arrange
        messages = [
            MessageContext(role="user", content="Question"),
            MessageContext(role="assistant", content="Answer"),
            MessageContext(role="user", content="Follow-up"),
        ]

        # Act
        result = build_prompt_from_context(messages)

        # Assert
        expected = "User: Question\n\nAssistant: Answer\n\nUser: Follow-up"
        assert result == expected


class TestBuildBranchContext:
    """Tests for build_branch_context function.

    Execution branches:
    1. Empty parent messages, empty current messages
    2. Parent messages only (no current messages)
    3. Current messages only (no parent messages)
    4. Both parent and current messages
    """

    def test_returns_only_branch_marker_when_both_parent_and_current_are_empty(self):
        # Arrange
        parent_messages = []
        origin = BranchOriginContext(
            highlighted_text="some text",
            user_prompt="explain this",
        )
        current_messages = []

        # Act
        result = build_branch_context(parent_messages, origin, current_messages)

        # Assert
        assert len(result) == 1
        assert result[0].role == "system"
        assert '"some text"' in result[0].content
        assert "explain this" in result[0].content

    def test_includes_parent_messages_before_branch_marker(self):
        # Arrange
        parent_messages = [
            MessageContext(role="user", content="Parent question"),
            MessageContext(role="assistant", content="Parent answer"),
        ]
        origin = BranchOriginContext(
            highlighted_text="answer",
            user_prompt="what?",
        )
        current_messages = []

        # Act
        result = build_branch_context(parent_messages, origin, current_messages)

        # Assert
        assert len(result) == 3
        assert result[0] == MessageContext(role="user", content="Parent question")
        assert result[1] == MessageContext(role="assistant", content="Parent answer")
        assert result[2].role == "system"

    def test_includes_current_messages_after_branch_marker(self):
        # Arrange
        parent_messages = []
        origin = BranchOriginContext(
            highlighted_text="text",
            user_prompt="why?",
        )
        current_messages = [
            MessageContext(role="user", content="Branch question"),
            MessageContext(role="assistant", content="Branch answer"),
        ]

        # Act
        result = build_branch_context(parent_messages, origin, current_messages)

        # Assert
        assert len(result) == 3
        assert result[0].role == "system"
        assert result[1] == MessageContext(role="user", content="Branch question")
        assert result[2] == MessageContext(role="assistant", content="Branch answer")

    def test_preserves_order_parent_then_marker_then_current(self):
        # Arrange
        parent_messages = [MessageContext(role="user", content="P1")]
        origin = BranchOriginContext(highlighted_text="hl", user_prompt="up")
        current_messages = [MessageContext(role="user", content="C1")]

        # Act
        result = build_branch_context(parent_messages, origin, current_messages)

        # Assert
        assert len(result) == 3
        assert result[0].content == "P1"
        assert result[1].role == "system"
        assert result[2].content == "C1"


class TestBuildLinearContext:
    """Tests for build_linear_context function.

    Execution branches:
    1. Empty list
    2. Non-empty list (returns copy)
    """

    def test_returns_empty_list_when_input_is_empty(self):
        # Arrange
        messages = []

        # Act
        result = build_linear_context(messages)

        # Assert
        assert result == []

    def test_returns_copy_of_input_messages(self):
        # Arrange
        original = [MessageContext(role="user", content="test")]

        # Act
        result = build_linear_context(original)

        # Assert
        assert result == original
        assert result is not original  # Verify it's a copy
