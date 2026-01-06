"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_user_message():
    """Sample user message context."""
    from app.services.context import MessageContext
    return MessageContext(role="user", content="Hello, how are you?")


@pytest.fixture
def sample_assistant_message():
    """Sample assistant message context."""
    from app.services.context import MessageContext
    return MessageContext(role="assistant", content="I'm doing well, thanks for asking!")


@pytest.fixture
def sample_branch_origin():
    """Sample branch origin context."""
    from app.services.context import BranchOriginContext
    return BranchOriginContext(
        highlighted_text="doing well",
        user_prompt="What do you mean by this?",
    )
