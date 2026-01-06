"""Conversation context service for Tree of Knowledge.

Handles building conversation context for Claude, including branch context.
This is pure business logic - no HTTP or database ORM concerns.
"""

from dataclasses import dataclass


@dataclass
class MessageContext:
    """A message in conversation context."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class BranchOriginContext:
    """Context about how a branch was created."""
    highlighted_text: str
    user_prompt: str


def build_prompt_from_context(messages: list[MessageContext]) -> str:
    """Convert message context list to a prompt string for Claude.

    Args:
        messages: List of messages in conversation order

    Returns:
        Formatted prompt string
    """
    parts = []
    for msg in messages:
        if msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
        else:
            # System messages (like branch points) go through as-is
            parts.append(msg.content)
    return "\n\n".join(parts)


def build_branch_context(
    parent_messages: list[MessageContext],
    origin: BranchOriginContext,
    current_messages: list[MessageContext],
) -> list[MessageContext]:
    """Build full context for a branched conversation.

    Args:
        parent_messages: Messages from parent branch up to branch point
        origin: The branch origin (highlighted text + user prompt)
        current_messages: Messages in the current branch

    Returns:
        Complete message context list
    """
    context = list(parent_messages)

    # Add branch point marker
    context.append(MessageContext(
        role="system",
        content=(
            f"[BRANCH POINT: User highlighted the following text]\n\n"
            f'"{origin.highlighted_text}"\n\n'
            f"[User's question about this text]: {origin.user_prompt}"
        ),
    ))

    # Add current branch messages
    context.extend(current_messages)

    return context


def build_linear_context(messages: list[MessageContext]) -> list[MessageContext]:
    """Build context for a linear (non-branched) conversation.

    Args:
        messages: All messages in the conversation

    Returns:
        Message context list (just returns input, but provides consistent interface)
    """
    return list(messages)
