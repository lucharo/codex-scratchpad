"""Claude Agent SDK integration service for Tree of Knowledge.

This service handles all interactions with Claude via the Agent SDK,
including streaming responses, tool calls, and thinking blocks.
"""

import json
from typing import AsyncGenerator, Optional
from dataclasses import dataclass

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
)


@dataclass
class StreamEvent:
    """Events emitted during streaming."""

    type: str  # "text", "thinking", "tool_use", "tool_result", "complete", "error"
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_result: Optional[str] = None
    is_error: bool = False
    session_id: Optional[str] = None
    model: Optional[str] = None

    def to_sse(self) -> str:
        """Convert to Server-Sent Event format."""
        data = {
            "type": self.type,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_result": self.tool_result,
            "is_error": self.is_error,
            "session_id": self.session_id,
            "model": self.model,
        }
        # Remove None values for cleaner JSON
        data = {k: v for k, v in data.items() if v is not None}
        return f"data: {json.dumps(data)}\n\n"


class ClaudeService:
    """Service for interacting with Claude via the Agent SDK."""

    DEFAULT_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"]

    def __init__(
        self,
        allowed_tools: Optional[list[str]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.allowed_tools = allowed_tools or self.DEFAULT_TOOLS
        self.system_prompt = system_prompt

    def _build_options(
        self,
        session_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for a request."""
        options = ClaudeAgentOptions(
            allowed_tools=self.allowed_tools,
            permission_mode="acceptEdits",
        )

        if self.system_prompt:
            options.system_prompt = self.system_prompt

        if session_id:
            options.resume = session_id

        if cwd:
            options.cwd = cwd

        return options

    async def stream_response(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response from Claude.

        Args:
            prompt: The formatted prompt string
            session_id: Optional session ID to resume
            cwd: Working directory for file operations

        Yields:
            StreamEvent objects for each piece of the response
        """
        options = self._build_options(session_id=session_id, cwd=cwd)

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.connect()
                await client.query(prompt)

                current_model: Optional[str] = None

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        current_model = message.model

                        for block in message.content:
                            if isinstance(block, ThinkingBlock):
                                yield StreamEvent(
                                    type="thinking",
                                    content=block.thinking,
                                    model=current_model,
                                )
                            elif isinstance(block, TextBlock):
                                yield StreamEvent(
                                    type="text",
                                    content=block.text,
                                    model=current_model,
                                )
                            elif isinstance(block, ToolUseBlock):
                                yield StreamEvent(
                                    type="tool_use",
                                    tool_name=block.name,
                                    tool_input=block.input,
                                    model=current_model,
                                )
                            elif isinstance(block, ToolResultBlock):
                                yield StreamEvent(
                                    type="tool_result",
                                    tool_result=block.content if isinstance(block.content, str) else str(block.content),
                                    is_error=block.is_error or False,
                                )

                    elif isinstance(message, ResultMessage):
                        yield StreamEvent(
                            type="complete",
                            session_id=message.session_id,
                        )

        except Exception as e:
            yield StreamEvent(
                type="error",
                content=str(e),
                is_error=True,
            )


# Default service instance
claude_service = ClaudeService()
