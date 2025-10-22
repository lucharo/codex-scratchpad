"""MCP server implementation for cc2cc."""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .models import SessionConfig
from .session import SessionManager, SessionError


# Initialize the MCP server
app = Server("cc2cc")

# Global session manager (initialized on startup)
session_manager: SessionManager = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="create_session",
            description=(
                "Create a new Claude Code session. The session will be created "
                "in an isolated git worktree with its own branch."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task description for the Claude Code session",
                    },
                    "branch_name": {
                        "type": "string",
                        "description": "Optional custom branch name (auto-generated if not provided)",
                    },
                    "use_worktree": {
                        "type": "boolean",
                        "description": "Whether to use git worktree for isolation (default: true)",
                        "default": True,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds",
                    },
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="start_session",
            description="Start a created Claude Code session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to start",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="get_session",
            description="Get detailed information about a session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to query",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="list_sessions",
            description="List all Claude Code sessions.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_session_output",
            description="Get output from a Claude Code session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to get output from",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="terminate_session",
            description="Terminate a running Claude Code session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to terminate",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force termination (kill instead of graceful shutdown)",
                        "default": False,
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="cleanup_session",
            description="Clean up a session and its resources (worktree, branch).",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to clean up",
                    },
                    "remove_branch": {
                        "type": "boolean",
                        "description": "Whether to delete the git branch (default: false)",
                        "default": False,
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "create_session":
            return await _create_session(arguments)
        elif name == "start_session":
            return await _start_session(arguments)
        elif name == "get_session":
            return await _get_session(arguments)
        elif name == "list_sessions":
            return await _list_sessions(arguments)
        elif name == "get_session_output":
            return await _get_session_output(arguments)
        elif name == "terminate_session":
            return await _terminate_session(arguments)
        elif name == "cleanup_session":
            return await _cleanup_session(arguments)
        else:
            return [
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Error: {str(e)}",
            )
        ]


async def _create_session(arguments: dict) -> list[TextContent]:
    """Create a new session."""
    config = SessionConfig(
        task=arguments["task"],
        branch_name=arguments.get("branch_name"),
        use_worktree=arguments.get("use_worktree", True),
        timeout=arguments.get("timeout"),
    )

    session = session_manager.create_session(config)

    return [
        TextContent(
            type="text",
            text=f"Session created successfully!\n\n"
            f"Session ID: {session.session_id}\n"
            f"Branch: {session.branch_name}\n"
            f"Worktree: {session.worktree_path or 'N/A'}\n"
            f"Status: {session.status}\n\n"
            f"Use 'start_session' with session_id '{session.session_id}' to start the session.",
        )
    ]


async def _start_session(arguments: dict) -> list[TextContent]:
    """Start a session."""
    session_id = arguments["session_id"]
    await session_manager.start_session(session_id)

    return [
        TextContent(
            type="text",
            text=f"Session {session_id} started successfully!",
        )
    ]


async def _get_session(arguments: dict) -> list[TextContent]:
    """Get session details."""
    session_id = arguments["session_id"]
    session = session_manager.get_session(session_id)

    if not session:
        raise SessionError(f"Session not found: {session_id}")

    return [
        TextContent(
            type="text",
            text=f"Session Details:\n\n"
            f"ID: {session.session_id}\n"
            f"Task: {session.config.task}\n"
            f"Status: {session.status}\n"
            f"Branch: {session.branch_name}\n"
            f"Worktree: {session.worktree_path or 'N/A'}\n"
            f"Process ID: {session.process_id or 'N/A'}\n"
            f"Created: {session.created_at.isoformat()}\n"
            f"Started: {session.started_at.isoformat() if session.started_at else 'N/A'}\n"
            f"Completed: {session.completed_at.isoformat() if session.completed_at else 'N/A'}\n"
            f"Error: {session.error_message or 'N/A'}\n",
        )
    ]


async def _list_sessions(arguments: dict) -> list[TextContent]:
    """List all sessions."""
    sessions = session_manager.list_sessions()

    if not sessions:
        return [
            TextContent(
                type="text",
                text="No sessions found.",
            )
        ]

    lines = ["Sessions:\n"]
    for s in sessions:
        lines.append(
            f"- {s.session_id}: {s.task[:50]}... ({s.status})"
        )

    return [
        TextContent(
            type="text",
            text="\n".join(lines),
        )
    ]


async def _get_session_output(arguments: dict) -> list[TextContent]:
    """Get session output."""
    session_id = arguments["session_id"]
    output = await session_manager.get_session_output(session_id)

    if not output:
        return [
            TextContent(
                type="text",
                text=f"No output available for session {session_id}",
            )
        ]

    return [
        TextContent(
            type="text",
            text=f"Session {session_id} output:\n\n" + "\n".join(output),
        )
    ]


async def _terminate_session(arguments: dict) -> list[TextContent]:
    """Terminate a session."""
    session_id = arguments["session_id"]
    force = arguments.get("force", False)

    await session_manager.terminate_session(session_id, force=force)

    return [
        TextContent(
            type="text",
            text=f"Session {session_id} terminated.",
        )
    ]


async def _cleanup_session(arguments: dict) -> list[TextContent]:
    """Clean up a session."""
    session_id = arguments["session_id"]
    remove_branch = arguments.get("remove_branch", False)

    await session_manager.cleanup_session(
        session_id, remove_branch=remove_branch
    )

    return [
        TextContent(
            type="text",
            text=f"Session {session_id} cleaned up.",
        )
    ]


async def main(repo_root: Path | str = None):
    """Run the MCP server."""
    global session_manager

    # Initialize session manager
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    session_manager = SessionManager(repo_root)

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
