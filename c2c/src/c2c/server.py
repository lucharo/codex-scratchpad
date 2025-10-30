"""MCP server implementation for c2c."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .models import SessionConfig
from .permissions import (
    PermissionAction,
    PermissionDecision,
    PermissionManager,
    PermissionRequest,
    RiskLevel,
)
from .session import SessionManager, SessionError


# Initialize the MCP server
app = Server("c2c")

# Global managers (initialized on startup)
session_manager: SessionManager = None
permission_manager: PermissionManager = None


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
        Tool(
            name="request_permission",
            description=(
                "Request permission for an action from a sub-agent. "
                "The main agent will auto-approve, auto-deny, or escalate to user based on policies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session requesting permission",
                    },
                    "action": {
                        "type": "string",
                        "description": "Type of action (execute_command, read_file, write_file, etc.)",
                        "enum": [a.value for a in PermissionAction],
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what you want to do",
                    },
                    "details": {
                        "type": "object",
                        "description": "Action-specific details (e.g., command, file path)",
                    },
                },
                "required": ["session_id", "action", "description"],
            },
        ),
        Tool(
            name="get_pending_permissions",
            description="Get all pending permission requests that need user review.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Optional: filter by session ID",
                    },
                },
            },
        ),
        Tool(
            name="approve_permission",
            description="Approve a permission request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "Permission request ID to approve",
                    },
                },
                "required": ["request_id"],
            },
        ),
        Tool(
            name="deny_permission",
            description="Deny a permission request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "Permission request ID to deny",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for denial",
                    },
                },
                "required": ["request_id"],
            },
        ),
        Tool(
            name="get_permission_status",
            description="Get the status of a permission request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "Permission request ID",
                    },
                },
                "required": ["request_id"],
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
        elif name == "request_permission":
            return await _request_permission(arguments)
        elif name == "get_pending_permissions":
            return await _get_pending_permissions(arguments)
        elif name == "approve_permission":
            return await _approve_permission(arguments)
        elif name == "deny_permission":
            return await _deny_permission(arguments)
        elif name == "get_permission_status":
            return await _get_permission_status(arguments)
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
        task_preview = s.task if len(s.task) <= 50 else f"{s.task[:50]}..."
        lines.append(
            f"- {s.session_id}: {task_preview} ({s.status})"
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


async def _request_permission(arguments: dict) -> list[TextContent]:
    """Request permission for an action."""
    request_id = f"perm-{uuid.uuid4().hex[:12]}"
    request = PermissionRequest(
        request_id=request_id,
        session_id=arguments["session_id"],
        action=PermissionAction(arguments["action"]),
        description=arguments["description"],
        details=arguments.get("details", {}),
    )

    # Process through permission manager
    request = permission_manager.request_permission(request)

    # Build response based on decision
    if request.decision == PermissionDecision.APPROVED:
        return [
            TextContent(
                type="text",
                text=f"✓ Permission APPROVED\n\n"
                f"Request ID: {request.request_id}\n"
                f"Action: {request.action}\n"
                f"Description: {request.description}\n"
                f"Decided by: {request.decided_by}\n\n"
                f"You may proceed with this action.",
            )
        ]
    elif request.decision == PermissionDecision.DENIED:
        return [
            TextContent(
                type="text",
                text=f"✗ Permission DENIED\n\n"
                f"Request ID: {request.request_id}\n"
                f"Action: {request.action}\n"
                f"Description: {request.description}\n"
                f"Reason: {request.denial_reason}\n"
                f"Decided by: {request.decided_by}\n\n"
                f"This action cannot be performed.",
            )
        ]
    elif request.decision == PermissionDecision.ESCALATED:
        return [
            TextContent(
                type="text",
                text=f"⚠ Permission ESCALATED to user review\n\n"
                f"Request ID: {request.request_id}\n"
                f"Action: {request.action}\n"
                f"Description: {request.description}\n"
                f"Risk Level: {request.risk_level}\n\n"
                f"This request requires user approval. The main agent will "
                f"ask the user to review this action.\n\n"
                f"Please wait for approval before proceeding.",
            )
        ]
    else:  # PENDING
        return [
            TextContent(
                type="text",
                text=f"⏳ Permission PENDING review\n\n"
                f"Request ID: {request.request_id}\n"
                f"Action: {request.action}\n"
                f"Description: {request.description}\n"
                f"Risk Level: {request.risk_level}\n\n"
                f"Please wait for a decision.",
            )
        ]


async def _get_pending_permissions(arguments: dict) -> list[TextContent]:
    """Get pending permission requests."""
    session_id = arguments.get("session_id")
    pending = permission_manager.get_pending_requests(session_id)

    if not pending:
        return [
            TextContent(
                type="text",
                text="No pending permission requests.",
            )
        ]

    lines = ["Pending Permission Requests:\n"]
    for req in pending:
        lines.append(
            f"\n- Request ID: {req.request_id}\n"
            f"  Session: {req.session_id}\n"
            f"  Action: {req.action}\n"
            f"  Description: {req.description}\n"
            f"  Risk: {req.risk_level}\n"
            f"  Status: {req.decision}\n"
            f"  Created: {req.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    return [
        TextContent(
            type="text",
            text="\n".join(lines),
        )
    ]


async def _approve_permission(arguments: dict) -> list[TextContent]:
    """Approve a permission request."""
    request_id = arguments["request_id"]

    try:
        request = permission_manager.make_decision(
            request_id,
            PermissionDecision.APPROVED,
            decided_by="user",
        )

        return [
            TextContent(
                type="text",
                text=f"✓ Permission request {request_id} APPROVED\n\n"
                f"Action: {request.action}\n"
                f"Description: {request.description}\n\n"
                f"The sub-agent may now proceed with this action.",
            )
        ]
    except KeyError:
        return [
            TextContent(
                type="text",
                text=f"Error: Permission request {request_id} not found.",
            )
        ]


async def _deny_permission(arguments: dict) -> list[TextContent]:
    """Deny a permission request."""
    request_id = arguments["request_id"]
    reason = arguments.get("reason", "Denied by user")

    try:
        request = permission_manager.make_decision(
            request_id,
            PermissionDecision.DENIED,
            decided_by="user",
            reason=reason,
        )

        return [
            TextContent(
                type="text",
                text=f"✗ Permission request {request_id} DENIED\n\n"
                f"Action: {request.action}\n"
                f"Description: {request.description}\n"
                f"Reason: {reason}\n\n"
                f"The sub-agent will be informed of the denial.",
            )
        ]
    except KeyError:
        return [
            TextContent(
                type="text",
                text=f"Error: Permission request {request_id} not found.",
            )
        ]


async def _get_permission_status(arguments: dict) -> list[TextContent]:
    """Get the status of a permission request."""
    request_id = arguments["request_id"]
    request = permission_manager.get_request(request_id)

    if not request:
        return [
            TextContent(
                type="text",
                text=f"Permission request {request_id} not found.",
            )
        ]

    status_symbols = {
        PermissionDecision.APPROVED: "✓",
        PermissionDecision.DENIED: "✗",
        PermissionDecision.PENDING: "⏳",
        PermissionDecision.ESCALATED: "⚠",
    }

    symbol = status_symbols.get(request.decision, "?")

    return [
        TextContent(
            type="text",
            text=f"{symbol} Permission Request Status\n\n"
            f"Request ID: {request.request_id}\n"
            f"Session: {request.session_id}\n"
            f"Action: {request.action}\n"
            f"Description: {request.description}\n"
            f"Risk Level: {request.risk_level}\n"
            f"Status: {request.decision}\n"
            f"Created: {request.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Decided: {request.decided_at.strftime('%Y-%m-%d %H:%M:%S') if request.decided_at else 'N/A'}\n"
            f"Decided by: {request.decided_by or 'N/A'}\n"
            f"Denial reason: {request.denial_reason or 'N/A'}\n",
        )
    ]


async def main(repo_root: Path | str = None):
    """Run the MCP server."""
    global session_manager, permission_manager

    # Initialize managers
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    session_manager = SessionManager(repo_root)
    permission_manager = PermissionManager()

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
