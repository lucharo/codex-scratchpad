"""MCP server implementation for c2c using FastMCP."""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .models import SessionConfig
from .permissions import (
    PermissionAction,
    PermissionDecision,
    PermissionManager,
    PermissionRequest,
)
from .session import SessionManager, SessionError


# Initialize FastMCP server
mcp = FastMCP("c2c")

# Global managers (initialized on startup)
session_manager: SessionManager = None
permission_manager: PermissionManager = None


@mcp.tool()
async def create_session(
    task: str,
    branch_name: Optional[str] = None,
    use_worktree: bool = True,
    timeout: Optional[int] = None,
    parent_session_id: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    metadata: Optional[dict[str, str]] = None,
) -> str:
    """Create a new Claude Code session.

    The session will be created in an isolated git worktree with its own branch.
    Supports hierarchical organization with parent-child relationships and tags.

    Args:
        task: Task description for the Claude Code session
        branch_name: Optional custom branch name (auto-generated if not provided)
        use_worktree: Whether to use git worktree for isolation (default: true)
        timeout: Optional timeout in seconds
        parent_session_id: Optional parent session ID for hierarchical organization
        tags: Tags for organizing sessions (e.g., {"feature": "AddAuth", "role": "reviewer", "strategy": "OptionA"})
        metadata: Additional metadata for custom use cases

    Returns:
        Success message with session details
    """
    config = SessionConfig(
        task=task,
        branch_name=branch_name,
        use_worktree=use_worktree,
        timeout=timeout,
        tags=tags or {},
        metadata=metadata or {},
    )

    session = session_manager.create_session(config, parent_session_id=parent_session_id)

    # Build response with hierarchy info
    hierarchy_info = ""
    if session.parent_session_id:
        hierarchy_info = f"Parent: {session.parent_session_id}\nDepth: {session.depth}\n"

    tags_info = ""
    if session.config.tags:
        tags_info = f"Tags: {session.config.tags}\n"

    return (
        f"Session created successfully!\n\n"
        f"Session ID: {session.session_id}\n"
        f"Branch: {session.branch_name}\n"
        f"Worktree: {session.worktree_path or 'N/A'}\n"
        f"Status: {session.status}\n"
        f"{hierarchy_info}"
        f"{tags_info}\n"
        f"Use 'start_session' with session_id '{session.session_id}' to start the session."
    )


@mcp.tool()
async def start_session(session_id: str) -> str:
    """Start a created Claude Code session.

    Args:
        session_id: Session ID to start

    Returns:
        Success message
    """
    await session_manager.start_session(session_id)
    return f"Session {session_id} started successfully!"


@mcp.tool()
async def get_session(session_id: str) -> str:
    """Get detailed information about a session.

    Args:
        session_id: Session ID to query

    Returns:
        Detailed session information
    """
    session = session_manager.get_session(session_id)

    if not session:
        raise SessionError(f"Session not found: {session_id}")

    return (
        f"Session Details:\n\n"
        f"ID: {session.session_id}\n"
        f"Task: {session.config.task}\n"
        f"Status: {session.status}\n"
        f"Branch: {session.branch_name}\n"
        f"Worktree: {session.worktree_path or 'N/A'}\n"
        f"Process ID: {session.process_id or 'N/A'}\n"
        f"Created: {session.created_at.isoformat()}\n"
        f"Started: {session.started_at.isoformat() if session.started_at else 'N/A'}\n"
        f"Completed: {session.completed_at.isoformat() if session.completed_at else 'N/A'}\n"
        f"Error: {session.error_message or 'N/A'}\n"
    )


@mcp.tool()
async def list_sessions() -> str:
    """List all Claude Code sessions.

    Returns:
        List of all sessions with their status
    """
    sessions = session_manager.list_sessions()

    if not sessions:
        return "No sessions found."

    lines = ["Sessions:\n"]
    for s in sessions:
        task_preview = s.task if len(s.task) <= 50 else f"{s.task[:50]}..."
        lines.append(f"- {s.session_id}: {task_preview} ({s.status})")

    return "\n".join(lines)


@mcp.tool()
async def get_session_output(session_id: str) -> str:
    """Get output from a Claude Code session.

    Args:
        session_id: Session ID to get output from

    Returns:
        Session output
    """
    output = await session_manager.get_session_output(session_id)

    if not output:
        return f"No output available for session {session_id}"

    return f"Session {session_id} output:\n\n" + "\n".join(output)


@mcp.tool()
async def terminate_session(session_id: str, force: bool = False) -> str:
    """Terminate a running Claude Code session.

    Args:
        session_id: Session ID to terminate
        force: Force termination (kill instead of graceful shutdown)

    Returns:
        Success message
    """
    await session_manager.terminate_session(session_id, force=force)
    return f"Session {session_id} terminated."


@mcp.tool()
async def cleanup_session(session_id: str, remove_branch: bool = False) -> str:
    """Clean up a session and its resources (worktree, branch).

    Args:
        session_id: Session ID to clean up
        remove_branch: Whether to delete the git branch (default: false)

    Returns:
        Success message
    """
    await session_manager.cleanup_session(session_id, remove_branch=remove_branch)
    return f"Session {session_id} cleaned up."


@mcp.tool()
async def request_permission(
    session_id: str,
    action: str,
    description: str,
    details: Optional[dict] = None,
) -> str:
    """Request permission for an action from a sub-agent.

    The main agent will auto-approve, auto-deny, or escalate to user based on policies.

    Args:
        session_id: Session requesting permission
        action: Type of action (execute_command, read_file, write_file, etc.)
        description: Human-readable description of what you want to do
        details: Action-specific details (e.g., command, file path)

    Returns:
        Permission decision with details
    """
    request_id = f"perm-{uuid.uuid4().hex[:12]}"
    request = PermissionRequest(
        request_id=request_id,
        session_id=session_id,
        action=PermissionAction(action),
        description=description,
        details=details or {},
    )

    # Process through permission manager
    request = permission_manager.request_permission(request)

    # Build response based on decision
    if request.decision == PermissionDecision.APPROVED:
        return (
            f"✓ Permission APPROVED\n\n"
            f"Request ID: {request.request_id}\n"
            f"Action: {request.action}\n"
            f"Description: {request.description}\n"
            f"Decided by: {request.decided_by}\n\n"
            f"You may proceed with this action."
        )
    elif request.decision == PermissionDecision.DENIED:
        return (
            f"✗ Permission DENIED\n\n"
            f"Request ID: {request.request_id}\n"
            f"Action: {request.action}\n"
            f"Description: {request.description}\n"
            f"Reason: {request.denial_reason}\n"
            f"Decided by: {request.decided_by}\n\n"
            f"This action cannot be performed."
        )
    elif request.decision == PermissionDecision.ESCALATED:
        return (
            f"⚠ Permission ESCALATED to user review\n\n"
            f"Request ID: {request.request_id}\n"
            f"Action: {request.action}\n"
            f"Description: {request.description}\n"
            f"Risk Level: {request.risk_level}\n\n"
            f"This request requires user approval. The main agent will "
            f"ask the user to review this action.\n\n"
            f"Please wait for approval before proceeding."
        )
    else:  # PENDING
        return (
            f"⏳ Permission PENDING review\n\n"
            f"Request ID: {request.request_id}\n"
            f"Action: {request.action}\n"
            f"Description: {request.description}\n"
            f"Risk Level: {request.risk_level}\n\n"
            f"Please wait for a decision."
        )


@mcp.tool()
async def get_pending_permissions(session_id: Optional[str] = None) -> str:
    """Get all pending permission requests that need user review.

    Args:
        session_id: Optional filter by session ID

    Returns:
        List of pending permission requests
    """
    pending = permission_manager.get_pending_requests(session_id)

    if not pending:
        return "No pending permission requests."

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

    return "\n".join(lines)


@mcp.tool()
async def approve_permission(request_id: str) -> str:
    """Approve a permission request.

    Args:
        request_id: Permission request ID to approve

    Returns:
        Success message
    """
    try:
        request = permission_manager.make_decision(
            request_id,
            PermissionDecision.APPROVED,
            decided_by="user",
        )

        return (
            f"✓ Permission request {request_id} APPROVED\n\n"
            f"Action: {request.action}\n"
            f"Description: {request.description}\n\n"
            f"The sub-agent may now proceed with this action."
        )
    except KeyError:
        return f"Error: Permission request {request_id} not found."


@mcp.tool()
async def deny_permission(request_id: str, reason: str = "Denied by user") -> str:
    """Deny a permission request.

    Args:
        request_id: Permission request ID to deny
        reason: Reason for denial

    Returns:
        Success message
    """
    try:
        request = permission_manager.make_decision(
            request_id,
            PermissionDecision.DENIED,
            decided_by="user",
            reason=reason,
        )

        return (
            f"✗ Permission request {request_id} DENIED\n\n"
            f"Action: {request.action}\n"
            f"Description: {request.description}\n"
            f"Reason: {reason}\n\n"
            f"The sub-agent will be informed of the denial."
        )
    except KeyError:
        return f"Error: Permission request {request_id} not found."


@mcp.tool()
async def get_permission_status(request_id: str) -> str:
    """Get the status of a permission request.

    Args:
        request_id: Permission request ID

    Returns:
        Permission request status details
    """
    request = permission_manager.get_request(request_id)

    if not request:
        return f"Permission request {request_id} not found."

    status_symbols = {
        PermissionDecision.APPROVED: "✓",
        PermissionDecision.DENIED: "✗",
        PermissionDecision.PENDING: "⏳",
        PermissionDecision.ESCALATED: "⚠",
    }

    symbol = status_symbols.get(request.decision, "?")

    return (
        f"{symbol} Permission Request Status\n\n"
        f"Request ID: {request.request_id}\n"
        f"Session: {request.session_id}\n"
        f"Action: {request.action}\n"
        f"Description: {request.description}\n"
        f"Risk Level: {request.risk_level}\n"
        f"Status: {request.decision}\n"
        f"Created: {request.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Decided: {request.decided_at.strftime('%Y-%m-%d %H:%M:%S') if request.decided_at else 'N/A'}\n"
        f"Decided by: {request.decided_by or 'N/A'}\n"
        f"Denial reason: {request.denial_reason or 'N/A'}\n"
    )


@mcp.tool()
async def get_session_tree(session_id: str) -> str:
    """Get the hierarchical tree of sessions starting from a root session.

    Args:
        session_id: Root session ID to start the tree from

    Returns:
        Formatted session tree with hierarchy visualization
    """
    try:
        tree = session_manager.get_session_tree(session_id)

        def format_tree(node: dict, indent: int = 0) -> list[str]:
            """Recursively format the session tree."""
            lines = []
            prefix = "  " * indent

            # Format current node
            status_icon = {
                "created": "○",
                "running": "●",
                "completed": "✓",
                "failed": "✗",
                "terminated": "⊗",
            }.get(node["status"], "?")

            tags_str = ""
            if node["tags"]:
                tags_str = f" {node['tags']}"

            lines.append(
                f"{prefix}{status_icon} {node['session_id']} (depth={node['depth']}){tags_str}"
            )
            lines.append(f"{prefix}  Task: {node['task'][:60]}")
            lines.append(f"{prefix}  Branch: {node['branch_name']}")

            # Format children
            for child in node["children"]:
                lines.extend(format_tree(child, indent + 1))

            return lines

        tree_lines = format_tree(tree)
        return f"Session Tree:\n\n" + "\n".join(tree_lines)

    except SessionError as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def get_sessions_by_tags(tags: dict[str, str]) -> str:
    """Find sessions matching specific tags.

    Useful for finding all sessions for a feature, role, or strategy.

    Args:
        tags: Tag key-value pairs to match (e.g., {"feature": "AddAuth", "role": "reviewer"})

    Returns:
        List of matching sessions with details
    """
    matching_sessions = session_manager.get_sessions_by_tags(tags)

    if not matching_sessions:
        return f"No sessions found matching tags: {tags}"

    lines = [f"Sessions matching {tags}:\n"]
    for session in matching_sessions:
        status_icon = {
            "created": "○",
            "running": "●",
            "completed": "✓",
            "failed": "✗",
            "terminated": "⊗",
        }.get(session.status, "?")

        lines.append(f"\n{status_icon} {session.session_id} (depth={session.depth})")
        lines.append(f"  Task: {session.config.task[:60]}")
        lines.append(f"  Branch: {session.branch_name}")
        lines.append(f"  Status: {session.status}")
        if session.config.tags:
            lines.append(f"  All tags: {session.config.tags}")

    return "\n".join(lines)


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

    # Run the FastMCP server
    await mcp.run()


if __name__ == "__main__":
    asyncio.run(main())
