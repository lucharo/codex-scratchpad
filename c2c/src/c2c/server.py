"""MCP server implementation for c2c using FastMCP."""

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


@mcp.tool()
async def list_worktrees() -> str:
    """List all c2c worktrees, showing which are tracked vs orphaned.

    Orphaned worktrees are those that exist on disk but aren't tracked by any
    active session (usually caused by MCP server restart).

    Returns:
        Formatted list of worktrees with their status
    """
    import os

    # Get all worktrees from git
    git_worktrees = session_manager.worktree_manager.list_worktrees()

    # Filter for c2c worktrees (in .c2c/worktrees/)
    c2c_worktrees = [
        wt for wt in git_worktrees
        if ".c2c/worktrees/" in wt.get("path", "") or ".c2c\\worktrees\\" in wt.get("path", "")
    ]

    if not c2c_worktrees:
        return "No c2c worktrees found."

    # Get tracked session IDs
    tracked_session_ids = set(session_manager.sessions.keys())

    lines = ["C2C Worktrees:\n"]

    tracked_count = 0
    orphaned_count = 0

    for wt in c2c_worktrees:
        path = wt.get("path", "")
        branch = wt.get("branch", "N/A")

        # Extract session ID from path (.c2c/worktrees/c2c-XXXXX)
        session_id = os.path.basename(path)

        if session_id in tracked_session_ids:
            # Tracked session
            session = session_manager.sessions[session_id]
            status_icon = {
                "created": "○",
                "running": "●",
                "completed": "✓",
                "failed": "✗",
                "terminated": "⊗",
            }.get(session.status, "?")
            lines.append(f"\n✓ {session_id} - TRACKED ({status_icon} {session.status})")
            tracked_count += 1
        else:
            # Orphaned worktree
            lines.append(f"\n⚠ {session_id} - ORPHANED")
            orphaned_count += 1

        lines.append(f"  Path: {path}")
        lines.append(f"  Branch: {branch}")

    lines.append(f"\n\nSummary: {tracked_count} tracked, {orphaned_count} orphaned")

    if orphaned_count > 0:
        lines.append("\nUse cleanup_worktrees() to remove orphaned worktrees.")

    return "\n".join(lines)


@mcp.tool()
async def cleanup_worktrees(orphaned_only: bool = True, force: bool = True) -> str:
    """Clean up c2c worktrees and their branches.

    By default, only removes orphaned worktrees (not tracked by active sessions).
    Use orphaned_only=False to remove ALL c2c worktrees.

    Args:
        orphaned_only: Only remove orphaned worktrees (default: True)
        force: Force removal even if there are uncommitted changes (default: True)

    Returns:
        Summary of cleanup operations
    """
    import os

    # Get all worktrees from git
    git_worktrees = session_manager.worktree_manager.list_worktrees()

    # Filter for c2c worktrees
    c2c_worktrees = [
        wt for wt in git_worktrees
        if ".c2c/worktrees/" in wt.get("path", "") or ".c2c\\worktrees\\" in wt.get("path", "")
    ]

    if not c2c_worktrees:
        return "No c2c worktrees found to clean up."

    # Get tracked session IDs
    tracked_session_ids = set(session_manager.sessions.keys())

    removed_worktrees = []
    removed_branches = []
    errors = []

    for wt in c2c_worktrees:
        path = wt.get("path", "")
        branch = wt.get("branch", "")
        session_id = os.path.basename(path)

        # Determine if we should remove this worktree
        should_remove = False
        if orphaned_only:
            should_remove = session_id not in tracked_session_ids
        else:
            should_remove = True

        if not should_remove:
            continue

        # Remove worktree
        try:
            session_manager.worktree_manager.remove_worktree(
                Path(path), force=force
            )
            removed_worktrees.append(session_id)

            # Remove branch if it exists
            if branch and branch != "N/A":
                try:
                    session_manager.worktree_manager.cleanup_branch(
                        branch, force=True
                    )
                    removed_branches.append(branch)
                except Exception:
                    # Branch might already be deleted, ignore
                    pass
        except Exception as e:
            errors.append(f"{session_id}: {str(e)}")

    # Build summary
    lines = ["Cleanup Summary:\n"]

    if removed_worktrees:
        lines.append(f"✓ Removed {len(removed_worktrees)} worktree(s):")
        for wt_id in removed_worktrees:
            lines.append(f"  - {wt_id}")
    else:
        lines.append("No worktrees were removed.")

    if removed_branches:
        lines.append(f"\n✓ Deleted {len(removed_branches)} branch(es):")
        for branch in removed_branches:
            lines.append(f"  - {branch}")

    if errors:
        lines.append(f"\n⚠ Errors ({len(errors)}):")
        for error in errors:
            lines.append(f"  - {error}")

    return "\n".join(lines)


# Message Passing Tools for Multi-Agent Communication


@mcp.tool()
async def send_message(session_id: str, message: str) -> str:
    """Send a message to a running Claude Code session.

    Allows agents to communicate with each other by sending messages
    through file-based message queues.

    Args:
        session_id: Target session ID to receive the message
        message: Message content to send

    Returns:
        Confirmation that message was queued
    """
    try:
        session = session_manager.get_session(session_id)

        if session.status != "running":
            return f"Error: Session {session_id} is not running (status: {session.status})"

        # Create message queue directory
        message_dir = session.working_dir / ".c2c" / "messages"
        message_dir.mkdir(parents=True, exist_ok=True)

        # Create message file with timestamp
        import time
        timestamp = int(time.time())
        message_file = message_dir / f"incoming_{timestamp}.txt"

        # Write message
        message_file.write_text(f"From: main_agent\nTimestamp: {timestamp}\n\n{message}")

        return f"Message sent to session {session_id}"

    except Exception as e:
        return f"Error sending message: {str(e)}"


@mcp.tool()
async def get_messages(session_id: str) -> str:
    """Get all messages for a session.

    Retrieves both incoming and outgoing messages for a session,
    allowing agents to check their communication history.

    Args:
        session_id: Session ID to get messages for

    Returns:
        Formatted list of all messages
    """
    try:
        session = session_manager.get_session(session_id)

        # Get message queue directory
        message_dir = session.working_dir / ".c2c" / "messages"

        if not message_dir.exists():
            return f"No messages found for session {session_id}"

        # Read all message files
        messages = []

        # Incoming messages
        for msg_file in sorted(message_dir.glob("incoming_*.txt")):
            content = msg_file.read_text()
            messages.append(f"📨 INCOMING [{msg_file.name}]:\n{content}\n")

        # Outgoing messages
        for msg_file in sorted(message_dir.glob("outgoing_*.txt")):
            content = msg_file.read_text()
            messages.append(f"📤 OUTGOING [{msg_file.name}]:\n{content}\n")

        if not messages:
            return f"No messages found for session {session_id}"

        return f"Messages for session {session_id}:\n\n" + "\n".join(messages)

    except Exception as e:
        return f"Error getting messages: {str(e)}"


@mcp.tool()
async def create_session_registry() -> str:
    """Create a session registry for tracking running agents.

    Creates a central registry file that tracks all active sessions,
    their capabilities, and current status. Enables agent discovery.

    Returns:
        Confirmation of registry creation
    """
    try:
        import time
        import json

        registry_dir = session_manager.repo_root / ".c2c" / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)

        registry_file = registry_dir / "sessions.json"

        # Build registry data
        registry_data = {
            "timestamp": int(time.time()),
            "sessions": {}
        }

        for session_id, session in session_manager.sessions.items():
            registry_data["sessions"][session_id] = {
                "status": session.status,
                "task": session.config.task[:100],  # Truncate long tasks
                "branch": session.branch_name,
                "worktree": str(session.working_dir),
                "created": session.config.created_at if hasattr(session.config, 'created_at') else None,
                "tags": session.config.tags or {},
                "pid": session.process.pid if session.process else None
            }

        # Write registry
        registry_file.write_text(json.dumps(registry_data, indent=2))

        return f"Session registry created with {len(registry_data['sessions'])} sessions"

    except Exception as e:
        return f"Error creating registry: {str(e)}"


@mcp.tool()
async def get_session_registry() -> str:
    """Get the current session registry.

    Returns information about all registered sessions,
    enabling agents to discover and communicate with each other.

    Returns:
        Formatted session registry information
    """
    try:
        registry_file = session_manager.repo_root / ".c2c" / "registry" / "sessions.json"

        if not registry_file.exists():
            return "No session registry found. Use create_session_registry() to create one."

        import json
        registry_data = json.loads(registry_file.read_text())

        lines = [f"Session Registry (updated: {registry_data['timestamp']}):\n"]

        for session_id, info in registry_data["sessions"].items():
            status_icon = {
                "created": "○",
                "running": "●",
                "completed": "✓",
                "failed": "✗",
                "terminated": "⊗",
            }.get(info["status"], "?")

            lines.append(f"{status_icon} {session_id}")
            lines.append(f"  Status: {info['status']}")
            lines.append(f"  Task: {info['task']}")
            lines.append(f"  Branch: {info['branch']}")
            if info.get("pid"):
                lines.append(f"  PID: {info['pid']}")
            if info.get("tags"):
                lines.append(f"  Tags: {info['tags']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error reading registry: {str(e)}"


def main(repo_root: Path | str = None):
    """Run the MCP server."""
    import sys

    global session_manager, permission_manager

    # Initialize managers
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    session_manager = SessionManager(repo_root)
    permission_manager = PermissionManager()

    # Run the FastMCP server (synchronous, uses stdio by default)
    # Handle the case where asyncio is already running (e.g., from Claude Code)
    try:
        mcp.run()
    except RuntimeError as e:
        if "asyncio" in str(e).lower() or "already running" in str(e).lower():
            # If there's already an event loop, use sniffio to run in that context
            print("Detected existing event loop, using nest_asyncio workaround...", file=sys.stderr)
            import nest_asyncio
            nest_asyncio.apply()
            mcp.run()
        else:
            raise


if __name__ == "__main__":
    main()
