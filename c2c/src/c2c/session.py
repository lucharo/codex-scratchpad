"""Session management for Claude Code instances using Agent SDK."""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from .models import Session, SessionConfig, SessionStatus, SessionSummary
from .worktree import WorktreeManager, WorktreeError


class SessionError(Exception):
    """Exception raised for session-related errors."""

    pass


class SessionManager:
    """Manages Claude Code sessions with worktree isolation."""

    def __init__(self, repo_root: Path, worktree_base: Optional[Path] = None):
        """Initialize the session manager.

        Args:
            repo_root: Path to the git repository root
            worktree_base: Base directory for worktrees (defaults to repo_root/.c2c/worktrees)
        """
        self.repo_root = repo_root
        self.worktree_base = worktree_base or repo_root / ".c2c" / "worktrees"
        self.worktree_manager = WorktreeManager(repo_root)
        self.sessions: dict[str, Session] = {}
        self.clients: dict[str, ClaudeSDKClient] = {}

        # Ensure worktree base directory exists
        self.worktree_base.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"c2c-{uuid.uuid4().hex[:12]}"

    def _generate_branch_name(self, session_id: str, task: str) -> str:
        """Generate a branch name from session ID and task.

        Args:
            session_id: Session identifier
            task: Task description

        Returns:
            Generated branch name
        """
        # Sanitize task for branch name
        sanitized = "".join(
            c if c.isalnum() or c in "-_" else "-"
            for c in task.lower()[:30]
        ).strip("-")
        return f"claude/{session_id}-{sanitized}"

    def create_session(
        self, config: SessionConfig, parent_session_id: Optional[str] = None
    ) -> Session:
        """Create a new Claude Code session.

        Args:
            config: Session configuration
            parent_session_id: Optional parent session ID for hierarchy tracking

        Returns:
            Created session

        Raises:
            SessionError: If session creation fails
        """
        session_id = self._generate_session_id()

        # Calculate depth based on parent
        depth = 0
        if parent_session_id:
            parent = self.sessions.get(parent_session_id)
            if parent:
                depth = parent.depth + 1
                # Add this session to parent's children
                parent.child_session_ids.append(session_id)
            else:
                raise SessionError(f"Parent session not found: {parent_session_id}")

        # Generate branch name if not provided
        branch_name = config.branch_name or self._generate_branch_name(
            session_id, config.task
        )

        # Create session object
        session = Session(
            session_id=session_id,
            config=config,
            status=SessionStatus.CREATED,
            branch_name=branch_name,
            parent_session_id=parent_session_id,
            depth=depth,
        )

        # Create worktree if requested
        if config.use_worktree:
            try:
                worktree_path = self.worktree_base / session_id
                self.worktree_manager.create_worktree(
                    branch_name, worktree_path
                )
                session.worktree_path = worktree_path
            except WorktreeError as e:
                raise SessionError(
                    f"Failed to create worktree: {e}"
                ) from e

        # Store session
        self.sessions[session_id] = session

        return session

    async def start_session(self, session_id: str) -> None:
        """Start a Claude Code session and execute its task using Agent SDK.

        Args:
            session_id: Session identifier

        Raises:
            SessionError: If session start fails
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session not found: {session_id}")

        if session.status != SessionStatus.CREATED:
            raise SessionError(
                f"Session cannot be started (status: {session.status})"
            )

        try:
            # Determine working directory
            working_dir = (
                session.worktree_path
                if session.config.use_worktree
                else self.repo_root
            )

            # Create Agent SDK options
            options = ClaudeAgentOptions(
                cwd=working_dir,
                env=session.config.env_vars or {},
                continue_conversation=True,
                permission_mode="default"
            )

            # Create and connect SDK client
            client = ClaudeSDKClient(options)
            await client.connect()

            # Update session
            session.status = SessionStatus.RUNNING
            session.started_at = datetime.now()

            # Store client
            self.clients[session_id] = client

            # Execute the task immediately
            await self._execute_task_in_session(session_id)

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error_message = str(e)
            raise SessionError(f"Failed to start session: {e}") from e

    async def _execute_task_in_session(self, session_id: str) -> None:
        """Execute the session's task using the Agent SDK.

        Args:
            session_id: Session identifier
        """
        session = self.sessions.get(session_id)
        client = self.clients.get(session_id)

        if not session or not client:
            return

        try:
            # Send the task to the agent
            response = await client.query(session.config.task)

            # Collect the response
            session.output = [response.content] if hasattr(response, 'content') else [str(response)]
            session.completed_at = datetime.now()
            session.status = SessionStatus.COMPLETED

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error_message = str(e)
            session.completed_at = datetime.now()
            session.output = [f"Error: {str(e)}"]

        finally:
            # Clean up client
            try:
                await client.disconnect()
            except:
                pass

    async def terminate_session(
        self, session_id: str, force: bool = False
    ) -> None:
        """Terminate a running session.

        Args:
            session_id: Session identifier
            force: Force termination (kill instead of graceful shutdown)

        Raises:
            SessionError: If session termination fails
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session not found: {session_id}")

        client = self.clients.get(session_id)
        if client:
            try:
                if force:
                    await client.interrupt()
                await client.disconnect()
            except:
                pass

        session.status = SessionStatus.TERMINATED
        session.completed_at = datetime.now()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[SessionSummary]:
        """List all sessions.

        Returns:
            List of session summaries
        """
        return [
            SessionSummary(
                session_id=s.session_id,
                task=s.config.task,
                status=s.status,
                branch_name=s.branch_name,
                created_at=s.created_at,
                completed_at=s.completed_at,
            )
            for s in self.sessions.values()
        ]

    async def cleanup_session(
        self, session_id: str, remove_branch: bool = False
    ) -> None:
        """Clean up a session and its resources.

        Args:
            session_id: Session identifier
            remove_branch: Whether to delete the git branch

        Raises:
            SessionError: If cleanup fails
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session not found: {session_id}")

        # Disconnect client if connected
        client = self.clients.get(session_id)
        if client:
            try:
                await client.disconnect()
            except:
                pass

        # Remove worktree if exists
        if session.worktree_path:
            try:
                self.worktree_manager.remove_worktree(
                    session.worktree_path, force=True
                )
            except WorktreeError as e:
                raise SessionError(
                    f"Failed to remove worktree: {e}"
                ) from e

        # Remove branch if requested
        if remove_branch and session.branch_name:
            try:
                self.worktree_manager.cleanup_branch(
                    session.branch_name, force=True
                )
            except WorktreeError:
                # Branch might not exist or already deleted, ignore
                pass

        # Remove from tracking
        self.sessions.pop(session_id, None)
        self.clients.pop(session_id, None)

    async def get_session_output(self, session_id: str) -> list[str]:
        """Get output from a session.

        Args:
            session_id: Session identifier

        Returns:
            List of output lines

        Raises:
            SessionError: If session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session not found: {session_id}")

        return session.output

    def get_session_tree(self, session_id: str) -> dict:
        """Get the session hierarchy tree starting from a session.

        Args:
            session_id: Root session identifier

        Returns:
            Dictionary representing the session tree with nested children

        Raises:
            SessionError: If session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session not found: {session_id}")

        def build_tree(s: Session) -> dict:
            return {
                "session_id": s.session_id,
                "task": s.config.task,
                "status": s.status,
                "depth": s.depth,
                "tags": s.config.tags,
                "branch_name": s.branch_name,
                "children": [
                    build_tree(self.sessions[child_id])
                    for child_id in s.child_session_ids
                    if child_id in self.sessions
                ],
            }

        return build_tree(session)

    def get_ancestors(self, session_id: str) -> list[Session]:
        """Get all ancestor sessions (parent, grandparent, etc.).

        Args:
            session_id: Session identifier

        Returns:
            List of ancestor sessions, ordered from immediate parent to root

        Raises:
            SessionError: If session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session not found: {session_id}")

        ancestors = []
        current = session

        while current.parent_session_id:
            parent = self.sessions.get(current.parent_session_id)
            if not parent:
                break
            ancestors.append(parent)
            current = parent

        return ancestors

    def get_descendants(self, session_id: str) -> list[Session]:
        """Get all descendant sessions (children, grandchildren, etc.).

        Args:
            session_id: Session identifier

        Returns:
            List of all descendant sessions

        Raises:
            SessionError: If session not found
        """
        session = self.sessions.get(session_id)
        if not session:
            raise SessionError(f"Session not found: {session_id}")

        descendants = []

        def collect_descendants(s: Session):
            for child_id in s.child_session_ids:
                child = self.sessions.get(child_id)
                if child:
                    descendants.append(child)
                    collect_descendants(child)

        collect_descendants(session)
        return descendants

    def get_root_sessions(self) -> list[Session]:
        """Get all root sessions (sessions with no parent).

        Returns:
            List of root sessions
        """
        return [s for s in self.sessions.values() if s.parent_session_id is None]

    def get_sessions_by_tags(self, tags: dict[str, str]) -> list[Session]:
        """Find sessions matching specific tags.

        Args:
            tags: Dictionary of tag key-value pairs to match

        Returns:
            List of sessions where all specified tags match
        """
        matching_sessions = []

        for session in self.sessions.values():
            if all(
                session.config.tags.get(key) == value
                for key, value in tags.items()
            ):
                matching_sessions.append(session)

        return matching_sessions
