"""Session management for Claude Code instances."""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Session, SessionConfig, SessionStatus, SessionSummary
from .process import ClaudeCodeProcess, ProcessError
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
            worktree_base: Base directory for worktrees (defaults to repo_root/.worktrees)
        """
        self.repo_root = repo_root
        self.worktree_base = worktree_base or repo_root / ".worktrees"
        self.worktree_manager = WorktreeManager(repo_root)
        self.sessions: dict[str, Session] = {}
        self.processes: dict[str, ClaudeCodeProcess] = {}

        # Ensure worktree base directory exists
        self.worktree_base.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"cc2cc-{uuid.uuid4().hex[:12]}"

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

    def create_session(self, config: SessionConfig) -> Session:
        """Create a new Claude Code session.

        Args:
            config: Session configuration

        Returns:
            Created session

        Raises:
            SessionError: If session creation fails
        """
        session_id = self._generate_session_id()

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
        """Start a Claude Code session.

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

            # Create and start process
            process = ClaudeCodeProcess(
                working_dir=working_dir,
                task=session.config.task,
                env_vars=session.config.env_vars,
            )

            pid = await process.start()

            # Update session
            session.process_id = pid
            session.status = SessionStatus.RUNNING
            session.started_at = datetime.now()

            # Store process
            self.processes[session_id] = process

            # Start monitoring task
            asyncio.create_task(self._monitor_session(session_id))

        except ProcessError as e:
            session.status = SessionStatus.FAILED
            session.error_message = str(e)
            raise SessionError(f"Failed to start session: {e}") from e

    async def _monitor_session(self, session_id: str) -> None:
        """Monitor a session and update its status.

        Args:
            session_id: Session identifier
        """
        session = self.sessions.get(session_id)
        process = self.processes.get(session_id)

        if not session or not process:
            return

        try:
            # Wait for process to complete
            exit_code = await process.wait(timeout=session.config.timeout)

            # Update session based on exit code
            session.output = process.get_output()
            session.completed_at = datetime.now()

            if exit_code == 0:
                session.status = SessionStatus.COMPLETED
            else:
                session.status = SessionStatus.FAILED
                session.error_message = f"Process exited with code {exit_code}"

        except asyncio.TimeoutError:
            session.status = SessionStatus.FAILED
            session.error_message = "Session timeout"
            session.completed_at = datetime.now()
            session.output = process.get_output()

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error_message = str(e)
            session.completed_at = datetime.now()
            if process:
                session.output = process.get_output()

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

        process = self.processes.get(session_id)
        if process and process.is_running():
            if force:
                await process.kill()
            else:
                await process.terminate()

        session.status = SessionStatus.TERMINATED
        session.completed_at = datetime.now()
        if process:
            session.output = process.get_output()

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

        # Terminate process if running
        process = self.processes.get(session_id)
        if process and process.is_running():
            await process.terminate()

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
        self.processes.pop(session_id, None)

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

        # Get latest output from process if still running
        process = self.processes.get(session_id)
        if process:
            session.output = process.get_output()

        return session.output
