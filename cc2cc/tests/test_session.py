"""Tests for session management."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cc2cc.models import SessionConfig, SessionStatus
from cc2cc.session import SessionError, SessionManager


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Configure git for testing
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "safe.directory", "*"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "commit.gpgsign", "false"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Create initial commit
        test_file = repo_path / "README.md"
        test_file.write_text("# Test Repository")
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        yield repo_path


def test_session_manager_init(temp_git_repo):
    """Test SessionManager initialization."""
    manager = SessionManager(temp_git_repo)

    assert manager.repo_root == temp_git_repo
    assert manager.worktree_base == temp_git_repo / ".worktrees"
    assert manager.worktree_base.exists()
    assert manager.sessions == {}
    assert manager.processes == {}


def test_session_manager_custom_worktree_base(temp_git_repo):
    """Test SessionManager with custom worktree base."""
    custom_base = temp_git_repo / "custom_worktrees"
    manager = SessionManager(temp_git_repo, worktree_base=custom_base)

    assert manager.worktree_base == custom_base
    assert custom_base.exists()


def test_create_session(temp_git_repo):
    """Test creating a session."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task")

    session = manager.create_session(config)

    assert session.session_id.startswith("cc2cc-")
    assert session.config.task == "Test task"
    assert session.status == SessionStatus.CREATED
    assert session.branch_name.startswith("claude/")
    assert session.session_id in manager.sessions


def test_create_session_with_custom_branch(temp_git_repo):
    """Test creating a session with custom branch name."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", branch_name="custom-branch")

    session = manager.create_session(config)

    assert session.branch_name == "custom-branch"


def test_create_session_with_worktree(temp_git_repo):
    """Test creating a session with worktree."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=True)

    session = manager.create_session(config)

    assert session.worktree_path is not None
    assert session.worktree_path.exists()
    assert (session.worktree_path / "README.md").exists()


def test_create_session_without_worktree(temp_git_repo):
    """Test creating a session without worktree."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=False)

    session = manager.create_session(config)

    assert session.worktree_path is None


@pytest.mark.asyncio
async def test_start_session_not_found(temp_git_repo):
    """Test starting a non-existent session."""
    manager = SessionManager(temp_git_repo)

    with pytest.raises(SessionError, match="Session not found"):
        await manager.start_session("nonexistent-id")


@pytest.mark.asyncio
async def test_start_session_already_running(temp_git_repo):
    """Test starting an already running session."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=False)
    session = manager.create_session(config)

    # Manually set status to running
    session.status = SessionStatus.RUNNING

    with pytest.raises(SessionError, match="cannot be started"):
        await manager.start_session(session.session_id)


@pytest.mark.asyncio
async def test_start_session_mock(temp_git_repo):
    """Test starting a session with mocked process."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=False)
    session = manager.create_session(config)

    mock_process = MagicMock()
    mock_process.start = AsyncMock(return_value=12345)
    mock_process.wait = AsyncMock(return_value=0)
    mock_process.get_output = MagicMock(return_value=["output line"])

    with patch(
        "cc2cc.session.ClaudeCodeProcess",
        return_value=mock_process,
    ):
        await manager.start_session(session.session_id)

    assert session.status == SessionStatus.RUNNING
    assert session.process_id == 12345
    assert session.started_at is not None


def test_get_session(temp_git_repo):
    """Test getting a session."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task")
    created_session = manager.create_session(config)

    retrieved_session = manager.get_session(created_session.session_id)

    assert retrieved_session == created_session


def test_get_session_not_found(temp_git_repo):
    """Test getting a non-existent session."""
    manager = SessionManager(temp_git_repo)

    session = manager.get_session("nonexistent-id")

    assert session is None


def test_list_sessions(temp_git_repo):
    """Test listing sessions."""
    manager = SessionManager(temp_git_repo)

    # Create multiple sessions
    config1 = SessionConfig(task="Task 1", use_worktree=False)
    config2 = SessionConfig(task="Task 2", use_worktree=False)

    session1 = manager.create_session(config1)
    session2 = manager.create_session(config2)

    summaries = manager.list_sessions()

    assert len(summaries) == 2
    session_ids = [s.session_id for s in summaries]
    assert session1.session_id in session_ids
    assert session2.session_id in session_ids


def test_list_sessions_empty(temp_git_repo):
    """Test listing sessions when none exist."""
    manager = SessionManager(temp_git_repo)

    summaries = manager.list_sessions()

    assert summaries == []


@pytest.mark.asyncio
async def test_terminate_session(temp_git_repo):
    """Test terminating a session."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=False)
    session = manager.create_session(config)

    # Mock process
    mock_process = MagicMock()
    mock_process.is_running = MagicMock(return_value=True)
    mock_process.terminate = AsyncMock()
    mock_process.get_output = MagicMock(return_value=["output"])

    manager.processes[session.session_id] = mock_process
    session.status = SessionStatus.RUNNING

    await manager.terminate_session(session.session_id)

    assert session.status == SessionStatus.TERMINATED
    assert session.completed_at is not None
    mock_process.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_terminate_session_force(temp_git_repo):
    """Test force terminating a session."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=False)
    session = manager.create_session(config)

    # Mock process
    mock_process = MagicMock()
    mock_process.is_running = MagicMock(return_value=True)
    mock_process.kill = AsyncMock()
    mock_process.get_output = MagicMock(return_value=["output"])

    manager.processes[session.session_id] = mock_process
    session.status = SessionStatus.RUNNING

    await manager.terminate_session(session.session_id, force=True)

    assert session.status == SessionStatus.TERMINATED
    mock_process.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_session(temp_git_repo):
    """Test cleaning up a session."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=True)
    session = manager.create_session(config)

    worktree_path = session.worktree_path
    assert worktree_path.exists()

    await manager.cleanup_session(session.session_id)

    # Session should be removed from tracking
    assert session.session_id not in manager.sessions

    # Worktree should be removed
    assert not manager.worktree_manager.worktree_exists(worktree_path)


@pytest.mark.asyncio
async def test_cleanup_session_with_branch_removal(temp_git_repo):
    """Test cleaning up a session and removing branch."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=True)
    session = manager.create_session(config)

    branch_name = session.branch_name

    await manager.cleanup_session(session.session_id, remove_branch=True)

    # Verify branch is removed
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout


@pytest.mark.asyncio
async def test_get_session_output(temp_git_repo):
    """Test getting session output."""
    manager = SessionManager(temp_git_repo)
    config = SessionConfig(task="Test task", use_worktree=False)
    session = manager.create_session(config)

    # Mock process
    mock_process = MagicMock()
    mock_process.get_output = MagicMock(return_value=["line 1", "line 2"])

    manager.processes[session.session_id] = mock_process

    output = await manager.get_session_output(session.session_id)

    assert output == ["line 1", "line 2"]
    assert session.output == ["line 1", "line 2"]


@pytest.mark.asyncio
async def test_get_session_output_not_found(temp_git_repo):
    """Test getting output from non-existent session."""
    manager = SessionManager(temp_git_repo)

    with pytest.raises(SessionError, match="Session not found"):
        await manager.get_session_output("nonexistent-id")
