"""Integration tests for MCP server."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cc2cc.server import (
    _cleanup_session,
    _create_session,
    _get_session,
    _get_session_output,
    _list_sessions,
    _start_session,
    _terminate_session,
    list_tools,
    session_manager,
)


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


@pytest.fixture
def setup_session_manager(temp_git_repo):
    """Set up the global session manager for tests."""
    import cc2cc.server as server_module
    from cc2cc.session import SessionManager

    server_module.session_manager = SessionManager(temp_git_repo)
    yield server_module.session_manager
    server_module.session_manager = None


@pytest.mark.asyncio
async def test_list_tools():
    """Test listing available tools."""
    tools = await list_tools()

    assert len(tools) == 7

    tool_names = [tool.name for tool in tools]
    assert "create_session" in tool_names
    assert "start_session" in tool_names
    assert "get_session" in tool_names
    assert "list_sessions" in tool_names
    assert "get_session_output" in tool_names
    assert "terminate_session" in tool_names
    assert "cleanup_session" in tool_names


@pytest.mark.asyncio
async def test_create_session_tool(setup_session_manager):
    """Test create_session tool."""
    arguments = {
        "task": "Test task for session",
        "use_worktree": False,
    }

    result = await _create_session(arguments)

    assert len(result) == 1
    assert "Session created successfully" in result[0].text
    assert "Session ID:" in result[0].text
    assert "claude/" in result[0].text


@pytest.mark.asyncio
async def test_create_session_with_custom_branch(setup_session_manager):
    """Test creating session with custom branch name."""
    arguments = {
        "task": "Test task",
        "branch_name": "my-custom-branch",
        "use_worktree": False,
    }

    result = await _create_session(arguments)

    assert "my-custom-branch" in result[0].text


@pytest.mark.asyncio
async def test_list_sessions_empty(setup_session_manager):
    """Test listing sessions when none exist."""
    result = await _list_sessions({})

    assert len(result) == 1
    assert "No sessions found" in result[0].text


@pytest.mark.asyncio
async def test_list_sessions(setup_session_manager):
    """Test listing sessions."""
    # Create a session first
    await _create_session({"task": "Test task 1", "use_worktree": False})
    await _create_session({"task": "Test task 2", "use_worktree": False})

    result = await _list_sessions({})

    assert len(result) == 1
    assert "Test task 1" in result[0].text
    assert "Test task 2" in result[0].text


@pytest.mark.asyncio
async def test_get_session(setup_session_manager):
    """Test getting session details."""
    # Create a session first
    create_result = await _create_session({
        "task": "Test task for get",
        "use_worktree": False,
    })

    # Extract session ID from result
    session_id = None
    for line in create_result[0].text.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    assert session_id is not None

    # Get session details
    result = await _get_session({"session_id": session_id})

    assert len(result) == 1
    assert "Session Details:" in result[0].text
    assert session_id in result[0].text
    assert "Test task for get" in result[0].text


@pytest.mark.asyncio
async def test_get_session_not_found(setup_session_manager):
    """Test getting non-existent session."""
    from cc2cc.session import SessionError

    with pytest.raises(SessionError, match="Session not found"):
        await _get_session({"session_id": "nonexistent-id"})


@pytest.mark.asyncio
async def test_start_session_mock(setup_session_manager):
    """Test starting a session with mock."""
    # Create a session
    create_result = await _create_session({
        "task": "Test task to start",
        "use_worktree": False,
    })

    # Extract session ID
    session_id = None
    for line in create_result[0].text.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Mock the process start
    mock_process = MagicMock()
    mock_process.start = AsyncMock(return_value=12345)
    mock_process.wait = AsyncMock(return_value=0)
    mock_process.get_output = MagicMock(return_value=["output"])

    with patch(
        "cc2cc.session.ClaudeCodeProcess",
        return_value=mock_process,
    ):
        result = await _start_session({"session_id": session_id})

    assert "started successfully" in result[0].text


@pytest.mark.asyncio
async def test_terminate_session(setup_session_manager):
    """Test terminating a session."""
    # Create a session
    create_result = await _create_session({
        "task": "Test task to terminate",
        "use_worktree": False,
    })

    # Extract session ID
    session_id = None
    for line in create_result[0].text.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Terminate without actually starting (should still work)
    result = await _terminate_session({"session_id": session_id, "force": False})

    assert "terminated" in result[0].text


@pytest.mark.asyncio
async def test_get_session_output_no_output(setup_session_manager):
    """Test getting output from session with no output."""
    # Create a session
    create_result = await _create_session({
        "task": "Test task for output",
        "use_worktree": False,
    })

    # Extract session ID
    session_id = None
    for line in create_result[0].text.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Get output
    result = await _get_session_output({"session_id": session_id})

    assert "No output available" in result[0].text


@pytest.mark.asyncio
async def test_cleanup_session(setup_session_manager):
    """Test cleaning up a session."""
    # Create a session
    create_result = await _create_session({
        "task": "Test task to cleanup",
        "use_worktree": False,
    })

    # Extract session ID
    session_id = None
    for line in create_result[0].text.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Cleanup
    result = await _cleanup_session({
        "session_id": session_id,
        "remove_branch": False,
    })

    assert "cleaned up" in result[0].text

    # Verify session is gone
    sessions = await _list_sessions({})
    assert "No sessions found" in sessions[0].text


@pytest.mark.asyncio
async def test_cleanup_session_with_branch_removal(setup_session_manager):
    """Test cleaning up session with branch removal."""
    # Create a session with worktree
    create_result = await _create_session({
        "task": "Test task to cleanup with branch",
        "use_worktree": True,
    })

    # Extract session ID
    session_id = None
    for line in create_result[0].text.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Cleanup with branch removal
    result = await _cleanup_session({
        "session_id": session_id,
        "remove_branch": True,
    })

    assert "cleaned up" in result[0].text
