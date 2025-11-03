"""Integration tests for MCP server."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from c2c.server import (
    cleanup_session,
    cleanup_worktrees,
    create_session,
    get_session,
    get_session_output,
    list_sessions,
    list_worktrees,
    start_session,
    terminate_session,
    mcp,
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
    import c2c.server as server_module
    from c2c.session import SessionManager

    server_module.session_manager = SessionManager(temp_git_repo)
    yield server_module.session_manager
    server_module.session_manager = None


def test_list_tools():
    """Test that all tool functions are available via FastMCP."""
    # FastMCP automatically registers tools via decorators
    # Verify that the tool functions exist and are callable
    from c2c.server import (
        approve_permission,
        cleanup_session,
        create_session,
        deny_permission,
        get_pending_permissions,
        get_permission_status,
        get_session,
        get_session_output,
        get_session_tree,
        get_sessions_by_tags,
        list_sessions,
        request_permission,
        start_session,
        terminate_session,
    )

    # Verify all 16 tools are callable functions
    tools = [
        create_session,
        start_session,
        get_session,
        list_sessions,
        get_session_output,
        terminate_session,
        cleanup_session,
        request_permission,
        get_pending_permissions,
        approve_permission,
        deny_permission,
        get_permission_status,
        get_session_tree,
        get_sessions_by_tags,
        list_worktrees,
        cleanup_worktrees,
    ]

    assert len(tools) == 16
    for tool in tools:
        assert callable(tool), f"{tool.__name__} is not callable"


@pytest.mark.asyncio
async def test_create_session_tool(setup_session_manager):
    """Test create_session tool."""
    result = await create_session(
        task="Test task for session",
        use_worktree=False,
    )

    assert "Session created successfully" in result
    assert "Session ID:" in result
    assert "claude/" in result


@pytest.mark.asyncio
async def test_create_session_with_custom_branch(setup_session_manager):
    """Test creating session with custom branch name."""
    result = await create_session(
        task="Test task",
        branch_name="my-custom-branch",
        use_worktree=False,
    )

    assert "my-custom-branch" in result


@pytest.mark.asyncio
async def test_list_sessions_empty(setup_session_manager):
    """Test listing sessions when none exist."""
    result = await list_sessions()

    assert "No sessions found" in result


@pytest.mark.asyncio
async def test_list_sessions(setup_session_manager):
    """Test listing sessions."""
    # Create a session first
    await create_session(task="Test task 1", use_worktree=False)
    await create_session(task="Test task 2", use_worktree=False)

    result = await list_sessions()

    assert "Test task 1" in result
    assert "Test task 2" in result


@pytest.mark.asyncio
async def test_get_session(setup_session_manager):
    """Test getting session details."""
    # Create a session first
    create_result = await create_session(
        task="Test task for get",
        use_worktree=False,
    )

    # Extract session ID from result
    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    assert session_id is not None

    # Get session details
    result = await get_session(session_id=session_id)

    assert "Session Details:" in result
    assert session_id in result
    assert "Test task for get" in result


@pytest.mark.asyncio
async def test_get_session_not_found(setup_session_manager):
    """Test getting non-existent session."""
    from c2c.session import SessionError

    with pytest.raises(SessionError, match="Session not found"):
        await get_session(session_id="nonexistent-id")


@pytest.mark.asyncio
async def test_start_session_mock(setup_session_manager):
    """Test starting a session with mock."""
    # Create a session
    create_result = await create_session(
        task="Test task to start",
        use_worktree=False,
    )

    # Extract session ID
    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Mock the Agent SDK client
    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.query = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "Task completed successfully"
    mock_client.query.return_value = mock_response
    mock_client.disconnect = AsyncMock()

    with patch(
        "c2c.session.ClaudeSDKClient",
        return_value=mock_client,
    ):
        result = await start_session(session_id=session_id)

    assert "started successfully" in result


@pytest.mark.asyncio
async def test_terminate_session(setup_session_manager):
    """Test terminating a session."""
    # Create a session
    create_result = await create_session(
        task="Test task to terminate",
        use_worktree=False,
    )

    # Extract session ID
    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Terminate without actually starting (should still work)
    result = await terminate_session(session_id=session_id, force=False)

    assert "terminated" in result


@pytest.mark.asyncio
async def test_get_session_output_no_output(setup_session_manager):
    """Test getting output from session with no output."""
    # Create a session
    create_result = await create_session(
        task="Test task for output",
        use_worktree=False,
    )

    # Extract session ID
    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Get output
    result = await get_session_output(session_id=session_id)

    assert "No output available" in result


@pytest.mark.asyncio
async def test_cleanup_session(setup_session_manager):
    """Test cleaning up a session."""
    # Create a session
    create_result = await create_session(
        task="Test task to cleanup",
        use_worktree=False,
    )

    # Extract session ID
    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Cleanup
    result = await cleanup_session(
        session_id=session_id,
        remove_branch=False,
    )

    assert "cleaned up" in result

    # Verify session is gone
    sessions = await list_sessions()
    assert "No sessions found" in sessions


@pytest.mark.asyncio
async def test_cleanup_session_with_branch_removal(setup_session_manager):
    """Test cleaning up session with branch removal."""
    # Create a session with worktree
    create_result = await create_session(
        task="Test task to cleanup with branch",
        use_worktree=True,
    )

    # Extract session ID
    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Cleanup with branch removal
    result = await cleanup_session(
        session_id=session_id,
        remove_branch=True,
    )

    assert "cleaned up" in result


# =============================================================================
# Worktree Management Tests
# =============================================================================


@pytest.mark.asyncio
async def test_list_worktrees_when_no_c2c_worktrees_exist(setup_session_manager):
    """Test listing worktrees when no c2c worktrees exist."""
    # Mock list_worktrees to return only main repo (no c2c worktrees)
    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {"path": str(setup_session_manager.repo_root), "branch": "main"}
        ],
    ):
        result = await list_worktrees()

    assert "No c2c worktrees found" in result


@pytest.mark.asyncio
async def test_list_worktrees_shows_tracked_session_as_tracked(setup_session_manager):
    """Test that list_worktrees shows tracked sessions with correct status."""
    # Create a session with worktree
    create_result = await create_session(
        task="Test tracked worktree",
        use_worktree=True,
    )

    # Extract session ID
    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    # Get the session to find worktree path
    session = setup_session_manager.sessions[session_id]
    worktree_path = str(session.worktree_path)

    # Mock list_worktrees to return the tracked worktree
    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {
                "path": worktree_path,
                "branch": session.branch_name,
            }
        ],
    ):
        result = await list_worktrees()

    assert "✓" in result
    assert "TRACKED" in result
    assert session_id in result


@pytest.mark.asyncio
async def test_list_worktrees_shows_orphaned_worktree_as_orphaned(
    setup_session_manager,
):
    """Test that list_worktrees identifies orphaned worktrees."""
    # Mock an orphaned worktree (not tracked in session_manager)
    fake_orphan_path = (
        str(setup_session_manager.repo_root) + "/.c2c/worktrees/c2c-orphaned123"
    )

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {
                "path": fake_orphan_path,
                "branch": "some-orphan-branch",
            }
        ],
    ):
        result = await list_worktrees()

    assert "⚠" in result
    assert "ORPHANED" in result
    assert "c2c-orphaned123" in result


@pytest.mark.asyncio
async def test_list_worktrees_distinguishes_tracked_from_orphaned(
    setup_session_manager,
):
    """Test that list_worktrees correctly categorizes tracked vs orphaned."""
    # Create a tracked session
    create_result = await create_session(
        task="Test mixed worktrees",
        use_worktree=True,
    )

    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    session = setup_session_manager.sessions[session_id]
    tracked_path = str(session.worktree_path)
    orphan_path = str(setup_session_manager.repo_root) + "/.c2c/worktrees/c2c-orphan"

    # Mock both tracked and orphaned worktrees
    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {"path": tracked_path, "branch": session.branch_name},
            {"path": orphan_path, "branch": "orphan-branch"},
        ],
    ):
        result = await list_worktrees()

    assert "✓" in result and "TRACKED" in result
    assert "⚠" in result and "ORPHANED" in result


@pytest.mark.asyncio
async def test_list_worktrees_shows_correct_summary_counts(setup_session_manager):
    """Test that list_worktrees shows accurate summary counts."""
    # Create 2 tracked sessions
    for i in range(2):
        await create_session(
            task=f"Tracked session {i}",
            use_worktree=True,
        )

    # Get tracked worktrees
    tracked_worktrees = [
        {"path": str(s.worktree_path), "branch": s.branch_name}
        for s in setup_session_manager.sessions.values()
    ]

    # Add 3 orphaned worktrees
    orphan_worktrees = [
        {
            "path": f"{setup_session_manager.repo_root}/.c2c/worktrees/c2c-orphan{i}",
            "branch": f"orphan-branch-{i}",
        }
        for i in range(3)
    ]

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=tracked_worktrees + orphan_worktrees,
    ):
        result = await list_worktrees()

    assert "Summary: 2 tracked, 3 orphaned" in result


@pytest.mark.asyncio
async def test_list_worktrees_suggests_cleanup_when_orphans_exist(
    setup_session_manager,
):
    """Test that list_worktrees suggests cleanup when orphans are found."""
    orphan_path = str(setup_session_manager.repo_root) + "/.c2c/worktrees/c2c-orphan"

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[{"path": orphan_path, "branch": "orphan-branch"}],
    ):
        result = await list_worktrees()

    assert "Use cleanup_worktrees()" in result


@pytest.mark.asyncio
async def test_cleanup_worktrees_when_no_c2c_worktrees_exist(setup_session_manager):
    """Test cleanup when no c2c worktrees exist."""
    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[],
    ):
        result = await cleanup_worktrees()

    assert "No c2c worktrees found to clean up" in result


@pytest.mark.asyncio
async def test_cleanup_worktrees_removes_only_orphaned_by_default(
    setup_session_manager,
):
    """Test that cleanup_worktrees only removes orphaned worktrees by default."""
    # Create a tracked session
    create_result = await create_session(
        task="Tracked session",
        use_worktree=True,
    )

    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    session = setup_session_manager.sessions[session_id]
    tracked_path = str(session.worktree_path)
    orphan_path = str(setup_session_manager.repo_root) + "/.c2c/worktrees/c2c-orphan"

    # Track removal calls
    removed_paths = []

    def mock_remove(path, force=False):
        removed_paths.append(str(path))

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {"path": tracked_path, "branch": session.branch_name},
            {"path": orphan_path, "branch": "orphan-branch"},
        ],
    ), patch.object(
        setup_session_manager.worktree_manager,
        "remove_worktree",
        side_effect=mock_remove,
    ), patch.object(
        setup_session_manager.worktree_manager,
        "cleanup_branch",
    ):
        result = await cleanup_worktrees(orphaned_only=True)

    # Only orphan should be removed
    assert orphan_path in removed_paths
    assert tracked_path not in removed_paths
    assert "c2c-orphan" in result


@pytest.mark.asyncio
async def test_cleanup_worktrees_removes_all_when_orphaned_only_false(
    setup_session_manager,
):
    """Test that cleanup removes all worktrees when orphaned_only=False."""
    # Create a tracked session
    create_result = await create_session(
        task="Tracked session",
        use_worktree=True,
    )

    session_id = None
    for line in create_result.split("\n"):
        if "Session ID:" in line:
            session_id = line.split("Session ID:")[1].strip()
            break

    session = setup_session_manager.sessions[session_id]
    tracked_path = str(session.worktree_path)
    orphan_path = str(setup_session_manager.repo_root) + "/.c2c/worktrees/c2c-orphan"

    removed_paths = []

    def mock_remove(path, force=False):
        removed_paths.append(str(path))

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {"path": tracked_path, "branch": session.branch_name},
            {"path": orphan_path, "branch": "orphan-branch"},
        ],
    ), patch.object(
        setup_session_manager.worktree_manager,
        "remove_worktree",
        side_effect=mock_remove,
    ), patch.object(
        setup_session_manager.worktree_manager,
        "cleanup_branch",
    ):
        result = await cleanup_worktrees(orphaned_only=False)

    # Both should be removed
    assert tracked_path in removed_paths
    assert orphan_path in removed_paths


@pytest.mark.asyncio
async def test_cleanup_worktrees_removes_associated_branch(setup_session_manager):
    """Test that cleanup removes the branch associated with a worktree."""
    orphan_path = str(setup_session_manager.repo_root) + "/.c2c/worktrees/c2c-orphan"
    branch_name = "orphan-branch"

    removed_branches = []

    def mock_cleanup_branch(branch, force=False):
        removed_branches.append(branch)

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[{"path": orphan_path, "branch": branch_name}],
    ), patch.object(
        setup_session_manager.worktree_manager,
        "remove_worktree",
    ), patch.object(
        setup_session_manager.worktree_manager,
        "cleanup_branch",
        side_effect=mock_cleanup_branch,
    ):
        result = await cleanup_worktrees()

    assert branch_name in removed_branches


@pytest.mark.asyncio
async def test_cleanup_worktrees_shows_removed_worktrees_in_summary(
    setup_session_manager,
):
    """Test that cleanup shows removed worktrees in summary."""
    orphan_paths = [
        f"{setup_session_manager.repo_root}/.c2c/worktrees/c2c-orphan1",
        f"{setup_session_manager.repo_root}/.c2c/worktrees/c2c-orphan2",
    ]

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {"path": orphan_paths[0], "branch": "branch1"},
            {"path": orphan_paths[1], "branch": "branch2"},
        ],
    ), patch.object(
        setup_session_manager.worktree_manager,
        "remove_worktree",
    ), patch.object(
        setup_session_manager.worktree_manager,
        "cleanup_branch",
    ):
        result = await cleanup_worktrees()

    assert "✓ Removed 2 worktree(s)" in result
    assert "c2c-orphan1" in result
    assert "c2c-orphan2" in result


@pytest.mark.asyncio
async def test_cleanup_worktrees_shows_removed_branches_in_summary(
    setup_session_manager,
):
    """Test that cleanup shows removed branches in summary."""
    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[
            {
                "path": f"{setup_session_manager.repo_root}/.c2c/worktrees/c2c-orphan1",
                "branch": "orphan-branch-1",
            },
            {
                "path": f"{setup_session_manager.repo_root}/.c2c/worktrees/c2c-orphan2",
                "branch": "orphan-branch-2",
            },
        ],
    ), patch.object(
        setup_session_manager.worktree_manager,
        "remove_worktree",
    ), patch.object(
        setup_session_manager.worktree_manager,
        "cleanup_branch",
    ):
        result = await cleanup_worktrees()

    assert "✓ Deleted 2 branch(es)" in result
    assert "orphan-branch-1" in result
    assert "orphan-branch-2" in result


@pytest.mark.asyncio
async def test_cleanup_worktrees_handles_removal_errors_gracefully(
    setup_session_manager,
):
    """Test that cleanup handles removal errors without crashing."""
    orphan_path = f"{setup_session_manager.repo_root}/.c2c/worktrees/c2c-orphan"

    def mock_remove_with_error(path, force=False):
        raise Exception("Permission denied")

    with patch.object(
        setup_session_manager.worktree_manager,
        "list_worktrees",
        return_value=[{"path": orphan_path, "branch": "orphan-branch"}],
    ), patch.object(
        setup_session_manager.worktree_manager,
        "remove_worktree",
        side_effect=mock_remove_with_error,
    ):
        result = await cleanup_worktrees()

    assert "⚠ Errors" in result
    assert "c2c-orphan" in result
    assert "Permission denied" in result
