"""Tests for worktree management."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from cc2cc.worktree import WorktreeError, WorktreeManager


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


def test_worktree_manager_init(temp_git_repo):
    """Test WorktreeManager initialization."""
    manager = WorktreeManager(temp_git_repo)
    assert manager.repo_root == temp_git_repo


def test_worktree_manager_init_non_git_repo():
    """Test WorktreeManager initialization with non-git directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(WorktreeError, match="Not a git repository"):
            WorktreeManager(Path(tmpdir))


def test_create_worktree(temp_git_repo):
    """Test creating a worktree."""
    manager = WorktreeManager(temp_git_repo)
    worktree_path = temp_git_repo / "worktrees" / "test-branch"

    result = manager.create_worktree("test-branch", worktree_path)

    assert result == worktree_path
    assert worktree_path.exists()
    assert (worktree_path / "README.md").exists()


def test_create_worktree_existing_path(temp_git_repo):
    """Test creating a worktree at an existing path."""
    manager = WorktreeManager(temp_git_repo)
    worktree_path = temp_git_repo / "worktrees" / "test-branch"
    worktree_path.mkdir(parents=True)

    with pytest.raises(WorktreeError, match="Path already exists"):
        manager.create_worktree("test-branch", worktree_path)


def test_list_worktrees(temp_git_repo):
    """Test listing worktrees."""
    manager = WorktreeManager(temp_git_repo)

    # Initially only the main worktree
    worktrees = manager.list_worktrees()
    assert len(worktrees) >= 1
    assert any(wt.get("path") == str(temp_git_repo) for wt in worktrees)

    # Create a new worktree
    worktree_path = temp_git_repo / "worktrees" / "test-branch"
    manager.create_worktree("test-branch", worktree_path)

    # Now should have two worktrees
    worktrees = manager.list_worktrees()
    assert len(worktrees) >= 2
    assert any(wt.get("path") == str(worktree_path) for wt in worktrees)


def test_worktree_exists(temp_git_repo):
    """Test checking if worktree exists."""
    manager = WorktreeManager(temp_git_repo)
    worktree_path = temp_git_repo / "worktrees" / "test-branch"

    assert not manager.worktree_exists(worktree_path)

    manager.create_worktree("test-branch", worktree_path)

    assert manager.worktree_exists(worktree_path)


def test_remove_worktree(temp_git_repo):
    """Test removing a worktree."""
    manager = WorktreeManager(temp_git_repo)
    worktree_path = temp_git_repo / "worktrees" / "test-branch"

    manager.create_worktree("test-branch", worktree_path)
    assert worktree_path.exists()

    manager.remove_worktree(worktree_path)
    assert not manager.worktree_exists(worktree_path)


def test_remove_worktree_force(temp_git_repo):
    """Test force removing a worktree with uncommitted changes."""
    manager = WorktreeManager(temp_git_repo)
    worktree_path = temp_git_repo / "worktrees" / "test-branch"

    manager.create_worktree("test-branch", worktree_path)

    # Make uncommitted changes
    test_file = worktree_path / "new_file.txt"
    test_file.write_text("test content")

    # Force remove should work
    manager.remove_worktree(worktree_path, force=True)
    assert not manager.worktree_exists(worktree_path)


def test_get_current_branch(temp_git_repo):
    """Test getting current branch."""
    manager = WorktreeManager(temp_git_repo)

    # Default branch (master or main)
    branch = manager.get_current_branch()
    assert branch in ("master", "main")


def test_cleanup_branch(temp_git_repo):
    """Test cleaning up a branch."""
    manager = WorktreeManager(temp_git_repo)
    worktree_path = temp_git_repo / "worktrees" / "test-branch"

    # Create and remove worktree
    manager.create_worktree("test-branch", worktree_path)
    manager.remove_worktree(worktree_path)

    # Clean up the branch
    manager.cleanup_branch("test-branch", force=True)

    # Verify branch is gone
    result = subprocess.run(
        ["git", "branch", "--list", "test-branch"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert "test-branch" not in result.stdout
