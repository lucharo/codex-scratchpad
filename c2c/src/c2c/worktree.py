"""Git worktree management for isolated Claude Code sessions."""

import subprocess
from pathlib import Path
from typing import Optional


class WorktreeError(Exception):
    """Exception raised for worktree-related errors."""

    pass


class WorktreeManager:
    """Manages git worktrees for isolated Claude Code sessions."""

    def __init__(self, repo_root: Path):
        """Initialize the worktree manager.

        Args:
            repo_root: Path to the git repository root
        """
        self.repo_root = repo_root
        self._validate_git_repo()

    def _validate_git_repo(self) -> None:
        """Validate that the path is a git repository."""
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"Not a git repository: {self.repo_root}"
            ) from e

    def _run_git_command(
        self, args: list[str], cwd: Optional[Path] = None
    ) -> str:
        """Run a git command and return the output.

        Args:
            args: Git command arguments
            cwd: Working directory for the command

        Returns:
            Command output as string

        Raises:
            WorktreeError: If the command fails
        """
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd or self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"Git command failed: {' '.join(args)}\n{e.stderr}"
            ) from e

    def create_worktree(
        self, branch_name: str, worktree_path: Path
    ) -> Path:
        """Create a new git worktree with a new branch.

        Args:
            branch_name: Name of the branch to create
            worktree_path: Path where the worktree should be created

        Returns:
            Path to the created worktree

        Raises:
            WorktreeError: If worktree creation fails
        """
        if worktree_path.exists():
            raise WorktreeError(f"Path already exists: {worktree_path}")

        # Create worktree with new branch from current HEAD
        self._run_git_command(
            ["worktree", "add", "-b", branch_name, str(worktree_path)]
        )

        return worktree_path

    def remove_worktree(self, worktree_path: Path, force: bool = False) -> None:
        """Remove a git worktree.

        Args:
            worktree_path: Path to the worktree to remove
            force: Force removal even if there are uncommitted changes

        Raises:
            WorktreeError: If worktree removal fails
        """
        args = ["worktree", "remove", str(worktree_path)]
        if force:
            args.append("--force")

        self._run_git_command(args)

    def list_worktrees(self) -> list[dict[str, str]]:
        """List all git worktrees.

        Returns:
            List of worktree information dictionaries
        """
        output = self._run_git_command(["worktree", "list", "--porcelain"])

        worktrees = []
        current_worktree = {}

        for line in output.split("\n"):
            if not line:
                if current_worktree:
                    worktrees.append(current_worktree)
                    current_worktree = {}
                continue

            if line.startswith("worktree "):
                current_worktree["path"] = line.split(" ", 1)[1]
            elif line.startswith("branch "):
                current_worktree["branch"] = line.split(" ", 1)[1]
            elif line.startswith("HEAD "):
                current_worktree["head"] = line.split(" ", 1)[1]

        if current_worktree:
            worktrees.append(current_worktree)

        return worktrees

    def worktree_exists(self, worktree_path: Path) -> bool:
        """Check if a worktree exists at the given path.

        Args:
            worktree_path: Path to check

        Returns:
            True if worktree exists, False otherwise
        """
        worktrees = self.list_worktrees()
        return any(
            wt.get("path") == str(worktree_path) for wt in worktrees
        )

    def get_current_branch(self) -> str:
        """Get the current branch name in the repository.

        Returns:
            Current branch name
        """
        return self._run_git_command(["branch", "--show-current"])

    def cleanup_branch(self, branch_name: str, force: bool = False) -> None:
        """Delete a branch.

        Args:
            branch_name: Name of the branch to delete
            force: Force deletion even if not fully merged

        Raises:
            WorktreeError: If branch deletion fails
        """
        flag = "-D" if force else "-d"
        self._run_git_command(["branch", flag, branch_name])
