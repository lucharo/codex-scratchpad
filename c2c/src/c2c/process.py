"""Process management for Claude Code instances."""

import asyncio
import shutil
import signal
from pathlib import Path
from typing import Optional


class ProcessError(Exception):
    """Exception raised for process-related errors."""

    pass


class ClaudeCodeProcess:
    """Manages a single Claude Code process."""

    def __init__(
        self,
        working_dir: Path,
        task: str,
        env_vars: Optional[dict[str, str]] = None,
    ):
        """Initialize a Claude Code process.

        Args:
            working_dir: Working directory for the process
            task: Task to execute
            env_vars: Additional environment variables
        """
        self.working_dir = working_dir
        self.task = task
        self.env_vars = env_vars or {}
        self.process: Optional[asyncio.subprocess.Process] = None
        self.output_lines: list[str] = []

    async def start(self) -> int:
        """Start the Claude Code process.

        Returns:
            Process ID

        Raises:
            ProcessError: If the process fails to start
        """
        # Find claude executable
        claude_path = shutil.which("claude")
        if not claude_path:
            raise ProcessError(
                "Claude Code executable not found. "
                "Please ensure 'claude' is installed and in PATH."
            )

        # Prepare environment
        env = {**self.env_vars}

        try:
            # Start Claude Code with the task
            self.process = await asyncio.create_subprocess_exec(
                claude_path,
                "--task",
                self.task,
                "--non-interactive",
                cwd=self.working_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            if self.process.pid is None:
                raise ProcessError("Failed to start Claude Code process")

            # Start output collection in background
            asyncio.create_task(self._collect_output())

            return self.process.pid

        except Exception as e:
            raise ProcessError(f"Failed to start Claude Code process: {e}") from e

    async def _collect_output(self) -> None:
        """Collect output from the process."""
        if not self.process or not self.process.stdout:
            return

        try:
            async for line in self.process.stdout:
                decoded_line = line.decode("utf-8", errors="replace").rstrip()
                self.output_lines.append(decoded_line)
        except Exception:
            # Process terminated or output stream closed
            pass

    async def wait(self, timeout: Optional[int] = None) -> int:
        """Wait for the process to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Process exit code

        Raises:
            ProcessError: If the process is not running
            asyncio.TimeoutError: If timeout is reached
        """
        if not self.process:
            raise ProcessError("Process not started")

        try:
            if timeout:
                return await asyncio.wait_for(
                    self.process.wait(), timeout=timeout
                )
            else:
                return await self.process.wait()
        except asyncio.TimeoutError:
            # Timeout reached, terminate the process
            await self.terminate()
            raise

    async def terminate(self) -> None:
        """Terminate the process gracefully."""
        if not self.process:
            return

        try:
            # Send SIGTERM for graceful shutdown
            self.process.send_signal(signal.SIGTERM)

            # Wait up to 5 seconds for graceful shutdown
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill if still running
                self.process.kill()
                await self.process.wait()
        except ProcessLookupError:
            # Process already terminated
            pass

    async def kill(self) -> None:
        """Kill the process immediately."""
        if not self.process:
            return

        try:
            self.process.kill()
            await self.process.wait()
        except ProcessLookupError:
            # Process already terminated
            pass

    def is_running(self) -> bool:
        """Check if the process is currently running.

        Returns:
            True if running, False otherwise
        """
        return self.process is not None and self.process.returncode is None

    def get_output(self) -> list[str]:
        """Get all output lines from the process.

        Returns:
            List of output lines
        """
        return self.output_lines.copy()

    def get_exit_code(self) -> Optional[int]:
        """Get the process exit code.

        Returns:
            Exit code if process has terminated, None otherwise
        """
        return self.process.returncode if self.process else None
