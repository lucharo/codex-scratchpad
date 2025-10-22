"""Tests for process management."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cc2cc.process import ClaudeCodeProcess, ProcessError


@pytest.fixture
def temp_workdir():
    """Create a temporary working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_process_init(temp_workdir):
    """Test ClaudeCodeProcess initialization."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
        env_vars={"TEST_VAR": "test_value"},
    )

    assert process.working_dir == temp_workdir
    assert process.task == "Test task"
    assert process.env_vars == {"TEST_VAR": "test_value"}
    assert process.process is None
    assert process.output_lines == []


@pytest.mark.asyncio
async def test_process_start_no_claude(temp_workdir):
    """Test starting a process when claude is not available."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    with patch("shutil.which", return_value=None):
        with pytest.raises(ProcessError, match="Claude Code executable not found"):
            await process.start()


@pytest.mark.asyncio
async def test_process_start_mock(temp_workdir):
    """Test starting a process with mocked subprocess."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.stdout = AsyncMock()
    mock_process.stdout.__aiter__.return_value = iter([])

    with patch("shutil.which", return_value="/usr/bin/claude"):
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            pid = await process.start()

    assert pid == 12345
    assert process.process == mock_process


@pytest.mark.asyncio
async def test_process_is_running(temp_workdir):
    """Test checking if process is running."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    # Not started
    assert not process.is_running()

    # Mock running process
    mock_process = MagicMock()
    mock_process.returncode = None
    process.process = mock_process

    assert process.is_running()

    # Mock terminated process
    mock_process.returncode = 0
    assert not process.is_running()


@pytest.mark.asyncio
async def test_process_wait_not_started(temp_workdir):
    """Test waiting for a process that hasn't started."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    with pytest.raises(ProcessError, match="Process not started"):
        await process.wait()


@pytest.mark.asyncio
async def test_process_wait_mock(temp_workdir):
    """Test waiting for a process with mock."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    mock_process = AsyncMock()
    mock_process.wait.return_value = 0
    process.process = mock_process

    exit_code = await process.wait()

    assert exit_code == 0
    mock_process.wait.assert_called_once()


@pytest.mark.asyncio
async def test_process_wait_timeout(temp_workdir):
    """Test process timeout."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    mock_process = AsyncMock()
    mock_process.wait.side_effect = asyncio.TimeoutError()
    mock_process.send_signal = MagicMock()
    mock_process.kill = MagicMock()
    process.process = mock_process

    with pytest.raises(asyncio.TimeoutError):
        await process.wait(timeout=1)

    # Should attempt to terminate
    mock_process.send_signal.assert_called()


@pytest.mark.asyncio
async def test_process_terminate(temp_workdir):
    """Test terminating a process."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    mock_process = AsyncMock()
    mock_process.send_signal = MagicMock()
    mock_process.wait.return_value = 0
    process.process = mock_process

    await process.terminate()

    mock_process.send_signal.assert_called_once()
    mock_process.wait.assert_called()


@pytest.mark.asyncio
async def test_process_terminate_force_kill(temp_workdir):
    """Test force killing a process that won't terminate gracefully."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    mock_process = AsyncMock()
    mock_process.send_signal = MagicMock()
    mock_process.wait.side_effect = asyncio.TimeoutError()
    mock_process.kill = MagicMock()
    process.process = mock_process

    # Mock the second wait call to succeed
    mock_process.wait.side_effect = [asyncio.TimeoutError(), 0]

    await process.terminate()

    # Should send SIGTERM, timeout, then kill
    mock_process.send_signal.assert_called_once()
    mock_process.kill.assert_called_once()


@pytest.mark.asyncio
async def test_process_kill(temp_workdir):
    """Test killing a process immediately."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    mock_process = AsyncMock()
    mock_process.kill = MagicMock()
    mock_process.wait.return_value = -9
    process.process = mock_process

    await process.kill()

    mock_process.kill.assert_called_once()
    mock_process.wait.assert_called_once()


@pytest.mark.asyncio
async def test_process_get_output(temp_workdir):
    """Test getting process output."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    process.output_lines = ["line 1", "line 2", "line 3"]

    output = process.get_output()

    assert output == ["line 1", "line 2", "line 3"]
    # Should return a copy
    assert output is not process.output_lines


@pytest.mark.asyncio
async def test_process_get_exit_code(temp_workdir):
    """Test getting process exit code."""
    process = ClaudeCodeProcess(
        working_dir=temp_workdir,
        task="Test task",
    )

    # No process started
    assert process.get_exit_code() is None

    # Mock process with exit code
    mock_process = MagicMock()
    mock_process.returncode = 0
    process.process = mock_process

    assert process.get_exit_code() == 0
