"""Data models for c2c session management."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SessionStatus(str, Enum):
    """Status of a Claude Code session."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class SessionConfig(BaseModel):
    """Configuration for a Claude Code session."""

    task: str = Field(..., description="Task description for the session")
    branch_name: Optional[str] = Field(
        None, description="Git branch name (auto-generated if not provided)"
    )
    use_worktree: bool = Field(
        True, description="Whether to use a git worktree for isolation"
    )
    timeout: Optional[int] = Field(
        None, description="Session timeout in seconds"
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the session"
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Tags for organizing and filtering sessions (e.g., feature, role, strategy)",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for custom use cases",
    )


class Session(BaseModel):
    """Represents a Claude Code session."""

    session_id: str = Field(..., description="Unique session identifier")
    config: SessionConfig = Field(..., description="Session configuration")
    status: SessionStatus = Field(
        default=SessionStatus.CREATED, description="Current session status"
    )
    worktree_path: Optional[Path] = Field(
        None, description="Path to git worktree if used"
    )
    branch_name: Optional[str] = Field(
        None, description="Git branch name for the session"
    )
    process_id: Optional[int] = Field(
        None, description="Process ID of Claude Code instance"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Session creation timestamp"
    )
    started_at: Optional[datetime] = Field(
        None, description="Session start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Session completion timestamp"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if session failed"
    )
    output: list[str] = Field(
        default_factory=list, description="Session output logs"
    )

    # Hierarchy tracking
    parent_session_id: Optional[str] = Field(
        None, description="Parent session ID if this is a sub-session"
    )
    child_session_ids: list[str] = Field(
        default_factory=list, description="List of child session IDs spawned by this session"
    )
    depth: int = Field(
        default=0, description="Depth in the session hierarchy (0 = root)"
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
        }
    )


class SessionSummary(BaseModel):
    """Summary information about a session."""

    session_id: str
    task: str
    status: SessionStatus
    branch_name: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )
