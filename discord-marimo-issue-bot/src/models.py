from __future__ import annotations

import os
from datetime import datetime
import re
from typing import List

from pydantic import BaseModel, Field, ValidationError, field_validator


class IssueDraft(BaseModel):
    """Structured representation of a GitHub issue draft."""

    title: str = Field(..., min_length=1, max_length=200)
    body_md: str = Field(..., min_length=1)
    labels: List[str] = Field(default_factory=list)


class ThreadMessage(BaseModel):
    """Message captured from a Discord thread."""

    author: str
    timestamp: datetime
    content: str = ""
    attachments: List[str] = Field(default_factory=list)

    def to_prompt_dict(self) -> dict:
        """Return a serialisable representation for the LLM prompt."""

        return {
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "attachments": self.attachments,
        }


_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    discord_token: str = Field(alias="DISCORD_TOKEN")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    github_token: str = Field(alias="GITHUB_TOKEN")
    issue_repo: str = Field(alias="DEFAULT_REPO")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_rate_limit: int = Field(default=5, alias="LLM_RATE_LIMIT", ge=1)
    llm_rate_period: int = Field(default=60, alias="LLM_RATE_PERIOD", ge=1)

    @field_validator("issue_repo")
    @classmethod
    def validate_repo_format(cls, value: str) -> str:
        if not _REPO_PATTERN.match(value):
            raise ValueError("DEFAULT_REPO must be in the format 'owner/repo'.")
        return value

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables.

        Raises:
            RuntimeError: If required configuration is missing or invalid.
        """

        data = {field.alias: os.getenv(field.alias) for field in cls.model_fields.values()}

        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - exercised in runtime
            raise RuntimeError("Invalid configuration: missing or malformed environment variables") from exc
