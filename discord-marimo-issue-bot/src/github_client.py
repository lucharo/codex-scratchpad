from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)

_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API indicates a failure."""

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class GitHubClient:
    """Minimal async client for creating GitHub issues with retries."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        timeout: float = 10.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport

    async def create_issue(
        self,
        owner_repo: str,
        title: str,
        body_md: str,
        labels: Sequence[str] | None = None,
    ) -> str:
        if not _REPO_PATTERN.match(owner_repo):
            raise ValueError("Repository must be in the format 'owner/repo'.")

        payload: dict[str, object] = {"title": title, "body": body_md}
        filtered_labels = [label for label in (labels or []) if label]
        if filtered_labels:
            payload["labels"] = filtered_labels

        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "discord-marimo-issue-bot",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout,
                    transport=self._transport,
                    headers=headers,
                ) as client:
                    response = await client.post(f"/repos/{owner_repo}/issues", json=payload)
            except httpx.HTTPError as exc:
                logger.warning("GitHub request failed on attempt %s: %s", attempt, exc)
                last_error = exc
            else:
                if 200 <= response.status_code < 300:
                    data = response.json()
                    issue_url = data.get("html_url") or data.get("url")
                    if not issue_url:
                        raise GitHubAPIError(response.status_code, "GitHub response missing issue URL")
                    return issue_url

                error_message = self._extract_error_message(response)
                api_error = GitHubAPIError(response.status_code, error_message)
                logger.warning(
                    "GitHub API returned %s on attempt %s: %s",
                    response.status_code,
                    attempt,
                    error_message,
                )

                if response.status_code < 500 or attempt == self._max_retries:
                    raise api_error

                last_error = api_error

            await asyncio.sleep(min(2 ** (attempt - 1), 5))

        raise GitHubAPIError(None, "Exceeded retry attempts while creating GitHub issue") from last_error

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text or "Unknown GitHub error"

        message = data.get("message") if isinstance(data, dict) else None
        if message:
            return message

        if isinstance(data, dict) and "errors" in data:
            return json.dumps(data["errors"])  # type: ignore[arg-type]

        return response.text or "Unknown GitHub error"
