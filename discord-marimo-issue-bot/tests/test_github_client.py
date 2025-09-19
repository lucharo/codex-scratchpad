import json

import httpx
import pytest

from src.github_client import GitHubAPIError, GitHubClient


@pytest.mark.asyncio
async def test_create_issue_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/test/repo/issues"
        assert request.headers["Authorization"] == "token test-token"
        payload = json.loads(request.content.decode())
        assert payload["title"] == "Test issue"
        assert payload["labels"] == ["bug"]
        return httpx.Response(201, json={"html_url": "https://github.com/test/repo/issues/1"})

    client = GitHubClient("test-token", transport=httpx.MockTransport(handler), max_retries=1)

    issue_url = await client.create_issue("test/repo", "Test issue", "Body", ["bug"])

    assert issue_url == "https://github.com/test/repo/issues/1"


@pytest.mark.asyncio
async def test_create_issue_invalid_repo() -> None:
    client = GitHubClient("token")

    with pytest.raises(ValueError):
        await client.create_issue("invalid repo", "Title", "Body", [])


@pytest.mark.asyncio
async def test_create_issue_github_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    client = GitHubClient(
        "test-token",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )

    with pytest.raises(GitHubAPIError) as exc_info:
        await client.create_issue("test/repo", "Test issue", "Body", [])

    assert exc_info.value.status_code == 401
