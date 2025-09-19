import json
from datetime import datetime, timezone

import pytest

from src.llm import LLMError, RateLimiter, ThreadSummarizer
from src.models import ThreadMessage


class FakeCompletionClient:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, str]] = []

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self.response


@pytest.mark.asyncio
async def test_summarize_thread_returns_issue_draft() -> None:
    message = ThreadMessage(
        author="alice",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        content="The bot crashes when clicking the save button.",
        attachments=["https://example.com/image.png"],
    )

    draft_json = json.dumps(
        {
            "title": "Fix crash on save",
            "body_md": "### Problem\nApp crashes\n\n### Context (selected quotes)\n- \"quote\" — @alice, 2024-01-01\n\n### Proposed solution / acceptance criteria\n- [ ] fix it\n\n### Alternatives considered\nNone\n\n### Risks / open questions\nNone\n\n**Source:** https://discord.com/threads/1",
            "labels": ["bug"],
        }
    )

    client = FakeCompletionClient(draft_json)
    summarizer = ThreadSummarizer(client, RateLimiter(10, 1))

    draft = await summarizer.summarize_thread([message], "https://discord.com/threads/1")

    assert draft.title == "Fix crash on save"
    assert draft.labels == ["bug"]
    assert "https://discord.com/threads/1" in client.calls[0]["user"]


@pytest.mark.asyncio
async def test_summarize_thread_raises_on_invalid_json() -> None:
    message = ThreadMessage(
        author="alice",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        content="Example",
    )

    client = FakeCompletionClient("not-json")
    summarizer = ThreadSummarizer(client, RateLimiter(10, 1))

    with pytest.raises(LLMError):
        await summarizer.summarize_thread([message], None)
