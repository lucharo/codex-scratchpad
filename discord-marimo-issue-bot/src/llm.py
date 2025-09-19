from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Protocol, Sequence

from openai import AsyncOpenAI
from pydantic import ValidationError

from .models import IssueDraft, ThreadMessage

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails or returns invalid output."""


class CompletionClient(Protocol):
    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - protocol definition
        """Return the raw string response from the language model."""


class RateLimiter:
    """A cooperative rate limiter suitable for async contexts."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self._max_calls = max_calls
        self._period = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return

                sleep_for = self._period - (now - self._timestamps[0])

            await asyncio.sleep(max(sleep_for, 0))


class OpenAICompletionClient:
    """Wrapper around the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.2) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover - network errors exercised in runtime
            raise LLMError("Failed to call OpenAI completions API") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive
            raise LLMError("OpenAI response did not contain any choices") from exc

        if not content:
            raise LLMError("OpenAI returned an empty response")

        return content


SYSTEM_PROMPT = "You convert Discord support threads into concise, actionable GitHub issues."

USER_TEMPLATE = """INSTRUCTIONS:
- Extract the core problem and suggested solution.
- Neutral, factual tone; no promises or hype.
- Deduplicate repetition, collapse long quotes.
- ≤600 words total.
- Output Markdown in the exact section order below.
- Suggest 1–3 labels if obvious.

RETURN FIELDS:
- title (≤80 chars, imperative)
- body_md (Markdown)
- labels (list[str])

SECTIONS FORMAT:
### Problem
…

### Context (selected quotes)
- "quote…" — @user, YYYY-MM-DD

### Proposed solution / acceptance criteria
- [ ] testable bullet
- [ ] …

### Alternatives considered
…

### Risks / open questions
…

**Source:** <DISCORD_THREAD_URL>

THREAD MESSAGES JSON:
{messages_json}

Respond with a JSON object containing the keys "title", "body_md", and "labels" (list of unique strings)."""


class ThreadSummarizer:
    """Generate issue drafts from Discord thread histories."""

    def __init__(
        self,
        completion_client: CompletionClient,
        rate_limiter: RateLimiter,
    ) -> None:
        self._client = completion_client
        self._rate_limiter = rate_limiter

    async def summarize_thread(
        self,
        messages: Sequence[ThreadMessage],
        thread_url: str | None,
    ) -> IssueDraft:
        await self._rate_limiter.acquire()

        messages_json = self._messages_to_json(messages)
        user_prompt = USER_TEMPLATE.format(messages_json=messages_json).replace(
            "<DISCORD_THREAD_URL>", thread_url or "(thread link unavailable)"
        )

        raw = await self._client.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return self._parse_response(raw)

    @staticmethod
    def _messages_to_json(messages: Sequence[ThreadMessage]) -> str:
        payload = [message.to_prompt_dict() for message in messages]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_response(raw: str) -> IssueDraft:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug("LLM raw output: %s", raw)
            raise LLMError("LLM response was not valid JSON") from exc

        labels = data.get("labels", [])
        if isinstance(labels, list):
            data["labels"] = list(dict.fromkeys(label.strip() for label in labels if isinstance(label, str) and label.strip()))

        try:
            return IssueDraft.model_validate(data)
        except ValidationError as exc:
            logger.debug("LLM parsed output: %s", data)
            raise LLMError("LLM response did not match the IssueDraft schema") from exc
