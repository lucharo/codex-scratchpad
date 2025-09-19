from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import List, Sequence
from urllib.parse import urlencode

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from .github_client import GitHubAPIError, GitHubClient
from .llm import LLMError, OpenAICompletionClient, RateLimiter, ThreadSummarizer
from .models import Settings, ThreadMessage
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def parse_labels(raw: str | None) -> List[str]:
    if not raw:
        return []
    parts = [label.strip() for label in raw.split(",")]
    return [label for label in parts if label]


async def gather_thread_messages(thread: discord.Thread) -> List[ThreadMessage]:
    messages: List[ThreadMessage] = []
    async for message in thread.history(limit=None, oldest_first=True):
        author = "Unknown user"
        if message.author:
            author = message.author.display_name or message.author.name

        timestamp = message.created_at or datetime.now(timezone.utc)

        attachments = [attachment.url for attachment in message.attachments]

        messages.append(
            ThreadMessage(
                author=author,
                timestamp=timestamp,
                content=message.content or "",
                attachments=attachments,
            )
        )

    return messages


class IssueBot(commands.Bot):
    def __init__(
        self,
        settings: Settings,
        summarizer: ThreadSummarizer,
        github_client: GitHubClient,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.settings = settings
        self.summarizer = summarizer
        self.github_client = github_client

    async def setup_hook(self) -> None:
        await self.tree.sync()
        logger.info("Slash commands synced")


async def ensure_thread(interaction: discord.Interaction) -> discord.Thread | None:
    channel = interaction.channel
    if isinstance(channel, discord.Thread):
        return channel

    if interaction.response.is_done():
        await interaction.followup.send(
            "This command only works inside a Discord thread.", ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "This command only works inside a Discord thread.", ephemeral=True
        )
    return None


def is_admin(interaction: discord.Interaction) -> bool:
    user = interaction.user
    if isinstance(user, discord.Member):
        return user.guild_permissions.administrator
    return False


async def require_admin(interaction: discord.Interaction) -> bool:
    if is_admin(interaction):
        return True

    message = "Only administrators can run this command."
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)
    return False


def build_github_draft_url(owner_repo: str, title: str, body_md: str, labels: Sequence[str]) -> str:
    params = {"title": title, "body": body_md}
    filtered_labels = [label for label in labels if label]
    if filtered_labels:
        params["labels"] = ",".join(filtered_labels)
    return f"https://github.com/{owner_repo}/issues/new?{urlencode(params)}"


def create_bot() -> IssueBot:
    load_dotenv()
    configure_logging()

    settings = Settings.from_env()

    summarizer = ThreadSummarizer(
        completion_client=OpenAICompletionClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        ),
        rate_limiter=RateLimiter(settings.llm_rate_limit, settings.llm_rate_period),
    )

    github_client = GitHubClient(token=settings.github_token)

    bot = IssueBot(settings, summarizer, github_client)

    @bot.event
    async def on_ready() -> None:
        logger.info("Logged in as %s", bot.user)

    @bot.tree.command(
        name="thread_issue",
        description="Summarize this thread and create a GitHub issue",
    )
    @app_commands.describe(
        mode="Choose 'verify' to open a draft in your browser before filing",
        labels="Optional comma-separated labels to append",
    )
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Auto create issue", value="auto"),
            app_commands.Choice(name="Verify in browser", value="verify"),
        ]
    )
    async def create_issue(
        interaction: discord.Interaction,
        mode: app_commands.Choice[str] | None = None,
        labels: str | None = None,
    ) -> None:
        thread = await ensure_thread(interaction)
        if not thread:
            return

        if not await require_admin(interaction):
            return

        verify_before_posting = mode is not None and mode.value == "verify"

        await interaction.response.defer(thinking=True, ephemeral=verify_before_posting)

        messages = await gather_thread_messages(thread)

        try:
            issue_draft = await bot.summarizer.summarize_thread(messages, thread.jump_url)
        except LLMError:
            await interaction.followup.send(
                "I couldn't summarize this thread right now. Please try again later.",
                ephemeral=True,
            )
            return

        labels_from_command = parse_labels(labels)

        combined_labels: List[str] = list(
            dict.fromkeys([*issue_draft.labels, *labels_from_command])
        )

        if verify_before_posting:
            draft_url = build_github_draft_url(
                bot.settings.issue_repo,
                issue_draft.title,
                issue_draft.body_md,
                combined_labels,
            )
            view = discord.ui.View()
            view.add_item(
                discord.ui.Button(label="Open draft on GitHub", url=draft_url)
            )

            labels_text = ", ".join(combined_labels) if combined_labels else "none"
            content = f"**{issue_draft.title}**\nLabels: {labels_text}"

            if len(issue_draft.body_md) > 1800:
                file = discord.File(
                    io.StringIO(issue_draft.body_md), filename="issue_draft.md"
                )
                await interaction.followup.send(
                    content=content,
                    file=file,
                    view=view,
                    ephemeral=True,
                )
            else:
                await interaction.followup.send(
                    content=f"{content}\n\n{issue_draft.body_md}",
                    view=view,
                    ephemeral=True,
                )
            return

        try:
            issue_url = await bot.github_client.create_issue(
                bot.settings.issue_repo,
                issue_draft.title,
                issue_draft.body_md,
                combined_labels,
            )
        except ValueError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except GitHubAPIError as exc:
            await interaction.followup.send(
                "GitHub rejected the issue. Please verify repository access and try again.",
                ephemeral=True,
            )
            logger.error("GitHub API error (%s): %s", exc.status_code, exc.message)
            return

        await interaction.followup.send(
            f"Created issue in `{bot.settings.issue_repo}`: {issue_url}",
            ephemeral=False,
        )

    return bot


def main() -> None:
    bot = create_bot()
    bot.run(bot.settings.discord_token)


if __name__ == "__main__":
    main()
