# discord-marimo-issue-bot

Discord bot that turns Discord support threads into actionable GitHub issues by summarising the conversation with an LLM and filing through the GitHub API.

## Features

- Single `/thread_issue` command that works only in threads and is restricted to server administrators
- Option to create the GitHub issue automatically or open a prefilled draft for verification
- Summaries powered by OpenAI with configurable rate limiting
- Resilient GitHub client with validation and retries
- Makefile, Dockerfile, and automated tests for quick development

## Requirements

- Python 3.11 (managed with [uv](https://docs.astral.sh/uv/))
- Discord application with a bot token and required intents
- GitHub personal access token with `repo` scope
- OpenAI API key

## Environment variables

Copy `.env.example` to `.env` and populate the secrets:

| Variable | Description |
| --- | --- |
| `DISCORD_TOKEN` | Discord bot token |
| `OPENAI_API_KEY` | OpenAI API key for summarisation |
| `GITHUB_TOKEN` | GitHub personal access token |
| `DEFAULT_REPO` | Repository in `owner/repo` format where issues are filed |
| `OPENAI_MODEL` | OpenAI model name (defaults to `gpt-4o-mini`) |
| `LLM_RATE_LIMIT` | Maximum LLM calls allowed per period |
| `LLM_RATE_PERIOD` | Period for the LLM rate limit in seconds |

## Installation

```bash
uv python install 3.11
uv init --app  # already run for this project
uv sync
```

## Running the bot

```bash
uv run python -m src.bot
```

or via the provided Makefile:

```bash
make run
```

## Deploying to Discord

1. Create a new application at <https://discord.com/developers/applications> and add a **Bot** user.
2. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent** so the bot can read thread history.
3. Generate an invite link from **OAuth2 → URL Generator** with the `bot` and `applications.commands` scopes. Grant at least the "Send Messages" permission.
4. Invite the bot to your Discord server using the generated link and ensure it has access to the channels that will host support threads.
5. Populate `.env` with the tokens for Discord, OpenAI, and GitHub plus `DEFAULT_REPO`.
6. Start the process (for example on a VM or container host) with `make run` or `uv run python -m src.bot`. Keep the process running via your preferred supervisor (systemd, Docker, etc.).
7. The first start automatically registers slash commands. Discord can take a minute to propagate them; once visible, use `/thread_issue` inside any thread where the bot is present.

### Docker deployment

```bash
docker build -t discord-marimo-issue-bot .
docker run --env-file .env --name issue-bot --restart unless-stopped discord-marimo-issue-bot
```

The image installs dependencies with `uv` and launches `python -m src.bot` inside the container. Mount a persistent volume if you later add on-disk storage.

## Slash command

- `/thread_issue mode:<auto|verify?> labels:<comma-separated?>`

The command must be invoked from within a Discord thread by a server administrator. When `mode` is omitted the bot files the issue immediately; choosing `verify` opens a prefilled GitHub issue draft for review in the caller's browser.

## Testing

```bash
make test
```

## Docker

To build and run inside Docker:

```bash
make docker-run
```

The Docker image uses `uv` to install dependencies and runs the bot with `uv run python -m src.bot`.
