.PHONY: run test docker-run

run:
uv run python -m src.bot

test:
uv run python -m pytest -q

docker-run:
docker build -t discord-marimo-issue-bot .
docker run --rm --env-file .env discord-marimo-issue-bot
