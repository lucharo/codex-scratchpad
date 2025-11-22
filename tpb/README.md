# tpb

A research-focused terminal client for The Pirate Bay. The package provides:

- A Python API (`TPBClient`) for querying TPB mirrors (Tor by default)
- A JSON helper for easy data export
- A Textual TUI for interactive searching
- A Typer-powered CLI that can output tables or JSON

## Quickstart

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Search from the CLI (Tor on by default):

```bash
uv run tpb search "ubuntu" --limit 5 --json
```

Launch the TUI:

```bash
uv run tpb tui --query "ubuntu"
```

## Development

Run the test suite:

```bash
uv run pytest
```
