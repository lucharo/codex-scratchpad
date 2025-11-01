"""CLI entry point for c2c MCP server."""

import argparse
import sys
from pathlib import Path

from .server import main


def cli():
    """Command-line interface for c2c."""
    parser = argparse.ArgumentParser(
        description="c2c - Claude Code to Claude Code MCP Server"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository root path (default: current directory)",
    )

    args = parser.parse_args()

    try:
        main(args.repo_root)
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
