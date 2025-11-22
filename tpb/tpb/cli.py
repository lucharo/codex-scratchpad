from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .client import TPBClient
from .exceptions import MirrorError, SearchError, VerificationError
from .json_utils import torrents_to_json
from .tui import TPBTuiApp

app = typer.Typer(help="Research-oriented terminal client for The Pirate Bay")


def _load_pgp_key(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def _print_table(rows):
    headers = ["Idx", "Title", "Seeders", "Leechers", "Size", "Uploader"]
    table = [headers]
    for idx, torrent in enumerate(rows, start=1):
        table.append(
            [
                str(idx),
                torrent.title,
                str(torrent.seeders),
                str(torrent.leechers),
                torrent.size,
                torrent.uploader,
            ]
        )

    column_widths = [max(len(row[col]) for row in table) for col in range(len(headers))]

    def format_row(row):
        return " | ".join(cell.ljust(column_widths[i]) for i, cell in enumerate(row))

    divider = "-+-".join("-" * width for width in column_widths)
    typer.echo(format_row(headers))
    typer.echo(divider)
    for row in table[1:]:
        typer.echo(format_row(row))


@app.command()
def search(
    query: str,
    limit: int = typer.Option(10, help="Maximum number of results to return"),
    tor: bool = typer.Option(True, "--tor/--no-tor", help="Use Tor for requests"),
    onion: bool = typer.Option(False, "--onion/--no-onion", help="Prefer the default onion mirror"),
    onion_mirror: Optional[str] = typer.Option(None, help="Override onion mirror URL"),
    pgp_strict: bool = typer.Option(
        False, "--pgp-strict/--no-pgp-strict", help="Require successful PGP verification for onions"
    ),
    pgp_key_file: Optional[Path] = typer.Option(None, help="Path to a PGP public key"),
    output_json: bool = typer.Option(False, "--json/--no-json", help="Emit JSON instead of a table"),
):
    """Search TPB mirrors for a query."""

    selected_onion = None
    if onion:
        selected_onion = onion_mirror or TPBClient.DEFAULT_ONION

    pgp_key = _load_pgp_key(pgp_key_file)

    client = TPBClient(
        use_tor=tor,
        onion_mirror=selected_onion,
        pgp_check=pgp_key is not None,
        pgp_public_key=pgp_key,
        allow_unverified_onion=not pgp_strict,
    )

    try:
        results = client.search(query=query, limit=limit)
    except (SearchError, MirrorError, VerificationError) as exc:
        raise typer.Exit(code=1) from exc

    if output_json:
        typer.echo(torrents_to_json(results))
    else:
        _print_table(results)


@app.command()
def tui(
    query: Optional[str] = typer.Option(None, help="Optional initial search query"),
    tor: bool = typer.Option(True, "--tor/--no-tor", help="Use Tor for requests"),
):
    """Launch the Textual TUI."""

    client = TPBClient(use_tor=tor)
    TPBTuiApp(client=client, initial_query=query).run()


def main():
    app()


if __name__ == "__main__":
    main()
