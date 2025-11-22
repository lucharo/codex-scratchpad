from __future__ import annotations

from typing import Optional

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from .client import TPBClient
from .exceptions import MirrorError, SearchError
from .models import Torrent


class TPBTuiApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #status {
        height: 1;
    }
    #controls {
        height: 3;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("a", "share_selected", "Print selected magnets"),
        Binding("enter", "show_magnet", "Show magnet"),
        Binding("space", "toggle_selection", "Toggle selection"),
    ]

    def __init__(self, client: TPBClient, initial_query: Optional[str] = None, limit: int = 15):
        super().__init__()
        self.client = client
        self.initial_query = initial_query
        self.limit = limit
        self._results: list[Torrent] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="controls"):
            yield Input(placeholder="Enter search query", id="query")
            yield Button("Search", id="search")
        yield Static("Ready", id="status")
        table = DataTable(id="results")
        table.add_columns("Idx", "Title", "Seeders", "Leechers", "Size", "Uploader")
        table.cursor_type = "row"
        yield table
        yield Footer()

    async def on_mount(self) -> None:
        if self.initial_query:
            query_widget = self.query_one("#query", Input)
            query_widget.value = self.initial_query
            await self.perform_search(self.initial_query)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search":
            query = self.query_one("#query", Input).value
            await self.perform_search(query)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query":
            await self.perform_search(event.value)

    async def action_show_magnet(self) -> None:
        table = self.query_one(DataTable)
        if table.cursor_row is None:
            return
        magnet = self._results[table.cursor_row].magnet
        self.query_one("#status", Static).update(f"Selected magnet: {magnet}")
        print(magnet)

    async def action_share_selected(self) -> None:
        table = self.query_one(DataTable)
        if not table.selected_rows:
            return
        magnets = [self._results[row].magnet for row in table.selected_rows]
        output = "\n".join(magnets)
        self.query_one("#status", Static).update(f"Shared {len(magnets)} magnets to stdout")
        print(output)

    async def action_toggle_selection(self) -> None:
        table = self.query_one(DataTable)
        if table.cursor_row is not None:
            table.toggle_row(table.cursor_row)

    async def perform_search(self, query: str) -> None:
        status = self.query_one("#status", Static)
        status.update(f"Searching for '{query}' ...")
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("Idx", "Title", "Seeders", "Leechers", "Size", "Uploader")
        try:
            self._results = self.client.search(query, limit=self.limit)
        except (SearchError, MirrorError) as exc:
            status.update(f"Search failed: {exc}")
            return

        for idx, torrent in enumerate(self._results, start=1):
            table.add_row(
                str(idx),
                torrent.title,
                str(torrent.seeders),
                str(torrent.leechers),
                torrent.size,
                torrent.uploader,
            )
        status.update(f"Loaded {len(self._results)} results")

    async def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            await self.action_quit()
