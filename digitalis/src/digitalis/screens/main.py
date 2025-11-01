from textual import events
from textual.app import ComposeResult
from textual.containers import HorizontalGroup, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)

from digitalis.screens.data import DataScreen
from digitalis.utilities import get_file_paths


class MainScreen(Screen[None]):
    DEFAULT_CSS = """
    MainScreen {
        align: center middle;
        padding: 0;
    }

    #title {
        width: auto;
    }
    #recent-files {
        padding: 2 2;
        Input {
            padding: 0 0;
            width: 1fr;
            margin: 0 0 5 0;
        }
    }
    .center {
        text-align: center;
    }
"""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            r"""  ____  _       _ _        _ _
 |  _ \(_) __ _(_) |_ __ _| (_)___
 | | | | |/ _` | | __/ _` | | / __|
 | |_| | | (_| | | || (_| | | \__ \
 |____/|_|\__, |_|\__\__,_|_|_|___/
          |___/""",
            id="title",
        )
        with Vertical(id="recent-files"):
            with HorizontalGroup():
                yield Input(
                    placeholder="Path or url to open",
                    id="topic-search",
                    classes="search",
                )
                yield Button(
                    "Open",
                    id="open-button",
                    variant="primary",
                    classes="search",
                )
            yield Static("[b]Recently Opened Files[/b]", classes="center")
            yield ListView(
                ListItem(Label("[b]ws://somewebsocket:8765[/b]\n2023-10-01")),
                ListItem(Label("Two\n2023-10-02")),
                ListItem(Label("Three\n2023-10-03")),
            )

    def on_paste(self, event: events.Paste) -> None:
        """Handle paste events to return to the previous screen."""
        for path in get_file_paths(event.text):
            if path.suffix == ".mcap":
                self.app.push_screen(DataScreen(str(path)))
                return
