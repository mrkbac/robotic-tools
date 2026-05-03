import logging
import os
from typing import ClassVar

from textual.app import App
from textual.binding import BindingType
from textual.logging import TextualHandler

from digitalis.screens.data import DataScreen


class DigitalisApp(App[None]):
    """MCAP Topic Browser app."""

    CSS_PATH = "app.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, file_or_url: str) -> None:
        super().__init__()
        self.file_or_url = file_or_url

        if os.environ.get("SSH_CONNECTION"):
            self._disable_tooltips = True

    def on_mount(self) -> None:
        self.push_screen(DataScreen(self.file_or_url))


def configure_logging() -> None:
    logging.basicConfig(level="NOTSET", handlers=[TextualHandler()])
