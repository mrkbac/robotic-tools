import argparse
import os
from typing import ClassVar

if os.environ.get("SSH_CONNECTION"):
    # Set default, overwriting still works this way
    os.environ.setdefault("TEXTUAL_FPS", "5")
    os.environ.setdefault("TEXTUAL_ANIMATIONS", "none")


import logging

from textual.app import App
from textual.binding import BindingType
from textual.logging import TextualHandler

logging.basicConfig(
    level="NOTSET",
    handlers=[TextualHandler()],
)
# This needs to be set before importing the Screens


from digitalis.screens.data import DataScreen  # noqa: E402


class DigitalisApp(App[None]):
    """MCAP Topic Browser app."""

    CSS_PATH = "app.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, file_or_url: str) -> None:
        super().__init__()
        self.file_or_url = file_or_url

        # Disable tooltips when connected via SSH to reduce bandwidth
        if os.environ.get("SSH_CONNECTION"):
            self._disable_tooltips = True

    def on_mount(self) -> None:
        self.push_screen(DataScreen(self.file_or_url))


def app() -> DigitalisApp:
    parser = argparse.ArgumentParser(description="Digitalis - MCAP Topic Browser")
    parser.add_argument(
        "file_or_url",
        help="Path to MCAP file or WebSocket URL to browse",
        default="ws://localhost:8765",
    )

    args = parser.parse_args()
    return DigitalisApp(args.file_or_url)


def main() -> None:
    app().run()


if __name__ == "__main__":
    main()
