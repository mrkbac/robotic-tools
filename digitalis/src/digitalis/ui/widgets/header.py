"""Custom header widget with integrated connection status."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from rich.text import Text
from textual.reactive import Reactive
from textual.widget import Widget

from digitalis.reader.source import SourceStatus

if TYPE_CHECKING:
    from textual.app import ComposeResult, RenderResult
    from textual.events import Click, Mount


class CustomHeaderIcon(Widget):
    """Display an 'icon' on the left of the header with command palette functionality."""

    DEFAULT_CSS = """
    CustomHeaderIcon {
        dock: left;
        padding: 0 1;
        width: 8;
        content-align: left middle;
    }

    CustomHeaderIcon:hover {
        background: $foreground 10%;
    }
    """

    icon = Reactive("⭘")
    """The character to use as the icon within the header."""

    def on_mount(self) -> None:
        if self.app.ENABLE_COMMAND_PALETTE:
            self.tooltip = "Open the command palette"
        else:
            self.disabled = True

    async def on_click(self, event: Click) -> None:
        """Launch the command palette when icon is clicked."""
        event.stop()
        await self.run_action("app.command_palette")

    def render(self) -> RenderResult:
        """Render the header icon.

        Returns:
            The rendered icon.
        """
        return self.icon


class CustomHeaderTitle(Widget):
    """Display the title / subtitle in the header."""

    DEFAULT_CSS = """
    CustomHeaderTitle {
        content-align: center middle;
        width: 100%;
    }
    """

    text: Reactive[str] = Reactive("")
    """The main title text."""

    sub_text = Reactive("")
    """The sub-title text."""

    def render(self) -> RenderResult:
        """Render the title and sub-title.

        Returns:
            The value to render.
        """
        text = Text(self.text, no_wrap=True, overflow="ellipsis")
        if self.sub_text:
            text.append(" — ")
            text.append(self.sub_text, "dim")
        return text


class CustomHeaderStatus(Widget):
    """Display connection status on the right of the header."""

    DEFAULT_CSS = """
    CustomHeaderStatus {
        dock: right;
        width: 16;
        padding: 0 1;
        content-align: right middle;
    }

    CustomHeaderStatus.connecting {
        color: $warning;
        text-style: blink;
    }

    CustomHeaderStatus.connected {
        color: $success;
    }

    CustomHeaderStatus.reconnecting {
        color: $warning;
        text-style: blink;
    }

    CustomHeaderStatus.disconnected {
        color: $error 50%;
    }

    CustomHeaderStatus.error {
        color: $error;
    }

    CustomHeaderStatus.initializing {
        color: $text 50%;
    }
    """

    status: Reactive[SourceStatus] = Reactive(SourceStatus.INITIALIZING)
    """The current connection status."""

    def watch_status(self, status: SourceStatus) -> None:
        """Update CSS classes when status changes."""
        # Remove all status classes
        self.remove_class(
            "connecting", "connected", "disconnected", "error", "reconnecting", "initializing"
        )

        # Add appropriate class based on status
        status_class_map = {
            SourceStatus.CONNECTING: "connecting",
            SourceStatus.CONNECTED: "connected",
            SourceStatus.RECONNECTING: "reconnecting",
            SourceStatus.DISCONNECTED: "disconnected",
            SourceStatus.ERROR: "error",
            SourceStatus.INITIALIZING: "initializing",
            SourceStatus.READY: "connected",
        }

        css_class = status_class_map.get(status, "error")
        self.add_class(css_class)

    def render(self) -> RenderResult:
        """Render the connection status.

        Returns:
            The rendered status.
        """
        status_info = self._get_status_info(self.status)
        return Text(f"{status_info['symbol']} {status_info['text']}", no_wrap=True)

    def _get_status_info(self, status: SourceStatus) -> dict[str, str]:
        """Get display information for a status."""
        status_map = {
            SourceStatus.INITIALIZING: {"symbol": "○", "text": "Initializing"},
            SourceStatus.CONNECTING: {"symbol": "◐", "text": "Connecting"},
            SourceStatus.CONNECTED: {"symbol": "●", "text": "Connected"},
            SourceStatus.RECONNECTING: {"symbol": "◑", "text": "Reconnecting"},
            SourceStatus.DISCONNECTED: {"symbol": "○", "text": "Disconnected"},
            SourceStatus.ERROR: {"symbol": "⚠", "text": "Error"},
            SourceStatus.READY: {"symbol": "●", "text": "Ready"},
        }

        return status_map.get(status, {"symbol": "?", "text": "Unknown"})


class CustomHeader(Widget):
    """A simplified header widget with icon, title, and connection status."""

    DEFAULT_CSS = """
    CustomHeader {
        dock: top;
        width: 100%;
        background: $primary;
        color: $foreground;
        height: 1;
    }
    """

    icon: Reactive[str] = Reactive("⭘")
    """A character for the icon at the top left."""

    status: Reactive[SourceStatus] = Reactive(SourceStatus.INITIALIZING)
    """The current connection status."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        icon: str | None = None,
    ) -> None:
        """Initialise the custom header widget.

        Args:
            name: The name of the header widget.
            id: The ID of the header widget in the DOM.
            classes: The CSS classes of the header widget.
            icon: Single character to use as an icon, or `None` for default.
        """
        super().__init__(name=name, id=id, classes=classes)
        if icon is not None:
            self.icon = icon

    def compose(self) -> ComposeResult:
        yield CustomHeaderIcon().data_bind(CustomHeader.icon)
        yield CustomHeaderTitle()
        yield CustomHeaderStatus().data_bind(CustomHeader.status)

    @property
    def screen_title(self) -> str:
        """The title that this header will display.

        This depends on Screen.title and App.title.
        """
        screen_title = self.screen.title
        return screen_title if screen_title is not None else self.app.title

    @property
    def screen_sub_title(self) -> str:
        """The sub-title that this header will display.

        This depends on Screen.sub_title and App.sub_title.
        """
        screen_sub_title = self.screen.sub_title
        return screen_sub_title if screen_sub_title is not None else self.app.sub_title

    def _on_mount(self, _: Mount) -> None:
        async def set_title() -> None:
            with contextlib.suppress(Exception):
                self.query_one(CustomHeaderTitle).text = self.screen_title

        async def set_sub_title() -> None:
            with contextlib.suppress(Exception):
                self.query_one(CustomHeaderTitle).sub_text = self.screen_sub_title

        self.watch(self.app, "title", set_title)
        self.watch(self.app, "sub_title", set_sub_title)
        self.watch(self.screen, "title", set_title)
        self.watch(self.screen, "sub_title", set_sub_title)
