from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import ContentSwitcher, Static

from digitalis.reader.types import MessageEvent, Topic
from digitalis.ui.panels import BasePanel, get_all_panels, get_available_panels


class DataView(Widget, can_focus=True):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(p.__name__[0].lower(), f"switch_panel('{p.__name__}')", p.__name__)
        for p in get_all_panels()
    ]

    data: reactive[MessageEvent | str | None] = reactive(None)
    topic: reactive[Topic | None] = reactive(None)

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._mounted_panel: BasePanel | None = None
        self.available_panels: list[type[BasePanel]] = []

    def compose(self) -> ComposeResult:
        """Compose the DataView widget."""
        with ContentSwitcher():
            yield Static("N/A", id="raw")
            yield Container(id="data")

    async def watch_topic(self, topic: Topic | None) -> None:
        """Handle topic changes by removing old panel and resetting state."""

        new_panel = None
        if topic is not None and topic.schema_name is not None:
            panels = get_available_panels(topic.schema_name)
            if not panels:
                self.available_panels = []
                self.refresh_bindings()
                return
            self.available_panels = panels
            self.refresh_bindings()
            new_panel = panels[0]

        await self.action_switch_panel(new_panel, notify=False)

    def watch_data(self, data: MessageEvent | str | None) -> None:
        """Update the displayed data when the data changes."""
        if not self.is_mounted:
            return
        switcher = self.query_one(ContentSwitcher)
        static = self.query_one("#raw", Static)
        if data is None:
            static.update("No data received")
            switcher.current = "raw"
            return

        if isinstance(data, str):
            static.update(data)
            switcher.current = "raw"
            return

        schema_name = data.schema_name
        if schema_name is None:
            static.update(f"Unsupported schema: {schema_name}")
            switcher.current = "raw"
            return

        if self._mounted_panel:
            switcher.current = "data"
            self._mounted_panel.data = data

    async def action_switch_panel(
        self, new_panel_type: type[BasePanel] | str | None, *, notify: bool = True
    ) -> None:
        """Switch to a new_panel_type panel type."""

        if isinstance(new_panel_type, str):
            new_panel_type = next(
                (p for p in self.available_panels if p.__name__ == new_panel_type), None
            )

        if new_panel_type is not None and not issubclass(new_panel_type, BasePanel):
            msg = f"Invalid panel type: {new_panel_type}"
            raise ValueError(msg)

        if self._mounted_panel:
            await self._mounted_panel.remove()
            self._mounted_panel = None

        if new_panel_type:
            container = self.query_one("#data", Container)
            new_panel = new_panel_type()
            await container.mount(new_panel)

            if notify and isinstance(self.data, MessageEvent):
                new_panel.data = self.data

            self._mounted_panel = new_panel

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if the action is valid for this widget."""

        if action == "switch_panel":
            if not parameters or len(parameters) != 1 or not isinstance(parameters[0], str):
                return False
            param: str = parameters[0]
            available_names = [p.__name__ for p in self.available_panels]
            return param in available_names
        return True
