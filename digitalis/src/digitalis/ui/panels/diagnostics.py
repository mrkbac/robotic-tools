from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, Protocol

from rich.highlighter import ISO8601Highlighter, ReprHighlighter
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalGroup
from textual.reactive import reactive
from textual.widgets import Input, Select, Tree
from textual.widgets.tree import TreeNode

from digitalis.reader.types import MessageEvent
from digitalis.ui.panels.base import BasePanel
from digitalis.utilities import NANOSECONDS_PER_SECOND, nanoseconds_to_iso


class KeyValue(Protocol):
    key: str
    value: str


class DiagnosticStatus(Protocol):
    level: int
    name: str
    message: str
    hardware_id: str
    values: list[KeyValue]


class DiagnosticArray(Protocol):
    """Protocol for diagnostic array messages."""

    header: Any
    status: list[dict[str, Any]]


class DiagnosticLevel(IntEnum):
    OK = 0
    WARN = 1
    ERROR = 2
    STALE = 3


_COLOR_MAP = {
    DiagnosticLevel.OK: "green",
    DiagnosticLevel.WARN: "yellow",
    DiagnosticLevel.ERROR: "red",
    DiagnosticLevel.STALE: "blue",
}

STALE_TIME_NANOSECONDS = 5 * NANOSECONDS_PER_SECOND


@dataclass
class StaleStatusWrapper:
    status: DiagnosticStatus
    is_stale: bool
    timestamp_ns: int


def _build_status(label: str, wrap: StaleStatusWrapper) -> Text:
    level = wrap.status.level
    if level in DiagnosticLevel:
        lvl = DiagnosticLevel(level)
        color = _COLOR_MAP[DiagnosticLevel.STALE] if wrap.is_stale else _COLOR_MAP[lvl]
        name = lvl.name
    else:
        color = "gray"
        name = f"UKN{level}"
    return Text.from_markup(f"[{color}]{name:5}[/{color}] [b]{label}[/b]")


class DiagTree(Tree):
    data: reactive[dict[str, StaleStatusWrapper]] = reactive({})

    highlighter = ReprHighlighter()
    iso_highlighter = ISO8601Highlighter()

    def __init__(self) -> None:
        super().__init__("root")
        self.show_root = False

    def watch_data(self, data: dict[str, StaleStatusWrapper]) -> None:
        _node_map: dict[str, TreeNode] = {
            kv.data: kv for kv in self.root.children if isinstance(kv.data, str)
        }  # pyright: ignore[reportAssignmentType]

        stale_keys = set(_node_map.keys()) - set(data.keys())
        for stale_key in stale_keys:
            stale_node = _node_map[stale_key]
            stale_node.remove()

        for key, wrap in data.items():
            if key in _node_map:
                parent_node = _node_map[key]
                parent_node.set_label(_build_status(key, wrap))

                self._update_child_status(parent_node, wrap)
            else:
                parent_node = self.root.add(_build_status(key, wrap), data=key)
                self._update_child_status(parent_node, wrap)

    def _update_child_status(self, parent_node: TreeNode, wrap: StaleStatusWrapper) -> None:
        # Build expected children: message first, then values in input order
        expected: dict[str, Text] = {
            "__message__": Text.assemble(
                f"{wrap.status.message} @ ",
                self.iso_highlighter(nanoseconds_to_iso(wrap.timestamp_ns)),
            )
        }

        expected.update(
            (value.key, Text.assemble(f"{value.key}=", self.highlighter(value.value)))
            for value in wrap.status.values
        )

        # Get current children by their stable keys
        current = {child.data: child for child in parent_node.children if child.data}
        current_keys = list(current.keys())  # dict ordering is stable
        expected_keys = list(expected.keys())

        # If structure matches, just update labels
        if current_keys == expected_keys:
            for key, label in expected.items():
                if str(current[key].label) != label:
                    current[key].set_label(label)
        else:
            # TODO: Reuse existing nodes if possible
            parent_node.remove_children()
            for key, label in expected.items():
                parent_node.add_leaf(label, data=key)


class Diagnostic(BasePanel[DiagnosticArray]):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {
        "diagnostic_msgs/msg/DiagnosticArray",  # ROS2
        "diagnostic_msgs/DiagnosticArray",  # ROS1
    }

    diag_data: reactive[dict[str, StaleStatusWrapper]] = reactive({})
    filter_level: reactive[DiagnosticLevel | None] = reactive(None)
    filter_text: reactive[str] = reactive("")
    _last_channel_id: str | None = None
    _newest_timestamp: int
    _pending_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._newest_timestamp = 0
        self.set_interval(1.0, self._validate_stale)  # Check staleness every second
        self.set_interval(0.05, self._flush_updates)  # Update UI at 20Hz

    DEFAULT_CSS = """
    Diagnostic {
        Select {
            width: 10;
        }
    }
"""

    def compose(self) -> ComposeResult:
        with HorizontalGroup():
            yield Select(
                [
                    ("ALL", None),
                    *[
                        (f"[{_COLOR_MAP.get(member, 'gray')}]{member.name}[/]", member.value)
                        for member in DiagnosticLevel
                    ],
                ],
                prompt="LogLevel",
                value=None,
                compact=True,
                allow_blank=False,
            )
            yield Input(compact=True, placeholder="Search")

        yield DiagTree()

    def _validate_stale(self) -> None:
        """Periodically check for stale diagnostics (runs every second)."""
        if self._newest_timestamp == 0:
            return

        stale_threshold = self._newest_timestamp - STALE_TIME_NANOSECONDS
        updated = False

        for key, wrap in self.diag_data.items():
            if wrap.timestamp_ns < stale_threshold and not wrap.is_stale:
                self.diag_data[key].is_stale = True
                updated = True

        if updated:
            self._pending_update = True

    def _update_child(self) -> None:
        diag = self.query_one(DiagTree)
        if self.filter_level is not None or self.filter_text:
            filtered_data = {
                k: v
                for k, v in self.diag_data.items()
                if (self.filter_level is None or v.status.level == self.filter_level)
                and (
                    not self.filter_text
                    or (
                        self.filter_text in v.status.name.lower()
                        or self.filter_text in v.status.message.lower()
                        or self.filter_text in v.status.hardware_id.lower()
                    )
                )
            }
            diag.data = filtered_data
        else:
            diag.data = self.diag_data
        diag.mutate_reactive(DiagTree.data)

    def _flush_updates(self) -> None:
        """Periodically flush pending updates to the UI (throttled at 20Hz)."""
        if self._pending_update:
            self._update_child()
            self._pending_update = False

    def watch_data(self, data: MessageEvent | None) -> None:
        if data is None:
            return
        if data.topic != self._last_channel_id:
            # TODO: cache this?
            self.diag_data.clear()
            self._newest_timestamp = 0
            self._last_channel_id = data.topic

        # Update newest timestamp
        self._newest_timestamp = max(self._newest_timestamp, data.timestamp_ns)

        for status in data.message.status:
            key = f"{status.hardware_id}: {status.name}"
            self.diag_data[key] = StaleStatusWrapper(
                status=status,
                is_stale=False,
                timestamp_ns=data.timestamp_ns,
            )

        # Mark that we need to update UI (will happen in next flush)
        self._pending_update = True

    @on(Select.Changed)
    def log_level_change(self, event: Select.Changed) -> None:
        self.filter_level = DiagnosticLevel(event.value) if isinstance(event.value, int) else None  # pyright: ignore[reportAttributeAccessIssue]
        self._update_child()

    @on(Input.Changed)
    def search_change(self, event: Input.Changed) -> None:
        self.filter_text = event.value.lower()
        self._update_child()
