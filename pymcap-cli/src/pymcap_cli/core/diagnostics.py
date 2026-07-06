"""Accumulate ROS2 ``diagnostic_msgs/DiagnosticArray`` state.

Shared by ``pymcap-cli diag`` (file) and ``pymcap-cli bridge diag`` (live bridge)
so both build the same per-component view. Rich rendering lives in
:mod:`pymcap_cli.display.diag_render`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import re
    from collections.abc import Sequence

LEVEL_NAMES = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}
LEVEL_STYLES = {0: "green", 1: "yellow", 2: "red", 3: "dim"}
LEVEL_CHARS = {0: "▁", 1: "▃", 2: "▇", 3: "▅"}

DEFAULT_TOPICS = ["/diagnostics", "/diagnostics_agg"]


class _KeyValueLike(Protocol):
    key: str
    value: str


class _DiagnosticStatusLike(Protocol):
    name: str
    level: int
    hardware_id: str
    message: str
    values: Sequence[_KeyValueLike]


class DiagnosticArrayLike(Protocol):
    """A decoded ``diagnostic_msgs/msg/DiagnosticArray``."""

    status: Sequence[_DiagnosticStatusLike]


@dataclass
class DiagEntry:
    """Accumulated state for one diagnostic component."""

    name: str
    hardware_id: str
    worst_level: int
    last_level: int
    last_message: str
    count: int
    level_counts: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    first_timestamp_ns: int = 0
    last_timestamp_ns: int = 0
    latest_values: list[tuple[str, str]] = field(default_factory=list)
    level_changes: list[tuple[int, int, str]] = field(default_factory=list)
    level_durations_ns: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})


def add_diagnostic_message(
    entries: dict[str, DiagEntry],
    timestamp_ns: int,
    message: DiagnosticArrayLike,
) -> None:
    """Fold one decoded ``DiagnosticArray`` into per-component ``entries``."""
    for status in message.status:
        name = status.name
        level = int(status.level)
        values = [(kv.key, kv.value) for kv in status.values]

        if name not in entries:
            entries[name] = DiagEntry(
                name=name,
                hardware_id=status.hardware_id,
                worst_level=level,
                last_level=level,
                last_message=status.message,
                count=1,
                first_timestamp_ns=timestamp_ns,
                last_timestamp_ns=timestamp_ns,
                latest_values=values,
                level_changes=[(timestamp_ns, level, status.message)],
            )
            entries[name].level_counts[level] += 1
            continue

        entry = entries[name]
        # Accumulate time spent at the previous level before switching.
        entry.level_durations_ns[entry.last_level] += timestamp_ns - entry.last_timestamp_ns
        entry.count += 1
        entry.level_counts[level] += 1
        entry.worst_level = max(entry.worst_level, level)
        entry.last_timestamp_ns = timestamp_ns
        entry.latest_values = values
        if level != entry.last_level:
            entry.level_changes.append((timestamp_ns, level, status.message))
        entry.last_level = level
        entry.last_message = status.message


def level_totals(entries: dict[str, DiagEntry]) -> dict[int, int]:
    """Count components by worst level."""
    totals = {0: 0, 1: 0, 2: 0, 3: 0}
    for entry in entries.values():
        totals[entry.worst_level] += 1
    return totals


def compute_hz(entry: DiagEntry) -> float | None:
    """Average publish rate in Hz from first to last timestamp, or None."""
    if entry.count < 2:
        return None
    duration_s = (entry.last_timestamp_ns - entry.first_timestamp_ns) / 1e9
    if duration_s <= 0:
        return None
    return (entry.count - 1) / duration_s


def format_duration_ns(ns: int) -> str:
    """Format nanoseconds as a human-readable duration."""
    total_s = ns / 1e9
    if total_s < 1:
        return f"{total_s:.1f}s"
    total_s = int(total_s)
    if total_s < 60:
        return f"{total_s}s"
    minutes, seconds = divmod(total_s, 60)
    if minutes < 60:
        return f"{minutes}m{seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes}m"


def filter_entries(
    entries: dict[str, DiagEntry],
    *,
    min_level: int,
    name_pattern: re.Pattern[str] | None,
    hw_pattern: re.Pattern[str] | None,
) -> list[DiagEntry]:
    """Filter by level / name / hardware id, sorted worst-first then busiest."""
    result = []
    for entry in entries.values():
        if entry.worst_level < min_level:
            continue
        if name_pattern and not name_pattern.search(entry.name):
            continue
        if hw_pattern and not hw_pattern.search(entry.hardware_id):
            continue
        result.append(entry)
    result.sort(key=lambda e: (-e.worst_level, -e.count))
    return result
