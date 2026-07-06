"""Tests for the shared diagnostics accumulation helpers."""

from __future__ import annotations

import dataclasses
from typing import Any

from pymcap_cli.core.diagnostics import (
    DiagEntry,
    add_diagnostic_message,
    compute_hz,
    filter_entries,
    level_totals,
)


def _mk(name: str, slots: list[str]) -> type:
    return dataclasses.make_dataclass(name, [(s, Any) for s in slots], slots=True)


def _status(
    name: str,
    level: int,
    *,
    message: str = "",
    hardware_id: str = "",
    values: tuple[tuple[str, str], ...] = (),
) -> Any:
    kv_cls = _mk("KeyValue", ["key", "value"])
    status_cls = _mk("DiagnosticStatus", ["name", "level", "hardware_id", "message", "values"])
    return status_cls(
        name=name,
        level=level,
        hardware_id=hardware_id,
        message=message,
        values=[kv_cls(key=k, value=v) for k, v in values],
    )


def _array(*statuses: Any) -> Any:
    array_cls = _mk("DiagnosticArray", ["status"])
    return array_cls(status=list(statuses))


def test_add_diagnostic_message_creates_entry() -> None:
    entries: dict[str, DiagEntry] = {}
    add_diagnostic_message(
        entries,
        1_000,
        _array(_status("motor", 1, message="warm", hardware_id="hw0", values=(("temp", "80"),))),
    )
    entry = entries["motor"]
    assert entry.worst_level == 1
    assert entry.last_level == 1
    assert entry.count == 1
    assert entry.hardware_id == "hw0"
    assert entry.last_message == "warm"
    assert entry.latest_values == [("temp", "80")]
    assert entry.level_changes == [(1_000, 1, "warm")]


def test_add_diagnostic_message_accumulates_and_tracks_changes() -> None:
    entries: dict[str, DiagEntry] = {}
    add_diagnostic_message(entries, 1_000_000_000, _array(_status("m", 1, message="warn")))
    add_diagnostic_message(entries, 2_000_000_000, _array(_status("m", 2, message="err")))
    entry = entries["m"]
    assert entry.count == 2
    assert entry.worst_level == 2
    assert entry.last_level == 2
    # A level change was recorded, and time spent at the previous level accrued.
    assert entry.level_changes == [(1_000_000_000, 1, "warn"), (2_000_000_000, 2, "err")]
    assert entry.level_durations_ns[1] == 1_000_000_000


def test_add_diagnostic_message_no_change_does_not_append_change() -> None:
    entries: dict[str, DiagEntry] = {}
    add_diagnostic_message(entries, 1_000, _array(_status("m", 0)))
    add_diagnostic_message(entries, 2_000, _array(_status("m", 0)))
    assert len(entries["m"].level_changes) == 1
    assert entries["m"].count == 2


def test_level_totals_counts_by_worst_level() -> None:
    entries: dict[str, DiagEntry] = {}
    add_diagnostic_message(
        entries, 1_000, _array(_status("a", 0), _status("b", 2), _status("c", 2))
    )
    assert level_totals(entries) == {0: 1, 1: 0, 2: 2, 3: 0}


def test_filter_entries_min_level_and_sort() -> None:
    entries: dict[str, DiagEntry] = {}
    add_diagnostic_message(
        entries, 1_000, _array(_status("ok", 0), _status("warn", 1), _status("err", 2))
    )
    filtered = filter_entries(entries, min_level=1, name_pattern=None, hw_pattern=None)
    # OK dropped; worst-level-first ordering puts ERROR before WARN.
    assert [e.name for e in filtered] == ["err", "warn"]


def test_compute_hz_needs_two_samples() -> None:
    entries: dict[str, DiagEntry] = {}
    add_diagnostic_message(entries, 0, _array(_status("m", 0)))
    assert compute_hz(entries["m"]) is None
    add_diagnostic_message(entries, 1_000_000_000, _array(_status("m", 0)))
    # Two samples one second apart → 1 Hz.
    assert compute_hz(entries["m"]) == 1.0
