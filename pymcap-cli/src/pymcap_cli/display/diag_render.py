"""Rich rendering of accumulated diagnostics, shared by `diag` and `bridge diag`."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.diagnostics import (
    LEVEL_CHARS,
    LEVEL_NAMES,
    LEVEL_STYLES,
    DiagEntry,
    compute_hz,
    format_duration_ns,
)
from pymcap_cli.display.sparkline import sparkline


def _level_text(level: int) -> Text:
    return Text(LEVEL_NAMES.get(level, f"L{level}"), style=LEVEL_STYLES.get(level, "dim"))


def _format_timestamp(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / NS_TO_SEC).strftime("%H:%M:%S.%f")[:-3]


def _sparkline(entry: DiagEntry, width: int = 20) -> Text:
    changes = [(ts, lvl) for ts, lvl, _msg in entry.level_changes]
    return sparkline(
        changes,
        entry.last_timestamp_ns,
        char_map=LEVEL_CHARS,
        style_map=LEVEL_STYLES,
        width=width,
    )


def build_summary_table(
    filtered: list[DiagEntry],
    total: int,
    totals: dict[int, int],
    show_all: bool,
) -> Table:
    """Scannable one-row-per-component summary table."""
    parts = [f"Total: {total}"]
    parts.extend(
        f"[{LEVEL_STYLES[lvl]}]{LEVEL_NAMES[lvl]}: {totals.get(lvl, 0)}[/]" for lvl in (0, 1, 2, 3)
    )

    title = "Diagnostics Summary  " + " | ".join(parts)
    if not show_all and len(filtered) < total:
        title += (
            f"\n[dim]Showing {len(filtered)} components with issues (use --all to show all)[/dim]"
        )

    table = Table(
        title=title,
        show_header=True,
        box=None,
        padding=(0, 1),
        title_style="bold",
        expand=True,
    )
    table.add_column("Lvl", no_wrap=True, width=5)
    table.add_column("Cnt", justify="right", style="cyan", width=5)
    table.add_column("Hz", justify="right", width=6, style="dim")
    table.add_column("Timeline", no_wrap=True, width=20)
    table.add_column("Name", no_wrap=True, ratio=1, overflow="ellipsis")
    table.add_column("Message", no_wrap=True, ratio=1, overflow="ellipsis")

    for entry in filtered:
        hz = compute_hz(entry)
        table.add_row(
            _level_text(entry.worst_level),
            str(entry.count),
            f"{hz:.1f}" if hz is not None else "",
            _sparkline(entry),
            entry.name,
            entry.last_message,
        )

    return table


def build_tree_view(filtered: list[DiagEntry]) -> Tree:
    """Hierarchical view grouped by hardware_id."""
    root = Tree("[bold]Diagnostics[/bold]")

    by_hw: dict[str, list[DiagEntry]] = {}
    for entry in filtered:
        by_hw.setdefault(entry.hardware_id or "(no hardware_id)", []).append(entry)

    for hw_id, hw_entries in sorted(by_hw.items()):
        hw_node = root.add(f"[bold]{hw_id}[/bold]")
        for entry in hw_entries:
            level_style = LEVEL_STYLES.get(entry.worst_level, "dim")
            level_name = LEVEL_NAMES.get(entry.worst_level, "?")
            label = Text()
            label.append(f"[{level_name}]", style=level_style)
            label.append(f" {entry.name}", style="bold" if entry.worst_level >= 2 else "")
            if entry.last_message:
                label.append(f"  {entry.last_message}", style="dim")
            hw_node.add(label)

    return root


def build_inspect_view(entries: list[DiagEntry]) -> list[Table | Text]:
    """Detailed per-component views (distribution, timeline, latest values)."""
    renderables: list[Table | Text] = []

    for entry in entries:
        header = Text()
        header.append(f"\n{entry.name}", style="bold")
        if entry.hardware_id:
            header.append(f"  hw: {entry.hardware_id}", style="dim")
        renderables.append(header)

        dist = Text("  Levels: ")
        for lvl in (0, 1, 2, 3):
            count = entry.level_counts.get(lvl, 0)
            if count > 0:
                dist.append(f"{LEVEL_NAMES[lvl]}={count}  ", style=LEVEL_STYLES[lvl])
        dist.append(f"(total={entry.count})", style="dim")
        renderables.append(dist)

        hz = compute_hz(entry)
        if hz is not None:
            renderables.append(Text(f"  Frequency: {hz:.1f} Hz", style="cyan"))

        if any(v > 0 for v in entry.level_durations_ns.values()):
            dur = Text("  Time in state: ")
            for lvl in (0, 1, 2, 3):
                ns = entry.level_durations_ns.get(lvl, 0)
                if ns > 0:
                    dur.append(
                        f"{LEVEL_NAMES[lvl]}={format_duration_ns(ns)}  ", style=LEVEL_STYLES[lvl]
                    )
            renderables.append(dur)

        spark = _sparkline(entry, width=40)
        if spark.plain:
            renderables.append(Text.assemble("  Timeline: ", spark))

        if len(entry.level_changes) > 1:
            timeline = Table(
                title="Level Changes",
                show_header=True,
                box=None,
                padding=(0, 1),
                title_style="bold",
            )
            timeline.add_column("Time", no_wrap=True, style="dim")
            timeline.add_column("Level", no_wrap=True, width=6)
            timeline.add_column("Message")

            changes = entry.level_changes
            truncated = len(changes) > 50
            if truncated:
                changes = changes[:25] + changes[-25:]
            for idx, (ts, lvl, msg) in enumerate(changes):
                if truncated and idx == 25:
                    timeline.add_row("...", "...", f"({len(entry.level_changes) - 50} more)")
                timeline.add_row(_format_timestamp(ts), _level_text(lvl), msg)
            renderables.append(timeline)

        if entry.latest_values:
            kv_table = Table(
                title="Latest Values",
                show_header=True,
                box=None,
                padding=(0, 1),
                title_style="bold",
            )
            kv_table.add_column("Key", style="cyan", no_wrap=True)
            kv_table.add_column("Value")
            for key, value in entry.latest_values:
                kv_table.add_row(key, value)
            renderables.append(kv_table)

    return renderables


def build_json_output(
    filtered: list[DiagEntry],
    total: int,
    totals: dict[int, int],
) -> dict[str, Any]:
    """JSON-serializable representation of the filtered components."""
    return {
        "summary": {
            "total_components": total,
            "level_counts": {LEVEL_NAMES[k]: v for k, v in sorted(totals.items())},
        },
        "components": [
            {
                "name": e.name,
                "hardware_id": e.hardware_id,
                "worst_level": e.worst_level,
                "worst_level_name": LEVEL_NAMES.get(e.worst_level, f"L{e.worst_level}"),
                "last_level": e.last_level,
                "last_level_name": LEVEL_NAMES.get(e.last_level, f"L{e.last_level}"),
                "count": e.count,
                "level_counts": {LEVEL_NAMES[k]: v for k, v in sorted(e.level_counts.items())},
                "last_message": e.last_message,
                "frequency_hz": compute_hz(e),
                "level_durations_s": {
                    LEVEL_NAMES[k]: round(v / 1e9, 2)
                    for k, v in sorted(e.level_durations_ns.items())
                    if v > 0
                },
                "values": dict(e.latest_values),
                "level_changes": [
                    {
                        "timestamp_ns": ts,
                        "level": lvl,
                        "level_name": LEVEL_NAMES.get(lvl, f"L{lvl}"),
                        "message": msg,
                    }
                    for ts, lvl, msg in e.level_changes
                ],
            }
            for e in filtered
        ],
    }
