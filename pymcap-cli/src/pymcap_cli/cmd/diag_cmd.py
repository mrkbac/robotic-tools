"""Diag command - inspect ROS2 diagnostics from MCAP files."""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from small_mcap.reader import include_topics, read_message_decoded

from pymcap_cli.input_handler import open_input

console = Console()
console_err = Console(stderr=True)

LEVEL_NAMES = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}
LEVEL_STYLES = {0: "green", 1: "yellow", 2: "red", 3: "dim"}

FILTERING_GROUP = Group("Filtering")
DISPLAY_GROUP = Group("Display")


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


def _collect_diagnostics(file: str, topic: str) -> dict[str, DiagEntry]:
    """Stream all diagnostics messages and accumulate per-component state."""
    entries: dict[str, DiagEntry] = {}
    msg_count = 0

    with (
        open_input(file) as (f, file_size),
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.fields[msgs]} msgs"),
            TextColumn("[dim]{task.fields[components]} components"),
            TimeElapsedColumn(),
            console=console_err,
            transient=True,
        ) as progress,
    ):
        task = progress.add_task(
            "Scanning diagnostics...",
            total=file_size or None,
            msgs=0,
            components=0,
        )

        for msg in read_message_decoded(
            f,
            should_include=include_topics([topic]),
            decoder_factories=[DecoderFactory()],
        ):
            msg_count += 1
            timestamp_ns = msg.message.log_time

            for status in msg.decoded_message.status:
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
                else:
                    entry = entries[name]
                    entry.count += 1
                    entry.level_counts[level] += 1
                    entry.worst_level = max(entry.worst_level, level)
                    entry.last_timestamp_ns = timestamp_ns
                    entry.latest_values = values

                    if level != entry.last_level:
                        entry.level_changes.append((timestamp_ns, level, status.message))

                    entry.last_level = level
                    entry.last_message = status.message

            if msg_count % 100 == 0:
                progress.update(
                    task,
                    completed=f.tell(),
                    msgs=msg_count,
                    components=len(entries),
                )

        progress.update(task, completed=file_size, msgs=msg_count, components=len(entries))

    return entries


def _level_text(level: int) -> Text:
    """Create a styled Text for a diagnostic level."""
    name = LEVEL_NAMES.get(level, f"L{level}")
    style = LEVEL_STYLES.get(level, "dim")
    return Text(name, style=style)


def _format_timestamp(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000).strftime("%H:%M:%S.%f")[:-3]


def _filter_entries(
    entries: dict[str, DiagEntry],
    *,
    min_level: int,
    name_pattern: str | None,
    hw_pattern: str | None,
) -> list[DiagEntry]:
    """Filter and sort entries."""
    result = []
    for entry in entries.values():
        if entry.worst_level < min_level:
            continue
        if name_pattern and not re.search(name_pattern, entry.name, re.IGNORECASE):
            continue
        if hw_pattern and not re.search(hw_pattern, entry.hardware_id, re.IGNORECASE):
            continue
        result.append(entry)

    result.sort(key=lambda e: (-e.worst_level, -e.count))
    return result


def _build_summary_table(
    filtered: list[DiagEntry],
    total: int,
    level_totals: dict[int, int],
    show_all: bool,
) -> Table:
    """Build the summary Rich Table."""
    parts = [f"Total: {total}"]
    for lvl in (0, 1, 2, 3):
        name = LEVEL_NAMES[lvl]
        style = LEVEL_STYLES[lvl]
        count = level_totals.get(lvl, 0)
        parts.append(f"[{style}]{name}: {count}[/{style}]")

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
    table.add_column("Name", no_wrap=True, ratio=1, overflow="ellipsis")
    table.add_column("Message", no_wrap=True, ratio=1, overflow="ellipsis")

    for entry in filtered:
        level_txt = _level_text(entry.worst_level)
        table.add_row(
            level_txt,
            str(entry.count),
            entry.name,
            entry.last_message,
        )

    return table


def _build_tree_view(filtered: list[DiagEntry]) -> Tree:
    """Build a hierarchical tree grouped by hardware_id."""
    root = Tree("[bold]Diagnostics[/bold]")

    by_hw: dict[str, list[DiagEntry]] = {}
    for entry in filtered:
        hw = entry.hardware_id or "(no hardware_id)"
        by_hw.setdefault(hw, []).append(entry)

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


def _build_inspect_view(entries: list[DiagEntry]) -> list[Table | Text]:
    """Build detailed inspection views for matching components."""
    renderables: list[Table | Text] = []

    for entry in entries:
        # Header
        header = Text()
        header.append(f"\n{entry.name}", style="bold")
        if entry.hardware_id:
            header.append(f"  hw: {entry.hardware_id}", style="dim")
        renderables.append(header)

        # Level distribution
        dist = Text("  Levels: ")
        for lvl in (0, 1, 2, 3):
            count = entry.level_counts.get(lvl, 0)
            if count > 0:
                style = LEVEL_STYLES[lvl]
                name = LEVEL_NAMES[lvl]
                dist.append(f"{name}={count}  ", style=style)
        dist.append(f"(total={entry.count})", style="dim")
        renderables.append(dist)

        # Timeline of level changes
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

            # Show up to 50 transitions
            changes = entry.level_changes
            truncated = len(changes) > 50
            if truncated:
                changes = changes[:25] + changes[-25:]

            for idx, (ts, lvl, msg) in enumerate(changes):
                if truncated and idx == 25:
                    timeline.add_row("...", "...", f"({len(entry.level_changes) - 50} more)")
                timeline.add_row(_format_timestamp(ts), _level_text(lvl), msg)

            renderables.append(timeline)

        # Latest key-value pairs
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


def _build_json_output(
    filtered: list[DiagEntry],
    total: int,
    level_totals: dict[int, int],
) -> dict:
    """Build JSON-serializable output."""
    return {
        "summary": {
            "total_components": total,
            "level_counts": {LEVEL_NAMES[k]: v for k, v in sorted(level_totals.items())},
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


def diag(
    file: str,
    *,
    level: Annotated[
        int | None,
        Parameter(
            name=["-l", "--level"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    show_all: Annotated[
        bool,
        Parameter(
            name=["-a", "--all"],
            group=FILTERING_GROUP,
        ),
    ] = False,
    name: Annotated[
        str | None,
        Parameter(
            name=["-n", "--name"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    hardware_id: Annotated[
        str | None,
        Parameter(
            name=["--hardware-id", "--hw"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    inspect: Annotated[
        str | None,
        Parameter(
            name=["-i", "--inspect"],
            group=DISPLAY_GROUP,
        ),
    ] = None,
    tree: Annotated[
        bool,
        Parameter(
            name=["--tree"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    topic: Annotated[
        str,
        Parameter(
            name=["-t", "--topic"],
            group=FILTERING_GROUP,
        ),
    ] = "/diagnostics",
    json_output: Annotated[
        bool,
        Parameter(
            name=["--json"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
) -> int:
    """Inspect ROS2 diagnostics from an MCAP file.

    Reads diagnostic_msgs/msg/DiagnosticArray messages and provides a scannable
    overview of system health. By default shows only components with issues
    (WARN, ERROR, STALE).

    Examples:
      # Show components with issues
      pymcap-cli diag recording.mcap

      # Show all components including OK
      pymcap-cli diag recording.mcap --all

      # Show only ERROR and STALE
      pymcap-cli diag recording.mcap --level 2

      # Search by component name
      pymcap-cli diag recording.mcap --name "encoder"

      # Inspect a specific component in detail
      pymcap-cli diag recording.mcap --inspect "encoder"

      # Hierarchical tree view
      pymcap-cli diag recording.mcap --tree

      # JSON output for scripting
      pymcap-cli diag recording.mcap --json

    Parameters
    ----------
    file
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    level
        Minimum diagnostic level to show (0=OK, 1=WARN, 2=ERROR, 3=STALE).
        Default: 1 (show WARN and above).
    show_all
        Show all components including OK.
    name
        Regex filter on component name (case-insensitive).
    hardware_id
        Regex filter on hardware ID (case-insensitive).
    inspect
        Show detailed view for components matching this regex.
    tree
        Display as hierarchical tree instead of flat table.
    topic
        Diagnostics topic name. Common alternatives: /diagnostics_agg.
    json_output
        Output as JSON for scripting.
    """
    try:
        entries = _collect_diagnostics(file, topic)
    except (OSError, ValueError, RuntimeError) as e:
        console_err.print(f"[red]Error reading MCAP file: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console_err.print("\n[yellow]Interrupted by user[/yellow]")
        return 0

    if not entries:
        console_err.print(f"[yellow]No diagnostics found on topic '{topic}'[/yellow]")
        return 0

    level_totals: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for entry in entries.values():
        level_totals[entry.worst_level] += 1

    if show_all:
        min_level = 0
    elif level is not None:
        min_level = level
    else:
        min_level = 1

    filtered = _filter_entries(
        entries, min_level=min_level, name_pattern=name, hw_pattern=hardware_id
    )

    if json_output:
        output = _build_json_output(filtered, len(entries), level_totals)
        print(json.dumps(output, indent=2), file=sys.stdout)  # noqa: T201
        return 0

    if inspect:
        # Try with current filters first, fall back to ignoring level filter
        matched = [e for e in filtered if re.search(inspect, e.name, re.IGNORECASE)]
        if not matched:
            matched = _filter_entries(
                entries, min_level=0, name_pattern=inspect, hw_pattern=hardware_id
            )
        if not matched:
            console_err.print(f"[yellow]No components matching '{inspect}'[/yellow]")
            return 0

        for renderable in _build_inspect_view(matched):
            console.print(renderable)
        return 0

    if tree:
        tree_widget = _build_tree_view(filtered)
        console.print(tree_widget)
        return 0

    table = _build_summary_table(filtered, len(entries), level_totals, show_all)
    console.print(table)
    return 0
