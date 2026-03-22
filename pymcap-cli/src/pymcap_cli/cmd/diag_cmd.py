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

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.sparkline import sparkline

console = Console()
console_err = Console(stderr=True)

LEVEL_NAMES = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}
LEVEL_STYLES = {0: "green", 1: "yellow", 2: "red", 3: "dim"}
LEVEL_CHARS = {0: "▁", 1: "▃", 2: "▇", 3: "▅"}

DEFAULT_TOPICS = ["/diagnostics", "/diagnostics_agg"]

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
    level_durations_ns: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})


def _collect_diagnostics(file: str, topics: list[str]) -> dict[str, DiagEntry]:
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
            should_include=include_topics(topics),
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
                    # Accumulate time spent at previous level
                    dt = timestamp_ns - entry.last_timestamp_ns
                    entry.level_durations_ns[entry.last_level] += dt

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


def _format_duration_ns(ns: int) -> str:
    """Format nanoseconds as human-readable duration."""
    total_s = ns / 1_000_000_000
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


def _compute_hz(entry: DiagEntry) -> float | None:
    """Compute average publish rate in Hz from first to last timestamp."""
    if entry.count < 2:
        return None
    duration_s = (entry.last_timestamp_ns - entry.first_timestamp_ns) / 1e9
    if duration_s <= 0:
        return None
    return (entry.count - 1) / duration_s


def _sparkline(entry: DiagEntry, width: int = 20) -> Text:
    """Build a colored sparkline of level changes over time."""
    changes = [(ts, lvl) for ts, lvl, _msg in entry.level_changes]
    return sparkline(
        changes,
        entry.last_timestamp_ns,
        char_map=LEVEL_CHARS,
        style_map=LEVEL_STYLES,
        width=width,
    )


def _compile_pattern(pattern: str, flag_name: str) -> re.Pattern[str]:
    """Compile a regex pattern with a user-friendly error on invalid syntax."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        console_err.print(f"[red]Invalid regex for {flag_name}: {e}[/red]")
        raise SystemExit(1) from e


def _filter_entries(
    entries: dict[str, DiagEntry],
    *,
    min_level: int,
    name_pattern: re.Pattern[str] | None,
    hw_pattern: re.Pattern[str] | None,
) -> list[DiagEntry]:
    """Filter and sort entries."""
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
    table.add_column("Hz", justify="right", width=6, style="dim")
    table.add_column("Timeline", no_wrap=True, width=20)
    table.add_column("Name", no_wrap=True, ratio=1, overflow="ellipsis")
    table.add_column("Message", no_wrap=True, ratio=1, overflow="ellipsis")

    for entry in filtered:
        hz = _compute_hz(entry)
        hz_str = f"{hz:.1f}" if hz is not None else ""
        table.add_row(
            _level_text(entry.worst_level),
            str(entry.count),
            hz_str,
            _sparkline(entry),
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

        # Frequency
        hz = _compute_hz(entry)
        if hz is not None:
            renderables.append(Text(f"  Frequency: {hz:.1f} Hz", style="cyan"))

        # Time in state
        has_durations = any(v > 0 for v in entry.level_durations_ns.values())
        if has_durations:
            dur = Text("  Time in state: ")
            for lvl in (0, 1, 2, 3):
                ns = entry.level_durations_ns.get(lvl, 0)
                if ns > 0:
                    style = LEVEL_STYLES[lvl]
                    name = LEVEL_NAMES[lvl]
                    dur.append(f"{name}={_format_duration_ns(ns)}  ", style=style)
            renderables.append(dur)

        # Sparkline
        spark = _sparkline(entry, width=40)
        if spark.plain:
            renderables.append(Text.assemble("  Timeline: ", spark))

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
                "frequency_hz": _compute_hz(e),
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
    inspect_all: Annotated[
        bool,
        Parameter(
            name=["-I", "--inspect-all"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    tree: Annotated[
        bool,
        Parameter(
            name=["--tree"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    topics: Annotated[
        list[str] | None,
        Parameter(
            name=["-t", "--topics"],
            group=FILTERING_GROUP,
        ),
    ] = None,
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
    (WARN, ERROR, STALE) and scans both /diagnostics and /diagnostics_agg.

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

      # Inspect all components
      pymcap-cli diag recording.mcap --inspect-all

      # Scan only /diagnostics (skip /diagnostics_agg)
      pymcap-cli diag recording.mcap -t /diagnostics

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
    inspect_all
        Show detailed view for all components.
    tree
        Display as hierarchical tree instead of flat table.
    topics
        Diagnostics topic names. Defaults to /diagnostics and /diagnostics_agg.
    json_output
        Output as JSON for scripting.
    """
    resolved_topics = topics if topics is not None else DEFAULT_TOPICS

    # Compile regex patterns upfront so invalid patterns fail fast
    name_re = _compile_pattern(name, "--name") if name else None
    hw_re = _compile_pattern(hardware_id, "--hardware-id") if hardware_id else None
    if inspect_all:
        inspect = "."
    inspect_re = _compile_pattern(inspect, "--inspect") if inspect else None

    try:
        entries = _collect_diagnostics(file, resolved_topics)
    except (OSError, ValueError, RuntimeError) as e:
        console_err.print(f"[red]Error reading MCAP file: {e}[/red]")
        return 1
    except KeyboardInterrupt:
        console_err.print("\n[yellow]Interrupted by user[/yellow]")
        return 0

    if not entries:
        topic_str = ", ".join(resolved_topics)
        console_err.print(f"[yellow]No diagnostics found on topic(s) '{topic_str}'[/yellow]")
        return 0

    level_totals: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for entry in entries.values():
        level_totals[entry.worst_level] += 1

    if show_all or inspect_re:
        min_level = 0
    elif level is not None:
        min_level = level
    else:
        min_level = 1

    filtered = _filter_entries(entries, min_level=min_level, name_pattern=name_re, hw_pattern=hw_re)

    if json_output:
        output = _build_json_output(filtered, len(entries), level_totals)
        print(json.dumps(output, indent=2), file=sys.stdout)  # noqa: T201
        return 0

    if inspect_re:
        matched = [e for e in filtered if inspect_re.search(e.name)]
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
