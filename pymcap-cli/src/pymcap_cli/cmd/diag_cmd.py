"""Diag command - inspect ROS2 diagnostics from MCAP files."""

import json
import logging
import re
import sys
from typing import Annotated

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from small_mcap import include_topics, read_message_decoded

from pymcap_cli.core.diagnostics import (
    DEFAULT_TOPICS,
    DiagEntry,
    add_diagnostic_message,
    filter_entries,
    level_totals,
)
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.diag_render import (
    build_inspect_view,
    build_json_output,
    build_summary_table,
    build_tree_view,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)
console = Console()

FILTERING_GROUP = Group("Filtering")
DISPLAY_GROUP = Group("Display")


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
            console=ERR,
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
            add_diagnostic_message(entries, msg.message.log_time, msg.decoded_message)

            if msg_count % 100 == 0:
                progress.update(task, completed=f.tell(), msgs=msg_count, components=len(entries))

        progress.update(task, completed=file_size, msgs=msg_count, components=len(entries))

    return entries


def _compile_pattern(pattern: str, flag_name: str) -> re.Pattern[str]:
    """Compile a regex pattern with a user-friendly error on invalid syntax."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        logger.exception(f"Invalid regex for {flag_name}")
        raise SystemExit(1) from e


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
    except (OSError, ValueError, RuntimeError):
        logger.exception("Error reading MCAP file")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0

    if not entries:
        topic_str = ", ".join(resolved_topics)
        logger.warning(f"No diagnostics found on topic(s) '{topic_str}'")
        return 0

    totals = level_totals(entries)

    if show_all or inspect_re:
        min_level = 0
    elif level is not None:
        min_level = level
    else:
        min_level = 1

    filtered = filter_entries(entries, min_level=min_level, name_pattern=name_re, hw_pattern=hw_re)

    if json_output:
        output = build_json_output(filtered, len(entries), totals)
        print(json.dumps(output, indent=2), file=sys.stdout)  # noqa: T201
        return 0

    if inspect_re:
        matched = [e for e in filtered if inspect_re.search(e.name)]
        if not matched:
            logger.warning(f"No components matching '{inspect}'")
            return 0

        for renderable in build_inspect_view(matched):
            console.print(renderable)
        return 0

    if tree:
        console.print(build_tree_view(filtered))
        return 0

    console.print(build_summary_table(filtered, len(entries), totals, show_all))
    return 0
