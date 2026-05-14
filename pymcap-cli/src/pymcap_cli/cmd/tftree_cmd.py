"""TF tree command - display transform tree from MCAP file."""

import logging
import math
from datetime import datetime
from typing import Annotated

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from rich.live import Live
from rich.table import Table
from small_mcap import include_topics, read_message_decoded

from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.tf_findings import (
    TfFinding,
    TfSeverity,
    collect_tf_findings,
    has_error_findings,
)
from pymcap_cli.core.tf_tree import (
    TfGraphBuilder,
    TransformData,
    build_tree_and_find_roots,
    quaternion_to_euler_rad,
    stamp_to_ns,
)

logger = logging.getLogger(__name__)
console = Console()

DISPLAY_GROUP = Group("Display")


def _format_timestamp(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / NS_TO_SEC).strftime("%Y-%m-%d %H:%M:%S")


def _build_table_rows(
    frame_id: str,
    transforms: dict[tuple[str, str], TransformData],
    transform_counts: dict[tuple[str, str], int],
    tree_dict: dict[str, list[str]],
    visited: set[str],
    prefix: str = "",
) -> list[tuple[str, int, str, float, float, float, float, float, float]]:
    rows: list[tuple[str, int, str, float, float, float, float, float, float]] = []

    if frame_id in visited:
        return rows
    visited.add(frame_id)

    children = sorted(tree_dict.get(frame_id, []))

    for i, child_frame_id in enumerate(children):
        key = (frame_id, child_frame_id)
        transform = transforms.get(key)
        if not transform:
            continue

        is_last_child = i == len(children) - 1
        connector = "└── " if is_last_child else "├── "

        if transform.is_static:
            frame_with_prefix = f"{prefix}{connector}[green]{child_frame_id}[/]"
        else:
            frame_with_prefix = f"{prefix}[red]{connector}{child_frame_id}[/]"
        count = transform_counts[key]
        timestamp = _format_timestamp(transform.timestamp_ns)
        tx, ty, tz = transform.translation
        qx, qy, qz, qw = transform.rotation
        roll_rad, pitch_rad, yaw_rad = quaternion_to_euler_rad(qx, qy, qz, qw)
        roll = math.degrees(roll_rad)
        pitch = math.degrees(pitch_rad)
        yaw = math.degrees(yaw_rad)

        rows.append((frame_with_prefix, count, timestamp, tx, ty, tz, roll, pitch, yaw))

        child_prefix = prefix + ("    " if is_last_child else "│   ")
        rows.extend(
            _build_table_rows(
                child_frame_id, transforms, transform_counts, tree_dict, visited, child_prefix
            )
        )

    return rows


def _build_tf_table(
    transforms: dict[tuple[str, str], TransformData],
    transform_counts: dict[tuple[str, str], int],
) -> Table | None:
    if not transforms:
        return None

    tree_dict, root_frames = build_tree_and_find_roots(transforms)

    if not root_frames:
        return None

    total = len(transforms)
    static = sum(1 for t in transforms.values() if t.is_static)
    title = f"TF Tree Total: {total} | Static: {static} | Dynamic: {total - static}"
    title += " [red]RED[/red]=Dynamic [green]GREEN[/green]=Static"

    table = Table(show_header=True, box=None, padding=(0, 1), title=title)
    table.add_column("Frame", style="bold", no_wrap=True)
    table.add_column("Count", style="yellow", justify="right")
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("tx", style="cyan", justify="right")
    table.add_column("ty", style="cyan", justify="right")
    table.add_column("tz", style="cyan", justify="right")
    table.add_column("roll", style="magenta", justify="right")
    table.add_column("pitch", style="magenta", justify="right")
    table.add_column("yaw", style="magenta", justify="right")

    visited: set[str] = set()
    for root_frame in root_frames:
        table.add_row(f"[bold]{root_frame}[/]", "", "", "", "", "", "", "", "", "")

        rows = _build_table_rows(root_frame, transforms, transform_counts, tree_dict, visited, "")
        for (
            frame_with_prefix,
            count,
            timestamp,
            tx,
            ty,
            tz,
            roll,
            pitch,
            yaw,
        ) in rows:
            table.add_row(
                frame_with_prefix,
                f"{count:<4}",
                timestamp,
                f"{tx:7.3f}",
                f"{ty:7.3f}",
                f"{tz:7.3f}",
                f"{roll:4.1f}",
                f"{pitch:4.1f}",
                f"{yaw:4.1f}",
            )

    return table


def tftree(
    file: str,
    *,
    static_only: Annotated[
        bool,
        Parameter(
            name=["--static-only"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    change_only: Annotated[
        bool,
        Parameter(
            name=["--change-only"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
) -> int:
    """Display TF transform tree from MCAP file.

    Parameters
    ----------
    file
        Path to MCAP file (local file or HTTP/HTTPS URL).
    static_only
        Show only static transforms (/tf_static).
    change_only
        Update display only when tree structure changes (new frames added).
    """
    builder = TfGraphBuilder()
    seen_frame_pairs: set[tuple[str, str]] = set()

    topics = ["/tf_static"]
    if not static_only:
        topics.append("/tf")

    try:
        with open_input(file) as (f, _file_size), Live(console=console) as live:
            for msg in read_message_decoded(
                f,
                should_include=include_topics(topics),
                decoder_factories=[DecoderFactory()],
            ):
                tree_changed = False
                for transform_stamped in msg.decoded_message.transforms:
                    trans = transform_stamped.transform.translation
                    rot = transform_stamped.transform.rotation

                    key = (transform_stamped.header.frame_id, transform_stamped.child_frame_id)

                    if key not in seen_frame_pairs:
                        seen_frame_pairs.add(key)
                        tree_changed = True

                    builder.add(
                        static=msg.channel.topic == "/tf_static",
                        stamp_ns=stamp_to_ns(transform_stamped.header.stamp),
                        parent=transform_stamped.header.frame_id,
                        child=transform_stamped.child_frame_id,
                        translation=(trans.x, trans.y, trans.z),
                        rotation=(rot.x, rot.y, rot.z, rot.w),
                    )

                if not change_only or tree_changed:
                    graph = builder.graph()
                    table = _build_tf_table(graph.transforms, graph.counts)
                    if table:
                        live.update(table)

            graph = builder.graph()
            table = _build_tf_table(graph.transforms, graph.counts)
            if table:
                live.update(table)

        graph = builder.graph()
        findings = collect_tf_findings(graph)
        if findings:
            console.print()
            console.print(_findings_table(findings))

    except (OSError, ValueError, RuntimeError):
        logger.exception("Error reading MCAP file")
        return 1

    return 1 if has_error_findings(findings) else 0


def _findings_table(findings: list[TfFinding]) -> Table:
    table = Table(title="TF Findings")
    table.add_column("Severity", no_wrap=True)
    table.add_column("Code", no_wrap=True)
    table.add_column("Message")
    for finding in findings:
        table.add_row(_severity_cell(finding.severity), finding.code.value, finding.message)
    return table


def _severity_cell(severity: TfSeverity) -> str:
    if severity is TfSeverity.ERROR:
        return "[red]error[/red]"
    if severity is TfSeverity.WARNING:
        return "[yellow]warning[/yellow]"
    return "[cyan]info[/cyan]"
