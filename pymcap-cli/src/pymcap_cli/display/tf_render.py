"""Rich rendering of a TF transform tree, shared by `tftree` and `bridge tf`."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.table import Table

from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.tf_findings import TfSeverity
from pymcap_cli.core.tf_tree import (
    TransformData,
    build_tree_and_find_roots,
    quaternion_to_euler_rad,
)

if TYPE_CHECKING:
    from pymcap_cli.core.tf_findings import TfFinding

_TfRow = tuple[str, int, str, float, float, float, float, float, float]

# Below this console width the Count + Timestamp columns are dropped so the
# translation/rotation numbers stay readable instead of ellipsizing to "0…".
TF_COMPACT_WIDTH = 110


def _format_timestamp(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / NS_TO_SEC, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _build_table_rows(
    frame_id: str,
    transforms: dict[tuple[str, str], TransformData],
    transform_counts: dict[tuple[str, str], int],
    tree_dict: dict[str, list[str]],
    visited: set[str],
    prefix: str = "",
) -> list[_TfRow]:
    rows: list[_TfRow] = []

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


def build_tf_table(
    transforms: dict[tuple[str, str], TransformData],
    transform_counts: dict[tuple[str, str], int],
    *,
    compact: bool = False,
) -> Table | None:
    """Render the transform forest as a Rich table, or None if there is nothing to show.

    When ``compact`` is set the Count and Timestamp columns are dropped so the
    translation/rotation values stay readable on a narrow terminal.
    """
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
    if not compact:
        table.add_column("Count", style="yellow", justify="right")
        table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("tx", style="cyan", justify="right")
    table.add_column("ty", style="cyan", justify="right")
    table.add_column("tz", style="cyan", justify="right")
    table.add_column("roll", style="magenta", justify="right")
    table.add_column("pitch", style="magenta", justify="right")
    table.add_column("yaw", style="magenta", justify="right")

    empty_lead = 6 if compact else 8
    visited: set[str] = set()
    for root_frame in root_frames:
        table.add_row(f"[bold]{root_frame}[/]", *([""] * empty_lead))

        rows = _build_table_rows(root_frame, transforms, transform_counts, tree_dict, visited, "")
        for frame_with_prefix, count, timestamp, tx, ty, tz, roll, pitch, yaw in rows:
            numbers = (
                f"{tx:7.3f}",
                f"{ty:7.3f}",
                f"{tz:7.3f}",
                f"{roll:4.1f}",
                f"{pitch:4.1f}",
                f"{yaw:4.1f}",
            )
            if compact:
                table.add_row(frame_with_prefix, *numbers)
            else:
                table.add_row(frame_with_prefix, f"{count:<4}", timestamp, *numbers)

    return table


def _severity_cell(severity: TfSeverity) -> str:
    if severity is TfSeverity.ERROR:
        return "[red]error[/red]"
    if severity is TfSeverity.WARNING:
        return "[yellow]warning[/yellow]"
    return "[cyan]info[/cyan]"


def build_findings_table(findings: list[TfFinding]) -> Table:
    """Render TF validation findings (missing roots, multiple parents, …) as a table."""
    table = Table(title="TF Findings")
    table.add_column("Severity", no_wrap=True)
    table.add_column("Code", no_wrap=True)
    table.add_column("Message")
    for finding in findings:
        table.add_row(_severity_cell(finding.severity), finding.code.value, finding.message)
    return table
