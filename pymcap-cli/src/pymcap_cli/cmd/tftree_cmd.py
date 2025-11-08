from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from small_mcap import read_message_decoded
from small_mcap.reader import include_topics

if TYPE_CHECKING:
    import argparse

console = Console()


@dataclass
class TransformData:
    """Data for a single transform between two frames."""

    frame_id: str
    child_frame_id: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    is_static: bool
    timestamp_ns: int


def _detect_multiple_parents(
    transforms: dict[tuple[str, str], TransformData],
) -> dict[str, set[str]]:
    """Detect frames that have multiple parent frames (violates tree structure).

    Args:
        transforms: Dictionary of all transforms keyed by (parent, child)

    Returns:
        Dictionary mapping child frame names to their set of parent frames,
        only including children with multiple parents
    """
    child_to_parents: dict[str, set[str]] = defaultdict(set)

    for parent, child in transforms:
        child_to_parents[child].add(parent)

    # Return only frames with multiple parents
    return {child: parents for child, parents in child_to_parents.items() if len(parents) > 1}


def _build_tree_and_find_roots(
    transforms: dict[tuple[str, str], TransformData],
) -> tuple[dict[str, list[str]], list[str]]:
    tree_dict: dict[str, list[str]] = defaultdict(list)
    parents = set()
    children = set()

    for transform in transforms.values():
        parent = transform.frame_id
        child = transform.child_frame_id

        tree_dict[parent].append(child)
        parents.add(parent)
        children.add(child)

    # Root frames are parents that are not children
    root_frames = sorted(parents - children) or sorted(parents)

    return dict(tree_dict), root_frames


def _format_timestamp(timestamp_ns: int) -> str:
    """Format timestamp in nanoseconds to a readable string."""
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")


def _quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Use 90 degrees if out of range
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _build_table_rows(
    frame_id: str,
    transforms: dict[tuple[str, str], TransformData],
    tree_dict: dict[str, list[str]],
    visited: set[str],
    prefix: str = "",
) -> list[tuple[str, str, float, float, float, float, float, float, str]]:
    """Recursively build table rows with tree structure prefixes.

    Args:
        frame_id: Current frame to process
        transforms: Dictionary of all transforms
        tree_dict: Dictionary mapping parents to children
        visited: Set of already visited frames (to prevent cycles)
        prefix: Current line prefix for tree structure

    Returns:
        List of tuples: (frame_with_prefix, timestamp, tx, ty, tz, roll, pitch, yaw, color)
    """
    rows: list[tuple[str, str, float, float, float, float, float, float, str]] = []

    if frame_id in visited:
        return rows
    visited.add(frame_id)

    children = sorted(tree_dict.get(frame_id, []))

    for i, child_frame_id in enumerate(children):
        transform = transforms.get((frame_id, child_frame_id))
        if not transform:
            continue

        is_last_child = i == len(children) - 1
        connector = "└── " if is_last_child else "├── "

        # Format row data
        color = "green" if transform.is_static else "yellow"
        frame_with_prefix = f"{prefix}{connector}{child_frame_id}"
        timestamp = _format_timestamp(transform.timestamp_ns)
        tx, ty, tz = transform.translation
        qx, qy, qz, qw = transform.rotation
        roll, pitch, yaw = _quaternion_to_euler(qx, qy, qz, qw)

        rows.append((frame_with_prefix, timestamp, tx, ty, tz, roll, pitch, yaw, color))

        # Build prefix for children
        child_prefix = prefix + ("    " if is_last_child else "│   ")

        # Recursively add children
        rows.extend(_build_table_rows(child_frame_id, transforms, tree_dict, visited, child_prefix))

    return rows


def _build_tf_table(transforms: dict[tuple[str, str], TransformData]) -> Table | None:
    if not transforms:
        return None

    # Build tree structure and find roots in one pass
    tree_dict, root_frames = _build_tree_and_find_roots(transforms)

    if not root_frames:
        return None

    # Create Rich table
    total = len(transforms)
    static = sum(1 for t in transforms.values() if t.is_static)
    title = f"TF Tree (Total: {total} | Static: {static} | Dynamic: {total - static})"

    table = Table(show_header=True, box=None, padding=(0, 1), title=title)
    table.add_column("Frame", style="bold", no_wrap=True)
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("tx", style="cyan", justify="right")
    table.add_column("ty", style="cyan", justify="right")
    table.add_column("tz", style="cyan", justify="right")
    table.add_column("roll", style="magenta", justify="right")
    table.add_column("pitch", style="magenta", justify="right")
    table.add_column("yaw", style="magenta", justify="right")

    # Build all rows
    visited: set[str] = set()
    for root_frame in root_frames:
        # Add root frame row
        table.add_row(root_frame, "", "", "", "", "", "", "")

        # Add child rows
        rows = _build_table_rows(root_frame, transforms, tree_dict, visited, "")
        for frame_with_prefix, timestamp, tx, ty, tz, roll, pitch, yaw, color in rows:
            table.add_row(
                Text(frame_with_prefix, style=f"bold {color}"),
                timestamp,
                f"{tx:.3f}",
                f"{ty:.3f}",
                f"{tz:.3f}",
                f"{roll:.1f}",
                f"{pitch:.1f}",
                f"{yaw:.1f}",
            )

    return table


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the tftree command parser to the subparsers."""
    parser = subparsers.add_parser(
        "tftree",
        help="Display TF transform tree from MCAP file",
        description="Display TF transform tree from MCAP file",
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file to analyze",
        type=str,
    )

    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Show only static transforms (/tf_static)",
    )

    return parser


def handle_command(args: argparse.Namespace) -> None:
    """Handle the tftree command execution."""
    file_path = Path(args.file)

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return

    transforms: dict[tuple[str, str], TransformData] = {}

    topics = ["/tf_static"]
    if not args.static_only:
        topics.append("/tf")

    try:
        with file_path.open("rb") as f, Live(auto_refresh=False, console=console) as live:
            for msg in read_message_decoded(
                f,
                should_include=include_topics(topics),
                decoder_factories=[DecoderFactory()],
            ):
                for transform_stamped in msg.decoded_message.transforms:
                    trans = transform_stamped.transform.translation
                    rot = transform_stamped.transform.rotation

                    key = (transform_stamped.header.frame_id, transform_stamped.child_frame_id)
                    transforms[key] = TransformData(
                        frame_id=transform_stamped.header.frame_id,
                        child_frame_id=transform_stamped.child_frame_id,
                        translation=(trans.x, trans.y, trans.z),
                        rotation=(rot.x, rot.y, rot.z, rot.w),
                        is_static=msg.channel.topic == "/tf_static",
                        timestamp_ns=msg.message.log_time,
                    )

                # Update display
                table = _build_tf_table(transforms)
                if table:
                    live.update(table)
                    live.refresh()

            # Final update
            table = _build_tf_table(transforms)
            if table:
                live.update(table)
                live.refresh()

        # Post-processing validation: check for multiple parents
        multiple_parents = _detect_multiple_parents(transforms)
        if multiple_parents:
            console.print()  # Add spacing
            console.print("[yellow]⚠ Tree Structure Violations Detected:[/yellow]")
            description = "[dim]The following frames have multiple parents, "
            console.print(description)
            for child, parents in sorted(multiple_parents.items()):
                parents_str = ", ".join(f"'{p}'" for p in sorted(parents))
                frame_msg = f"  - Frame [bold]'{child}'[/bold] has parents: {parents_str}"
                console.print(frame_msg, style="yellow")

    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error reading MCAP file: {e}[/red]")
        return
