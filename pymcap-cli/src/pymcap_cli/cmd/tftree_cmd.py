from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from rich.text import Text
from rich.tree import Tree
from small_mcap import read_message_decoded

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


def _filter_transforms(
    transforms: dict[tuple[str, str], TransformData], static_only: bool
) -> dict[tuple[str, str], TransformData]:
    if not static_only:
        return transforms
    return {k: v for k, v in transforms.items() if v.is_static}


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


def _format_transform_label(transform: TransformData) -> Text:
    """Format a transform as a Rich Text label."""
    color = "green" if transform.is_static else "yellow"
    x, y, z = transform.translation
    qx, qy, qz, qw = transform.rotation

    return Text.assemble(
        (transform.child_frame_id, f"bold {color}"),
        (" @ ", "dim"),
        (_format_timestamp(transform.timestamp_ns), "dim"),
        (f" t:[{x:.3f}, {y:.3f}, {z:.3f}]", "cyan"),
        (f" r:[{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}]", "magenta"),
    )


def _add_frame_to_tree(
    tree_node: Tree,
    frame_id: str,
    transforms: dict[tuple[str, str], TransformData],
    tree_dict: dict[str, list[str]],
    visited: set[str],
) -> None:
    """Recursively add a frame and its children to the tree.

    Args:
        tree_node: Rich Tree node to add children to
        frame_id: Current frame to add
        transforms: Dictionary of all transforms
        tree_dict: Dictionary mapping parents to children
        visited: Set of already visited frames (to prevent cycles)
    """
    if frame_id in visited:
        return
    visited.add(frame_id)

    for child_frame_id in sorted(tree_dict.get(frame_id, [])):
        transform = transforms.get((frame_id, child_frame_id))
        if not transform:
            continue

        child_node = tree_node.add(_format_transform_label(transform))
        _add_frame_to_tree(child_node, child_frame_id, transforms, tree_dict, visited)


def _display_tf_tree(
    all_transforms: dict[tuple[str, str], TransformData], static_only: bool
) -> None:
    if not all_transforms:
        console.print("[yellow]No transforms found in MCAP file[/yellow]")
        return

    # Filter transforms if needed
    transforms = _filter_transforms(all_transforms, static_only)

    # Build tree structure and find roots in one pass
    tree_dict, root_frames = _build_tree_and_find_roots(transforms)

    if not root_frames:
        console.print("[yellow]No root frames found[/yellow]")
        return

    # Create and populate Rich tree
    tree = Tree("[bold blue]TF Tree[/bold blue]")
    visited: set[str] = set()
    for root_frame in root_frames:
        root_node = tree.add(Text(root_frame, style="bold"))
        _add_frame_to_tree(root_node, root_frame, transforms, tree_dict, visited)

    console.print(tree)

    # Display summary
    total = len(all_transforms)
    static = sum(1 for t in all_transforms.values() if t.is_static)

    console.print()
    console.print(f"[dim]Total transforms: {total}[/dim]")
    console.print(f"[dim]  Static: [green]{static}[/green][/dim]")
    console.print(f"[dim]  Dynamic: [yellow]{total - static}[/yellow][/dim]")


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

    try:
        with file_path.open("rb") as f:
            for msg in read_message_decoded(
                f,
                should_include=lambda ch, _: ch.topic in ("/tf", "/tf_static"),
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

    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error reading MCAP file: {e}[/red]")
        return

    _display_tf_tree(transforms, args.static_only)
