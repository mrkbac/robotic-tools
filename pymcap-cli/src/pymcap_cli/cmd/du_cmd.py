from __future__ import annotations

import argparse
from pathlib import Path

import shtab
from rich.console import Console
from rich.table import Table

from pymcap_cli.cmd.info_json_cmd import info_to_dict
from pymcap_cli.display_utils import schema_to_color
from pymcap_cli.rebuild import rebuild_info
from pymcap_cli.utils import bytes_to_human

console = Console()


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the du command parser to the subparsers."""
    parser = subparsers.add_parser(
        "du",
        help="Report space usage within an MCAP file",
        description=(
            "This command reports space usage within an mcap file. Space usage for messages is "
            "calculated using the uncompressed size.\n\n"
            "Note: This command will scan and uncompress the entire file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    file_arg = parser.add_argument(
        "file",
        help="Path to the MCAP file to analyze",
        type=str,
    )
    file_arg.complete = shtab.FILE  # type: ignore[attr-defined]

    parser.add_argument(
        "--exact-sizes",
        "-e",
        action="store_true",
        help="Use exact sizes for message data (may be slower)",
    )

    return parser


def run_du(file_path: Path, *, exact_sizes: bool) -> None:
    """Run disk usage analysis on MCAP file."""

    file_size = file_path.stat().st_size
    with file_path.open("rb") as f:
        # Use rebuild_info to get channel sizes (du always rebuilds)
        info = rebuild_info(f, file_size, exact_sizes=exact_sizes)

        # Transform to JSON structure (shared logic with info-json command)
        data = info_to_dict(info, str(file_path), file_size)

        # Extract channel data for display
        channels = data["channels"]
        total_message_size = sum(ch["size_bytes"] for ch in channels if ch["size_bytes"])

        # Message size stats table
        message_table = Table()
        message_table.add_column("Topic", style="bold white")
        message_table.add_column("Size", justify="right", style="green")
        message_table.add_column("Total %", justify="right", style="yellow")
        message_table.add_column("per sec", justify="right", style="cyan")

        # Sort channels by size (largest first)
        sorted_channels = sorted(
            channels, key=lambda ch: ch["size_bytes"] if ch["size_bytes"] else 0, reverse=True
        )

        for channel in sorted_channels:
            size = channel["size_bytes"]
            if size is None:
                size = 0

            percentage = (size / total_message_size * 100) if total_message_size > 0 else 0
            per_seconds_bytes = channel["bytes_per_second"] or 0

            # Apply schema-based coloring to topic
            topic_color = schema_to_color(channel["schema_name"])
            colored_topic = f"[{topic_color}]{channel['topic']}[/{topic_color}]"

            message_table.add_row(
                colored_topic,
                bytes_to_human(size),
                f"{percentage:.2f}%",
                bytes_to_human(int(per_seconds_bytes)),
            )

        console.print(message_table)


def handle_command(args: argparse.Namespace) -> None:
    """Handle the du command execution."""
    file_path = Path(args.file)

    if not file_path.exists():
        console.print(f"[red]Error: File '{file_path}' does not exist[/red]")
        return

    if not file_path.is_file():
        console.print(f"[red]Error: '{file_path}' is not a file[/red]")
        return

    run_du(file_path, exact_sizes=args.exact_sizes)
