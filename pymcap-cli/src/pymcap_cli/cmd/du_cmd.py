from __future__ import annotations

import argparse
from pathlib import Path

import shtab
from rich.console import Console

from pymcap_cli.cmd.info_json_cmd import info_to_dict
from pymcap_cli.display_utils import ChannelTableColumn, display_channels_table
from pymcap_cli.utils import rebuild_info

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

        # Display channels table sorted by size
        display_channels_table(
            data,
            console,
            sort_key="size",
            reverse=True,
            columns=(
                ChannelTableColumn.TOPIC
                | ChannelTableColumn.MSGS
                | ChannelTableColumn.HZ
                | ChannelTableColumn.SIZE
                | ChannelTableColumn.PERCENT
                | ChannelTableColumn.BPS
                | ChannelTableColumn.B_PER_MSG
            ),
            responsive=False,
            index_duration=False,
        )


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
