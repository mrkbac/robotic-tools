import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pymcap_cli.rebuild import rebuild_info
from pymcap_cli.utils import bytes_to_human

console = Console()


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
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

    parser.add_argument(
        "file",
        help="Path to the MCAP file to analyze",
        type=str,
    )
    parser.add_argument(
        "--exact-sizes",
        action="store_true",
        help="Use exact sizes for message data (may be slower)",
    )

    return parser


def run_du(file_path: Path, *, exact_sizes: bool) -> None:
    """Run disk usage analysis on MCAP file."""

    file_size = file_path.stat().st_size
    with file_path.open("rb") as f:
        info = rebuild_info(f, file_size, exact_sizes=exact_sizes)
        assert info.channel_sizes
        assert info.summary.statistics
        total_message_size = sum(info.channel_sizes.values())
        start_time = info.summary.statistics.message_start_time
        end_time = info.summary.statistics.message_end_time
        topic_message_size = info.channel_sizes

        """Display the usage analysis results."""
        # Message size stats table
        message_table = Table()
        message_table.add_column("Topic", style="bold white")
        message_table.add_column("Size", justify="right", style="green")
        message_table.add_column("Total %", justify="right", style="yellow")
        message_table.add_column("per sec", justify="right", style="cyan")

        # Sort topics by size (largest first)
        sorted_topics = sorted(topic_message_size.items(), key=lambda x: x[1], reverse=True)

        duration_ns = end_time - start_time

        for channel_id, size in sorted_topics:
            percentage = (size / total_message_size * 100) if total_message_size > 0 else 0
            per_seconds_bytes = int(size / duration_ns * 1_000_000_000)
            topic = info.summary.channels.get(channel_id)
            name = topic.topic if topic else f"Unknown Topic ({channel_id})"
            message_table.add_row(
                name, bytes_to_human(size), f"{percentage:.2f}%", bytes_to_human(per_seconds_bytes)
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
