from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, ConsoleOptions, RenderResult
from rich.jupyter import JupyterMixin
from rich.measure import Measurement
from rich.style import Style
from rich.table import Table
from rich.text import Text
from small_mcap import InvalidMagicError

from pymcap_cli.cmd.info_json_cmd import info_to_dict
from pymcap_cli.debug_wrapper import DebugStreamWrapper
from pymcap_cli.rebuild import read_info, rebuild_info
from pymcap_cli.utils import bytes_to_human

if TYPE_CHECKING:
    import argparse

    from pymcap_cli.types import McapInfoOutput

console = Console()


def _schema_to_color(schema_name: str | None) -> str:
    """Generate a deterministic color from a schema name.

    Args:
        schema_name: The schema name to hash

    Returns:
        Rich color name string
    """
    if not schema_name:
        return "white"

    colors = [
        "blue",
        "bright_cyan",
        "bright_green",
        "bright_magenta",
        "bright_red",
        "bright_yellow",
        "cyan",
        "green",
        "magenta",
        "red",
        "yellow",
    ]

    # Use Python's built-in hash for deterministic value
    hash_value = hash(schema_name)
    color_index = hash_value % len(colors)

    return colors[color_index]


def _create_ros_docs_url(schema_name: str, distro: str = "jazzy") -> str:
    parts = schema_name.split("/")
    if len(parts) == 2:
        # Old format: package_name/MessageType
        package_name, message_type = parts
        return f"https://docs.ros.org/en/{distro}/p/{package_name}/msg/{message_type}.html"
    if len(parts) == 3:
        # New format: package_name/msg/MessageType
        package_name, msg_dir, message_type = parts
        return f"https://docs.ros.org/en/{distro}/p/{package_name}/{msg_dir}/{message_type}.html"

    return ""


def _format_schema_with_link(schema_name: str | None, distro: str = "jazzy") -> str:
    if not schema_name:
        return "[dim]unknown[/dim]"

    # Skip non-ROS schemas (foxglove, malformed)
    if "/" not in schema_name:
        return schema_name

    # Create clickable link for ROS schemas
    url = _create_ros_docs_url(schema_name, distro)
    color = _schema_to_color(schema_name)
    return f"[{color} link={url}]{schema_name}[/]"


class DistributionBar(JupyterMixin):
    """A width-aware distribution bar that automatically scales to fit table cells.

    This renderable creates a visual distribution histogram using Unicode block
    characters with a color gradient from blue (low) to red (high).
    """

    def __init__(
        self,
        counts: list[int],
        *,
        width: int | None = None,
        min_width: int = 10,
    ) -> None:
        """Initialize the distribution bar.

        Args:
            counts: List of message counts per bucket
            width: Fixed width in characters, or None to use available width
            min_width: Minimum width to request
        """
        self.counts = counts
        self.width = width
        self.min_width = min_width

        # Unicode block characters for vertical bars (8 levels)
        self.blocks = " ▁▂▃▄▅▆▇█"

        # Color gradient styles for each level (0-8)
        self.level_styles = [
            Style(color="gray89", bgcolor="gray89"),  # 0: Empty - light gray
            Style(color="blue", bgcolor="gray89"),  # 1: Very low - blue
            Style(color="bright_blue", bgcolor="gray89"),  # 2: Low - brighter blue
            Style(color="cyan", bgcolor="gray89"),  # 3: Medium-low - cyan
            Style(color="green", bgcolor="gray89"),  # 4: Medium - green
            Style(color="yellow", bgcolor="gray89"),  # 5: Medium-high - yellow
            Style(color="bright_yellow", bgcolor="gray89"),  # 6: High - bright yellow
            Style(color="red", bgcolor="gray89"),  # 7: Very high - red
            Style(color="bright_red", bgcolor="gray89"),  # 8: Maximum - bright red
        ]

    def _downsample_to_width(self, counts: list[int], target_width: int) -> list[int]:
        num_buckets = len(counts)
        buckets_per_char = num_buckets / target_width

        scaled = []
        for i in range(target_width):
            start_idx = int(i * buckets_per_char)
            end_idx = int((i + 1) * buckets_per_char)
            # Take max value in this range (preserves peaks)
            if start_idx < end_idx:
                scaled.append(max(counts[start_idx:end_idx]))
            else:
                scaled.append(0)

        return scaled

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Determine actual rendering width
        # Use specified width OR expand to available width
        width = min(self.width or options.max_width, options.max_width)
        width = max(width, self.min_width)  # Respect minimum

        # Handle empty data
        if not self.counts or max(self.counts) == 0:
            yield Text("no messages", style="dim")
            return

        # Scale distribution to fit available width
        num_buckets = len(self.counts)

        if num_buckets <= width:
            # Enough space: show each bucket
            scaled_counts = self.counts
        else:
            # Too many buckets: downsample by taking max in each group
            scaled_counts = self._downsample_to_width(self.counts, width)

        # Render the scaled distribution
        max_count = max(scaled_counts)
        result = Text()

        for count in scaled_counts:
            if count == 0:
                # Show subtle background for empty buckets
                result.append("░", style=self.level_styles[0])
            else:
                # Scale to 1-8 range based on this channel's max
                level = int((count / max_count) * 8) if max_count > 0 else 0
                level = min(level, 8)  # Clamp to max
                level = max(level, 1)  # Ensure non-zero counts get at least level 1
                result.append(self.blocks[level], style=self.level_styles[level])

        yield result

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        if self.width is not None:
            # Fixed width
            return Measurement(self.width, self.width)
        # Flexible: min_width to max available
        return Measurement(self.min_width, options.max_width)


def _display_message_distribution(data: McapInfoOutput) -> None:
    """Display message distribution histogram using Unicode block characters."""
    distribution = data["message_distribution"]
    max_count = distribution["max_count"]

    # Skip if no messages
    if max_count == 0:
        return

    # Format bucket duration using timedelta
    bucket_duration_ns = distribution["bucket_duration_ns"]
    bucket_duration = timedelta(milliseconds=bucket_duration_ns / 1_000_000)

    # Format timedelta for display
    total_seconds = bucket_duration.total_seconds()
    if total_seconds < 1:
        duration_str = f"{total_seconds * 1000:.0f} ms"
    elif total_seconds < 60:
        duration_str = f"{total_seconds:.0f} s"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        duration_str = f"{minutes:.0f} min"
    else:
        hours = total_seconds / 3600
        duration_str = f"{hours:.1f} hr"

    console.print("\n[bold cyan]Message Distribution:[/bold cyan] ", end="")
    console.print(DistributionBar(distribution["message_counts"]))
    console.print(f"[dim]Max: {max_count:,} msgs/bucket | Bucket size: {duration_str}[/dim]\n")


def _display_file_info_and_summary(data: McapInfoOutput) -> None:
    """Display file information and summary statistics."""
    stats = data["statistics"]
    duration_ns = stats["duration_ns"]
    duration_human = timedelta(milliseconds=duration_ns / 1_000_000)
    date_start = datetime.fromtimestamp(stats["message_start_time"] / 1_000_000_000)
    date_end = datetime.fromtimestamp(stats["message_end_time"] / 1_000_000_000)

    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold blue")
    info_table.add_column()
    info_table.add_row("File:", f"[green]{data['file']['path']}[/green]")
    info_table.add_row("Library:", f"[yellow]{data['header']['library']}[/yellow]")
    info_table.add_row("Profile:", f"[yellow]{data['header']['profile']}[/yellow]")
    info_table.add_row("Messages:", f"[green]{stats['message_count']:,}[/green]")
    info_table.add_row("Chunks:", f"[cyan]{stats['chunk_count']}[/cyan]")
    info_table.add_row(
        "Duration:",
        f"[yellow]{duration_ns / 1_000_000:.2f} ms[/yellow] [cyan]({duration_human})[/cyan]",
    )
    info_table.add_row("Start:", f"[cyan]{date_start}[/cyan]")
    info_table.add_row("End:", f"[cyan]{date_end}[/cyan]")
    info_table.add_row("Channels:", f"[green]{stats['channel_count']}[/green]")
    info_table.add_row("Attachments:", f"[yellow]{stats['attachment_count']}[/yellow]")
    info_table.add_row("Metadata:", f"[cyan]{stats['metadata_count']}[/cyan]")
    console.print(info_table)


def _display_compression_table(data: McapInfoOutput, has_chunk_info: bool) -> None:
    """Display compression statistics table."""
    compression_table = Table()
    compression_table.add_column("Type", style="bold cyan")
    compression_table.add_column("Chunks", justify="right", style="green")
    compression_table.add_column("Compressed", justify="right", style="yellow")
    compression_table.add_column("Uncompressed", justify="right", style="yellow")
    compression_table.add_column("Ratio", justify="right", style="magenta")
    compression_table.add_column("Min Size", justify="right", style="yellow")
    compression_table.add_column("Avg Size", justify="right", style="yellow")
    compression_table.add_column("Max Size", justify="right", style="yellow")
    compression_table.add_column("Min Dur", justify="right", style="green")
    compression_table.add_column("Avg Dur", justify="right", style="yellow")
    compression_table.add_column("Max Dur", justify="right", style="magenta")
    if has_chunk_info:
        compression_table.add_column("Msgs", justify="right", style="cyan")

    for compression_type, chunk_stats in data["chunks"]["by_compression"].items():
        size_stats = chunk_stats["size_stats"]
        duration_stats = chunk_stats["duration_stats"]

        row = [
            compression_type,
            f"{chunk_stats['count']}",
            bytes_to_human(chunk_stats["compressed_size"]),
            bytes_to_human(chunk_stats["uncompressed_size"]),
            f"{chunk_stats['compression_ratio']:.2%}",
            bytes_to_human(size_stats["minimum"]),
            bytes_to_human(int(size_stats["average"])),
            bytes_to_human(size_stats["maximum"]),
            f"{duration_stats['minimum'] / 1_000_000:.2f} ms",
            f"{duration_stats['average'] / 1_000_000:.2f} ms",
            f"{duration_stats['maximum'] / 1_000_000:.2f} ms",
        ]
        if has_chunk_info:
            row.append(f"{chunk_stats['message_count']}")

        compression_table.add_row(*row)

    console.print(compression_table)

    # Display chunk overlaps if applicable
    overlaps = data["chunks"]["overlaps"]
    if overlaps["max_concurrent"] > 1:
        overlap_table = Table.grid(padding=(0, 1))
        overlap_table.add_column(style="bold blue")
        overlap_table.add_column()
        overlap_table.add_row(
            "Overlaps:",
            f"[green]{overlaps['max_concurrent']}[/green] max concurrent, "
            f"[yellow]{bytes_to_human(overlaps['max_concurrent_bytes'])}[/yellow] "
            f"max total size at once",
        )
        console.print(overlap_table)


def _display_channels_table(data: McapInfoOutput, args: argparse.Namespace) -> None:
    """Display channels table with sorting support."""

    # Dictionary of sort key functions
    sort_keys = {
        "topic": lambda c: c["topic"],
        "id": lambda c: c["id"],
        "msgs": lambda c: c["message_count"],
        "size": lambda c: c.get("size_bytes") or 0,
        "hz": lambda c: (c["hz_channel"] if args.index_duration and c["hz_channel"] else c["hz"]),
        "bps": lambda c: c.get("bytes_per_second") or 0,
        "b_per_msg": lambda c: c.get("bytes_per_message") or 0,
        "schema": lambda c: c.get("schema_name") or "",
    }
    get_sort_key = sort_keys.get(args.sort, sort_keys["topic"])

    # Check if size data is available
    has_size_data = any(ch["size_bytes"] is not None for ch in data["channels"])

    # Check if distribution data is available
    has_distribution_data = any(
        ch.get("message_distribution") and len(ch["message_distribution"]) > 0
        for ch in data["channels"]
    )

    # Detect terminal width and determine which columns to show
    terminal_width = console.width
    show_size = terminal_width >= 80 and has_size_data
    show_bps = terminal_width >= 100 and has_size_data
    show_b_per_msg = terminal_width >= 120 and has_size_data
    show_distribution = terminal_width >= 140 and has_distribution_data

    # Channels table
    channels_table = Table()
    channels_table.add_column("ID", style="bold blue", no_wrap=True, justify="right")
    channels_table.add_column("Topic", overflow="fold")
    channels_table.add_column("Schema", style="blue")
    channels_table.add_column("Msgs", justify="right", style="green")
    channels_table.add_column("Hz", justify="right", style="yellow")
    if show_size:
        channels_table.add_column("Size", justify="right", style="yellow")
    if show_bps:
        channels_table.add_column("B/s", justify="right", style="magenta")
    if show_b_per_msg:
        channels_table.add_column("B/msg", justify="right", style="magenta")
    if show_distribution:
        channels_table.add_column("Distribution")

    for channel in sorted(data["channels"], key=get_sort_key, reverse=args.reverse):
        hz = (
            channel["hz_channel"]
            if args.index_duration and channel["hz_channel"]
            else channel["hz"]
        )

        # Apply color based on schema
        topic_color = _schema_to_color(channel["schema_name"])
        colored_topic = f"[{topic_color}]{channel['topic']}[/{topic_color}]"

        row = [
            str(channel["id"]),
            colored_topic,
            _format_schema_with_link(channel["schema_name"]),
            f"{channel['message_count']:,}",
            f"{hz:.2f}",
        ]
        if show_size:
            size = channel.get("size_bytes") or 0
            row.append(bytes_to_human(size))
        if show_bps:
            bps = channel["bytes_per_second"] if channel["bytes_per_second"] is not None else 0
            row.append(bytes_to_human(int(bps)))
        if show_b_per_msg:
            b_per_msg = (
                channel["bytes_per_message"] if channel["bytes_per_message"] is not None else 0
            )
            row.append(bytes_to_human(int(b_per_msg)))

        # Add distribution bar if available
        if show_distribution:
            distribution = channel.get("message_distribution", [])
            row.append(DistributionBar(distribution))  # type: ignore[arg-type]

        channels_table.add_row(*row)

    console.print(channels_table)


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the info command parser to the subparsers."""
    parser = subparsers.add_parser(
        "info",
        help="Report statistics about an MCAP file",
        description="Report statistics about an MCAP file",
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file to analyze",
        type=str,
    )

    parser.add_argument(
        "-r",
        "--rebuild",
        action="store_true",
        help="Rebuild the MCAP file from scratch",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    parser.add_argument(
        "-s",
        "--sort",
        choices=["topic", "id", "msgs", "size", "hz", "bps", "b_per_msg", "schema"],
        default="topic",
        help="Sort channels by field (default: topic)",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort order (descending)",
    )

    parser.add_argument(
        "--index-duration",
        action="store_true",
        help=(
            "Calculate Hz per channel based on each channel's first/last "
            "message times rather than global MCAP duration"
        ),
    )

    return parser


def handle_command(args: argparse.Namespace) -> None:
    """Handle the info command execution."""

    file = Path(args.file)
    file_size = file.stat().st_size

    debug_wrapper = None
    with file.open("rb", buffering=0) as f_raw:
        if args.debug:
            debug_wrapper = DebugStreamWrapper(f_raw)
            f_buffered: io.BufferedReader = io.BufferedReader(debug_wrapper, buffer_size=1024)
        else:
            f_buffered = io.BufferedReader(f_raw, buffer_size=1024)

        if args.rebuild:
            info = rebuild_info(f_buffered, file_size)
        else:
            try:
                info = read_info(f_buffered)
            except InvalidMagicError:
                console.print("[red]Invalid MCAP magic, rebuilding info.[/red]")
                info = rebuild_info(f_buffered, file_size)

    if debug_wrapper:
        debug_wrapper.print_stats(file_size)

    # Get structured JSON data
    data = info_to_dict(info, str(file), file_size)

    # Warn if sorting by size fields when channel_sizes is unavailable
    has_size_data = any(ch["size_bytes"] is not None for ch in data["channels"])
    if args.sort in ["size", "bps", "b_per_msg"] and not has_size_data:
        console.print(
            "[yellow]Warning:[/yellow] Sorting by size fields requires channel size data. "
            "Use [cyan]--rebuild[/cyan] to get accurate size information."
        )
        console.print()

    # Warn if --index-duration is enabled but no per-channel duration data available
    has_channel_durations = any(ch["hz_channel"] is not None for ch in data["channels"])
    if args.index_duration and not has_channel_durations:
        console.print(
            "[yellow]Warning:[/yellow] --index-duration requires message index data. "
            "Use [cyan]--rebuild[/cyan] to get per-channel timing information. "
            "Falling back to global duration."
        )
        console.print()

    # Display all sections
    _display_file_info_and_summary(data)
    _display_message_distribution(data)
    _display_compression_table(data, info.chunk_information is not None)
    _display_channels_table(data, args)
