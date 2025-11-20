"""Info command - report statistics about an MCAP file."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console
from rich.table import Table
from small_mcap import InvalidMagicError

from pymcap_cli.cmd.info_json_cmd import info_to_dict
from pymcap_cli.display_utils import (
    ChannelTableColumn,
    DistributionBar,
    display_channels_table,
)
from pymcap_cli.info_types import McapInfoOutput
from pymcap_cli.input_handler import open_input
from pymcap_cli.utils import bytes_to_human, read_info, rebuild_info

console = Console()

# Parameter groups
PROCESSING_GROUP = Group("Processing Options")
DISPLAY_GROUP = Group("Display Options")
_NS_TO_MS = 1_000_000
_NS_TO_SEC = 1_000_000_000


class SortChoice(str, Enum):
    """Sort field choices for channel display."""

    TOPIC = "topic"
    ID = "id"
    MSGS = "msgs"
    SIZE = "size"
    HZ = "hz"
    BPS = "bps"
    B_PER_MSG = "b_per_msg"
    SCHEMA = "schema"


def _display_message_distribution(data: McapInfoOutput) -> None:
    """Display message distribution histogram using Unicode block characters."""
    distribution = data["message_distribution"]
    max_count = distribution["max_count"]

    # Skip if no messages
    if max_count == 0:
        return

    # Format bucket duration using timedelta
    bucket_duration_ns = distribution["bucket_duration_ns"]
    bucket_duration = timedelta(milliseconds=bucket_duration_ns / _NS_TO_MS)

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

    console.print("\n[bold cyan]Message Distribution:[/] ", end="")
    console.print(DistributionBar(distribution["message_counts"]))
    console.print(f"[dim]Max: {max_count:,} msgs/bucket | Bucket size: {duration_str}[/]\n")


def _display_file_info_and_summary(data: McapInfoOutput) -> None:
    """Display file information and summary statistics."""
    stats = data["statistics"]
    duration_ns = stats["duration_ns"]
    duration_human = timedelta(milliseconds=duration_ns / _NS_TO_MS)
    date_start = datetime.fromtimestamp(stats["message_start_time"] / _NS_TO_SEC)
    date_end = datetime.fromtimestamp(stats["message_end_time"] / _NS_TO_SEC)

    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold blue")
    info_table.add_column()
    info_table.add_row("File:", f"[green]{data['file']['path']}[/]")

    bytes_per_sec = data["file"]["size_bytes"] / (duration_ns / _NS_TO_SEC)
    bytes_per_hour = bytes_per_sec * 3600
    info_table.add_row(
        "Size:",
        f"[green]{bytes_to_human(data['file']['size_bytes'])}[/]"
        f" [red]{bytes_to_human(bytes_per_sec)}/s[/]"
        f" [orange]{bytes_to_human(bytes_per_hour)}/h[/]",
    )
    info_table.add_row("Library:", f"[yellow]{data['header']['library']}[/]")
    info_table.add_row("Profile:", f"[yellow]{data['header']['profile']}[/]")
    info_table.add_row("Messages:", f"[green]{stats['message_count']:,}[/]")
    info_table.add_row("Chunks:", f"[cyan]{stats['chunk_count']}[/]")
    info_table.add_row(
        "Duration:",
        f"[yellow]{duration_ns / 1_000_000:.2f} ms[/] [cyan]({duration_human})[/]",
    )
    info_table.add_row("Start:", f"[cyan]{date_start}[/]")
    info_table.add_row("End:", f"[cyan]{date_end}[/]")
    info_table.add_row("Channels:", f"[green]{stats['channel_count']}[/]")
    info_table.add_row("Attachments:", f"[yellow]{stats['attachment_count']}[/]")
    info_table.add_row("Metadata:", f"[cyan]{stats['metadata_count']}[/]")
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
            f"{duration_stats['minimum'] / _NS_TO_MS:.2f} ms",
            f"{duration_stats['average'] / _NS_TO_MS:.2f} ms",
            f"{duration_stats['maximum'] / _NS_TO_MS:.2f} ms",
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
            f"[green]{overlaps['max_concurrent']}[/] max concurrent, "
            f"[yellow]{bytes_to_human(overlaps['max_concurrent_bytes'])}[/] "
            f"max total size at once",
        )
        console.print(overlap_table)


def _display_channels_table(
    data: McapInfoOutput,
    sort_key: str,
    reverse: bool,
    index_duration: bool,
    use_median: bool,
    tree: bool,
) -> None:
    """Display channels table with sorting support."""
    display_channels_table(
        data,
        console,
        sort_key=sort_key,
        reverse=reverse,
        columns=(
            ChannelTableColumn.ID
            | ChannelTableColumn.SCHEMA
            | ChannelTableColumn.MSGS
            | ChannelTableColumn.HZ
            | ChannelTableColumn.SIZE
            | ChannelTableColumn.BPS
            | ChannelTableColumn.B_PER_MSG
            | ChannelTableColumn.DISTRIBUTION
        ),
        responsive=True,
        index_duration=index_duration,
        use_median=use_median,
        tree=tree,
    )


def info(
    files: list[str],
    *,
    rebuild: Annotated[
        bool,
        Parameter(
            name=["-r", "--rebuild"],
            group=PROCESSING_GROUP,
        ),
    ] = False,
    exact_sizes: Annotated[
        bool,
        Parameter(
            name=["-e", "--exact-sizes"],
            group=PROCESSING_GROUP,
        ),
    ] = False,
    debug: Annotated[
        bool,
        Parameter(
            name=["--debug"],
            group=PROCESSING_GROUP,
        ),
    ] = False,
    sort: Annotated[
        SortChoice,
        Parameter(
            name=["-s", "--sort"],
            group=DISPLAY_GROUP,
        ),
    ] = SortChoice.TOPIC,
    reverse: Annotated[
        bool,
        Parameter(
            name=["--reverse"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    index_duration: Annotated[
        bool,
        Parameter(
            name=["--index-duration"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    median: Annotated[
        bool,
        Parameter(
            name=["--median"],
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
) -> None:
    """Report statistics about MCAP file(s).

    This command displays comprehensive statistics about MCAP files including:
    - File metadata and summary statistics
    - Message distribution over time
    - Compression statistics by type
    - Channel information with message counts, data rates, and distributions

    When multiple files are provided, statistics are displayed separately for each file.

    Parameters
    ----------
    files
        Path(s) to MCAP file(s) to analyze (local files or HTTP/HTTPS URLs).
    rebuild
        Rebuild file metadata by scanning all records (use for corrupt or
        summary-less files).
    exact_sizes
        Calculate exact message sizes by decompressing chunks (slower, requires
        --rebuild).
    debug
        Enable debug mode with additional diagnostic output.
    sort
        Sort channels by field (topic, id, msgs, size, hz, bps, b_per_msg,
        schema).
    reverse
        Reverse sort order (descending).
    index_duration
        Calculate Hz per channel based on each channel's first/last message
        times rather than global MCAP duration.
    median
        Display median rates (Hz, bytes/s) instead of mean rates. Requires
        --rebuild to calculate message intervals.
    tree
        Display channels in a hierarchical tree structure based on topic paths.

    Examples
    --------
    ```
    # Basic file info
    pymcap-cli info recording.mcap

    # Multiple files
    pymcap-cli info file1.mcap file2.mcap file3.mcap

    # Rebuild summary with exact message sizes
    pymcap-cli info recording.mcap --rebuild --exact-sizes

    # Sort channels by message count (descending)
    pymcap-cli info recording.mcap --sort msgs --reverse

    # Calculate per-channel Hz using individual durations
    pymcap-cli info recording.mcap --index-duration
    ```
    """
    # Validate input
    if not files:
        console.print("[red]Error:[/] At least one file must be specified")
        raise SystemExit(1)

    # Process all files and display each separately
    for i, file in enumerate(files):
        if i > 0:
            console.print("\n" + "=" * 80 + "\n")

        with open_input(file, buffering=0, debug=debug) as (f_buffered, file_size):
            if rebuild:
                info_data = rebuild_info(f_buffered, file_size, exact_sizes=exact_sizes)
            else:
                try:
                    info_data = read_info(f_buffered)
                except (InvalidMagicError, AssertionError):
                    console.print("[red]Invalid MCAP magic, rebuilding info.[/]")
                    f_buffered.seek(0)  # Reset to start
                    info_data = rebuild_info(f_buffered, file_size, exact_sizes=exact_sizes)

        # Get structured JSON data
        data = info_to_dict(info_data, str(file), file_size)
        has_chunk_info = info_data.chunk_information is not None

        # Warn if sorting by size fields when channel_sizes is unavailable
        has_size_data = any(ch.get("size_bytes") is not None for ch in data["channels"])
        if sort.value in ["size", "bps", "b_per_msg"] and not has_size_data:
            console.print(
                "[yellow]Warning:[/] Sorting by size fields requires channel size data. "
                "Use [cyan]--rebuild[/] to get accurate size information."
            )
            console.print()

        # Warn if --index-duration is enabled but no per-channel duration data available
        has_channel_durations = any(ch.get("hz_channel") is not None for ch in data["channels"])
        if index_duration and not has_channel_durations:
            console.print(
                "[yellow]Warning:[/] --index-duration requires message index data. "
                "Use [cyan]--rebuild[/] to get per-channel timing information. "
                "Falling back to global duration."
            )
            console.print()

        # Warn if --median is enabled but no median data available
        has_median_data = any(
            (hz_stats := ch.get("hz_stats")) and hz_stats.get("median") is not None
            for ch in data["channels"]
        )
        if median and not has_median_data:
            console.print(
                "[yellow]Warning:[/] --median requires message interval data. "
                "Use [cyan]--rebuild[/] to calculate median rates. "
                "Falling back to mean rates."
            )
            console.print()

        # Display all sections
        _display_file_info_and_summary(data)
        _display_message_distribution(data)
        _display_compression_table(data, has_chunk_info)
        _display_channels_table(data, sort.value, reverse, index_duration, median, tree)
