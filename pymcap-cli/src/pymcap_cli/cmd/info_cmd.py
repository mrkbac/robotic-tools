"""Info command - report statistics about an MCAP file."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console, ConsoleOptions, RenderResult
from rich.jupyter import JupyterMixin
from rich.measure import Measurement
from rich.style import Style
from rich.table import Table
from rich.text import Text
from small_mcap import InvalidMagicError

from pymcap_cli.cmd.info_json_cmd import info_to_dict
from pymcap_cli.display_utils import ChannelTableColumn, display_channels_table
from pymcap_cli.info_types import McapInfoOutput
from pymcap_cli.input_handler import open_input
from pymcap_cli.utils import bytes_to_human, read_info, rebuild_info

console = Console()

# Parameter groups
PROCESSING_GROUP = Group("Processing Options")
DISPLAY_GROUP = Group("Display Options")


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
        # Flexible: cap max width based on actual bucket count and a reasonable maximum
        # No point displaying wider than the number of buckets, and cap at 80 chars
        max_useful_width = min(len(self.counts), 80, options.max_width)
        return Measurement(self.min_width, max_useful_width)


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


def _display_channels_table(
    data: McapInfoOutput,
    sort_key: str,
    reverse: bool,
    index_duration: bool,
    use_median: bool,
) -> None:
    """Display channels table with sorting support."""
    display_channels_table(
        data,
        console,
        sort_key=sort_key,
        reverse=reverse,
        columns=(
            ChannelTableColumn.ID
            | ChannelTableColumn.TOPIC
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
        distribution_bar_class=DistributionBar,
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
        console.print("[red]Error:[/red] At least one file must be specified")
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
                    console.print("[red]Invalid MCAP magic, rebuilding info.[/red]")
                    f_buffered.seek(0)  # Reset to start
                    info_data = rebuild_info(f_buffered, file_size, exact_sizes=exact_sizes)

        # Get structured JSON data
        data = info_to_dict(info_data, str(file), file_size)
        has_chunk_info = info_data.chunk_information is not None

        # Warn if sorting by size fields when channel_sizes is unavailable
        has_size_data = any(ch.get("size_bytes") is not None for ch in data["channels"])
        if sort.value in ["size", "bps", "b_per_msg"] and not has_size_data:
            console.print(
                "[yellow]Warning:[/yellow] Sorting by size fields requires channel size data. "
                "Use [cyan]--rebuild[/cyan] to get accurate size information."
            )
            console.print()

        # Warn if --index-duration is enabled but no per-channel duration data available
        has_channel_durations = any(ch.get("hz_channel") is not None for ch in data["channels"])
        if index_duration and not has_channel_durations:
            console.print(
                "[yellow]Warning:[/yellow] --index-duration requires message index data. "
                "Use [cyan]--rebuild[/cyan] to get per-channel timing information. "
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
                "[yellow]Warning:[/yellow] --median requires message interval data. "
                "Use [cyan]--rebuild[/cyan] to calculate median rates. "
                "Falling back to mean rates."
            )
            console.print()

        # Display all sections
        _display_file_info_and_summary(data)
        _display_message_distribution(data)
        _display_compression_table(data, has_chunk_info)
        _display_channels_table(data, sort.value, reverse, index_duration, median)
