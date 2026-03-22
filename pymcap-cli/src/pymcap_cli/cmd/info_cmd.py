"""Info command - report statistics about an MCAP file."""

import base64
import gzip
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text
from small_mcap import McapError, rebuild_summary

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.display_utils import (
    ChannelTableColumn,
    DistributionBar,
    display_channels_table,
)
from pymcap_cli.types.info_data import info_to_dict
from pymcap_cli.types.info_link import ScanMode, generate_link
from pymcap_cli.types.info_types import McapInfoOutput
from pymcap_cli.utils import bytes_to_human, read_or_rebuild_info

console = Console()

# Parameter groups
PROCESSING_GROUP = CycloptsGroup("Processing Options")
DISPLAY_GROUP = CycloptsGroup("Display Options")
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


def _build_message_distribution(data: McapInfoOutput) -> RenderableType | None:
    """Build message distribution histogram renderable."""
    distribution = data["message_distribution"]
    max_count = distribution["max_count"]

    # Skip if no messages
    if max_count == 0:
        return None

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

    return Group(
        Text(""),
        Text.from_markup("[bold cyan]Message Distribution:[/]"),
        DistributionBar(distribution["message_counts"]),
        Text.from_markup(f"[dim]Max: {max_count:,} msgs/bucket | Bucket size: {duration_str}[/]"),
        Text(""),
    )


def _build_file_info_and_summary(data: McapInfoOutput) -> Table:
    """Build file information and summary statistics renderable."""
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
    info_table.add_row("Chunks:", f"[cyan]{stats['chunk_count']:,}[/]")
    info_table.add_row(
        "Duration:",
        f"[yellow]{duration_ns / 1_000_000:.2f} ms[/] [cyan]({duration_human})[/]",
    )
    info_table.add_row("Start:", f"[cyan]{date_start}[/]")
    info_table.add_row("End:", f"[cyan]{date_end}[/]")
    info_table.add_row("Channels:", f"[green]{stats['channel_count']:,}[/]")
    info_table.add_row("Attachments:", f"[yellow]{stats['attachment_count']:,}[/]")
    info_table.add_row("Metadata:", f"[cyan]{stats['metadata_count']:,}[/]")
    if msg_idx_count := stats.get("message_index_count"):
        info_table.add_row("Indexed Messages:", f"[green]{msg_idx_count:,}[/]")
    return info_table


def _build_compression_table(data: McapInfoOutput, has_chunk_info: bool) -> RenderableType:
    """Build compression statistics renderable."""
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
        return Group(compression_table, overlap_table)

    return compression_table


def _build_channels_table(
    data: McapInfoOutput,
    sort_key: str,
    reverse: bool,
    index_duration: bool,
    use_median: bool,
    tree: bool,
    terminal_width: int | None = None,
) -> Table:
    """Build channels table renderable."""
    return display_channels_table(
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
        terminal_width=terminal_width,
    )


def _build_info_display(
    data: McapInfoOutput,
    has_chunk_info: bool,
    sort_key: str,
    reverse: bool,
    index_duration: bool,
    use_median: bool,
    tree: bool,
    terminal_width: int | None = None,
) -> Group:
    """Assemble all info sections into a single renderable."""
    parts: list[RenderableType] = [_build_file_info_and_summary(data)]

    distribution = _build_message_distribution(data)
    if distribution is not None:
        parts.append(distribution)

    parts.append(_build_compression_table(data, has_chunk_info))
    parts.append(
        _build_channels_table(
            data, sort_key, reverse, index_duration, use_median, tree, terminal_width
        )
    )

    return Group(*parts)


def _watch_file(
    file_path: str,
    sort_key: str,
    reverse: bool,
    index_duration: bool,
    use_median: bool,
    tree: bool,
    interval: float,
) -> int:
    """Watch an MCAP file for changes and display live-updating statistics."""
    path = Path(file_path)
    last_size = 0
    info_data = None
    terminal_width = console.width

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                try:
                    current_size = path.stat().st_size
                except OSError as e:
                    console.print(f"[red]Error:[/] {e}")
                    return 1

                if current_size == last_size and info_data is not None:
                    time.sleep(interval)
                    continue

                needs_full_rescan = info_data is None or current_size < last_size

                try:
                    with path.open("rb") as f:
                        if needs_full_rescan:
                            info_data = rebuild_summary(
                                f,
                                validate_crc=False,
                                calculate_channel_sizes=True,
                                exact_sizes=False,
                            )
                        else:
                            assert info_data is not None
                            f.seek(info_data.next_offset)
                            info_data = rebuild_summary(
                                f,
                                validate_crc=False,
                                calculate_channel_sizes=True,
                                exact_sizes=False,
                                initial_state=info_data,
                                skip_magic=True,
                            )
                except OSError as e:
                    console.print(f"[red]Error reading file:[/] {e}")
                    return 1
                except (McapError, ValueError, AssertionError):
                    # File may be partially written; wait and retry
                    time.sleep(interval)
                    continue

                last_size = current_size

                data = info_to_dict(info_data, file_path, current_size)
                has_chunk_info = info_data.chunk_information is not None
                display = _build_info_display(
                    data,
                    has_chunk_info,
                    sort_key,
                    reverse,
                    index_duration,
                    use_median,
                    tree,
                    terminal_width,
                )

                now = datetime.now(tz=timezone.utc).astimezone().strftime("%H:%M:%S")
                status = Text.from_markup(
                    f"\n[dim]Watching... Last update: {now}"
                    f" | Size: {bytes_to_human(current_size)}"
                    f" | Ctrl+C to stop[/]"
                )
                live.update(Group(display, status))

                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/]")

    return 0


def _output_json(
    all_outputs: list[McapInfoOutput],
    compress: bool,
) -> None:
    """Output JSON (single file -> dict, multiple files -> list)."""
    output_data = all_outputs[0] if len(all_outputs) == 1 else all_outputs
    output_json = json.dumps(output_data)

    if compress:
        compressed_output = gzip.compress(output_json.encode("utf-8"))
        output_b64 = base64.b64encode(compressed_output).decode("utf-8")
        print(output_b64)  # noqa: T201
    else:
        print(output_json)  # noqa: T201


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
    json_output: Annotated[
        bool,
        Parameter(
            name=["--json"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
    compress: Annotated[
        bool,
        Parameter(
            name=["--compress"],
            group=DISPLAY_GROUP,
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
    watch: Annotated[
        bool,
        Parameter(
            name=["-w", "--watch"],
            group=PROCESSING_GROUP,
        ),
    ] = False,
    watch_interval: Annotated[
        float,
        Parameter(
            name=["--watch-interval"],
            group=PROCESSING_GROUP,
        ),
    ] = 0.5,
    link: Annotated[
        bool,
        Parameter(
            name=["-l", "--link"],
            group=DISPLAY_GROUP,
        ),
    ] = False,
) -> int:
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
    json_output
        Output as JSON instead of Rich tables.
    compress
        Compressed JSON output using gzip+base64 (requires --json).
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
    watch
        Watch the file for changes and display live-updating statistics.
        Requires exactly one local file. Incompatible with --json and --compress.
    watch_interval
        Polling interval in seconds for watch mode (default: 0.5).
    link
        Generate a shareable URL for the web inspector. Incompatible with
        --json, --compress, and --watch.

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

    # JSON output
    pymcap-cli info recording.mcap --json

    # Watch a file for live updates
    pymcap-cli info recording.mcap --watch
    ```
    """
    # Validate input
    if not files:
        if json_output:
            print('{"error": "At least one file must be specified"}', file=sys.stderr)  # noqa: T201
        else:
            console.print("[red]Error:[/] At least one file must be specified")
        return 1

    if compress and not json_output:
        console.print("[red]Error:[/] --compress requires --json")
        return 1

    if link:
        if json_output:
            console.print("[red]Error:[/] --link is incompatible with --json")
            return 1
        if compress:
            console.print("[red]Error:[/] --link is incompatible with --compress")
            return 1
        if watch:
            console.print("[red]Error:[/] --link is incompatible with --watch")
            return 1

    if watch:
        if json_output:
            console.print("[red]Error:[/] --watch is incompatible with --json")
            return 1
        if compress:
            console.print("[red]Error:[/] --watch is incompatible with --compress")
            return 1
        if len(files) != 1:
            console.print("[red]Error:[/] --watch requires exactly one file")
            return 1
        return _watch_file(
            files[0], sort.value, reverse, index_duration, median, tree, watch_interval
        )

    # Link output mode
    if link:
        mode: ScanMode = "exact" if exact_sizes else ("rebuild" if rebuild else "summary")
        for file in files:
            with open_input(file, buffering=0, debug=debug) as (f_buffered, file_size):
                info_data = read_or_rebuild_info(
                    f_buffered, file_size, rebuild=rebuild, exact_sizes=exact_sizes
                )
            data = info_to_dict(info_data, str(file), file_size)
            url = generate_link(data, str(file), file_size, mode)
            print(url)  # noqa: T201
        return 0

    # JSON output mode
    if json_output:
        all_outputs: list[McapInfoOutput] = []
        for file in files:
            with open_input(file, buffering=0, debug=debug) as (f_buffered, file_size):
                info_data = read_or_rebuild_info(
                    f_buffered, file_size, rebuild=rebuild, exact_sizes=exact_sizes
                )
            data = info_to_dict(info_data, str(file), file_size)
            all_outputs.append(data)
        _output_json(all_outputs, compress)
        return 0

    # Rich table output mode
    for i, file in enumerate(files):
        if i > 0:
            console.print("\n" + "=" * 80 + "\n")

        with open_input(file, buffering=0, debug=debug) as (f_buffered, file_size):
            info_data = read_or_rebuild_info(
                f_buffered, file_size, rebuild=rebuild, exact_sizes=exact_sizes
            )

        # Get structured data
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
        has_channel_durations = any(ch.get("duration_ns") is not None for ch in data["channels"])
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
        console.print(
            _build_info_display(
                data, has_chunk_info, sort.value, reverse, index_duration, median, tree
            )
        )

        # Show shareable web inspector link
        mode: ScanMode = "exact" if exact_sizes else ("rebuild" if rebuild else "summary")
        url = generate_link(data, str(file), file_size, mode)
        console.print(f"\n[link={url}][dim]View in web inspector[/dim][/]")

    return 0
