from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from small_mcap import Channel, ChunkIndex, InvalidMagicError

from pymcap_cli.debug_wrapper import DebugStreamWrapper
from pymcap_cli.rebuild import Info, read_info, rebuild_info
from pymcap_cli.utils import bytes_to_human

if TYPE_CHECKING:
    import argparse

console = Console()


def _calculate_chunk_overlaps(chunk_indexes: list[ChunkIndex]) -> tuple[int, int]:
    if len(chunk_indexes) <= 1:
        return 0, 0

    # Create events for start and end of each chunk
    # Each event is (time, event_type, chunk_id)
    # event_type: 0 = start, 1 = end (so starts come before ends at same time)
    events: list[tuple[int, int, int]] = []
    for idx, chunk in enumerate(chunk_indexes):
        events.append((chunk.message_start_time, 0, idx))
        events.append((chunk.message_end_time, 1, idx))

    # Sort by time, then by event type (starts before ends)
    events.sort(key=lambda e: (e[0], e[1]))

    # Map of chunk ID to ChunkIndex for currently active chunks
    current_active: dict[int, ChunkIndex] = {}
    # Chunks active at first point of max concurrency
    max_concurrent_chunks: dict[int, ChunkIndex] = {}
    max_concurrent_bytes = 0

    for _, event_type, chunk_id in events:
        if event_type == 0:  # Start event
            current_active[chunk_id] = chunk_indexes[chunk_id]
            if len(current_active) > len(max_concurrent_chunks):
                # Save the chunks that are active at this point of maximum concurrency
                max_concurrent_chunks = current_active.copy()
                # Sum the uncompressed size of chunks at the point of maximum concurrency
                max_concurrent_bytes = max(
                    max_concurrent_bytes,
                    sum(chunk.uncompressed_size for chunk in max_concurrent_chunks.values()),
                )
        else:  # End event
            current_active.pop(chunk_id, None)

    return len(max_concurrent_chunks), max_concurrent_bytes


def _calculate_channel_durations(info: Info) -> dict[int, int]:
    """Calculate per-channel duration in nanoseconds from message indexes.

    Returns a dict mapping channel_id -> duration_ns (last_msg_time - first_msg_time).
    Uses message index information from rebuild.
    """
    channel_durations: dict[int, int] = {}

    # If no chunk information available, return empty dict
    if not info.chunk_information:
        return channel_durations

    # Track first and last message times per channel
    channel_times: dict[int, tuple[int, int]] = {}  # channel_id -> (first_time, last_time)

    for chunk_info_list in info.chunk_information.values():
        for chunk_info in chunk_info_list:
            channel_id = chunk_info.channel_id

            # Get timestamps from message index records
            if not chunk_info.records:
                continue

            # Records are list of (timestamp, offset) tuples
            timestamps = [record[0] for record in chunk_info.records]

            if not timestamps:
                continue

            msg_first = min(timestamps)
            msg_last = max(timestamps)

            if channel_id in channel_times:
                prev_first, prev_last = channel_times[channel_id]
                channel_times[channel_id] = (min(prev_first, msg_first), max(prev_last, msg_last))
            else:
                channel_times[channel_id] = (msg_first, msg_last)

    # Calculate durations
    for channel_id, (first_time, last_time) in channel_times.items():
        channel_durations[channel_id] = last_time - first_time

    return channel_durations


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
        help="Calculate Hz per channel based on each channel's first/last message times rather than global MCAP duration",
    )

    return parser


@dataclass(slots=True)
class ChunkStats:
    count: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
    uncompressed_sizes: list[int] = field(default_factory=list)

    message_count: int = 0

    durations_ns: list[float] = field(default_factory=list)

    @property
    def size_stats(self) -> tuple[str, str, str]:
        if not self.uncompressed_sizes:
            return "0.0", "0.0", "0.0"
        return (
            bytes_to_human(min(self.uncompressed_sizes)),
            bytes_to_human(sum(self.uncompressed_sizes) / len(self.uncompressed_sizes)),
            bytes_to_human(max(self.uncompressed_sizes)),
        )

    @property
    def duration_stats(self) -> tuple[str, str, str]:
        if not self.durations_ns:
            return "0.0", "0.0", "0.0"
        divider = 1_000_000

        return (
            f"{min(self.durations_ns) / divider:.2f}ms",
            f"{sum(self.durations_ns) / len(self.durations_ns) / divider:.2f}ms",
            f"{max(self.durations_ns) / divider:.2f}ms",
        )

    @property
    def ratio(self) -> str:
        return (
            f"{self.compressed_size / self.uncompressed_size * 100:.1f}%"
            if self.uncompressed_size > 0
            else "N/A"
        )


def handle_command(args: argparse.Namespace) -> None:
    """Handle the info command execution."""

    file = Path(args.file)
    file_size = file.stat().st_size

    debug_wrapper: DebugStreamWrapper | None = None
    with file.open("rb", buffering=0) as f_raw:
        f_buffered: io.BufferedReader
        if args.debug:
            debug_wrapper = DebugStreamWrapper(f_raw)
            f_buffered = io.BufferedReader(debug_wrapper, buffer_size=1024)
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

    header = info.header
    summary = info.summary
    assert header is not None, "Header should not be None"
    statistics = summary.statistics
    assert statistics is not None, "Statistics should not be None"

    # File info panel
    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold blue")
    info_table.add_column()
    info_table.add_row("File:", f"[green]{file}[/green]")
    info_table.add_row("Library:", f"[yellow]{header.library}[/yellow]")
    info_table.add_row("Profile:", f"[yellow]{header.profile}[/yellow]")

    # Timing and messages panel
    duration_ns = statistics.message_end_time - statistics.message_start_time
    duration_human = timedelta(milliseconds=duration_ns / 1_000_000)
    date_start = datetime.fromtimestamp(statistics.message_start_time / 1_000_000_000)
    date_end = datetime.fromtimestamp(statistics.message_end_time / 1_000_000_000)

    info_table.add_row("Messages:", f"[green]{statistics.message_count:,}[/green]")
    info_table.add_row("Chunks:", f"[cyan]{statistics.chunk_count}[/cyan]")
    info_table.add_row(
        "Duration:",
        f"[yellow]{duration_ns / 1_000_000:.2f} ms[/yellow] [cyan]({duration_human})[/cyan]",
    )
    info_table.add_row("Start:", f"[cyan]{date_start}[/cyan]")
    info_table.add_row("End:", f"[cyan]{date_end}[/cyan]")

    console.print(info_table)

    # compression, count, compressed size, uncompressed size, min size, max size
    chunks: dict[str, ChunkStats] = {}
    for x in summary.chunk_indexes:
        stats = chunks.get(x.compression, ChunkStats())
        stats.count += 1
        stats.compressed_size += x.compressed_size
        stats.uncompressed_size += x.uncompressed_size
        stats.uncompressed_sizes.append(x.uncompressed_size)
        stats.durations_ns.append(x.message_end_time - x.message_start_time)
        if info.chunk_information and (cinfo := info.chunk_information.get(x.chunk_start_offset)):
            stats.message_count += sum(len(ci.records) for ci in cinfo)
        chunks[x.compression] = stats

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
    if info.chunk_information:
        compression_table.add_column("Msgs", justify="right", style="cyan")

    for compression_type, chunk in chunks.items():
        row = [
            compression_type,
            f"{chunk.count}",
            bytes_to_human(chunk.compressed_size),
            bytes_to_human(chunk.uncompressed_size),
            chunk.ratio,
            *chunk.size_stats,
            *chunk.duration_stats,
        ]
        if info.chunk_information:
            row.append(f"{chunk.message_count}")

        compression_table.add_row(*row)

    console.print(compression_table)

    # Calculate chunk overlaps
    max_concurrent, overlap_size = _calculate_chunk_overlaps(summary.chunk_indexes)
    # Display overlap information if there are overlapping chunks
    if max_concurrent > 1:
        chunk_stats = Table.grid(padding=(0, 1))
        chunk_stats.add_column(style="bold blue")
        chunk_stats.add_column()
        chunk_stats.add_row(
            "Overlaps:",
            f"[green]{max_concurrent}[/green] max concurrent, "
            f"[yellow]{bytes_to_human(overlap_size)}[/yellow] max total size at once",
        )
        console.print(chunk_stats)

    # Warn if sorting by size fields when channel_sizes is unavailable
    if args.sort in ["size", "bps", "b_per_msg"] and not info.channel_sizes:
        console.print(
            "[yellow]Warning:[/yellow] Sorting by size fields requires channel size data. "
            "Use [cyan]--rebuild[/cyan] to get accurate size information."
        )
        console.print()

    # Calculate per-channel durations if --index-duration is enabled
    channel_durations: dict[int, int] = {}
    if args.index_duration:
        channel_durations = _calculate_channel_durations(info)
        if not channel_durations and not info.chunk_information:
            console.print(
                "[yellow]Warning:[/yellow] --index-duration requires message index data. "
                "Use [cyan]--rebuild[/cyan] to get per-channel timing information. "
                "Falling back to global duration."
            )
            console.print()

    # Create sort key function
    def get_sort_key(channel: Channel) -> str | int | float:
        channel_id = channel.id
        count = statistics.channel_message_counts.get(channel_id, 0)

        if args.sort == "topic":
            return channel.topic
        if args.sort == "id":
            return channel_id
        if args.sort == "msgs":
            return count
        if args.sort == "size":
            return info.channel_sizes.get(channel_id, 0) if info.channel_sizes else 0
        if args.sort == "hz":
            # Use per-channel duration if available and flag is set
            if args.index_duration and channel_id in channel_durations:
                ch_duration_ns = channel_durations[channel_id]
                return count / (ch_duration_ns / 1_000_000_000) if ch_duration_ns > 0 else 0
            return count / (duration_ns / 1_000_000_000) if duration_ns > 0 else 0
        if args.sort == "bps":
            if not info.channel_sizes or duration_ns == 0:
                return 0
            return info.channel_sizes.get(channel_id, 0) / (duration_ns / 1_000_000_000)
        if args.sort == "b_per_msg":
            if not info.channel_sizes or count == 0:
                return 0
            return info.channel_sizes.get(channel_id, 0) / count
        if args.sort == "schema":
            schema = summary.schemas.get(channel.schema_id)
            return schema.name if schema else ""
        return channel.topic  # fallback

    # Channels table (improved)
    channels_table = Table()
    channels_table.add_column("ID", style="bold blue", no_wrap=True, justify="right")
    channels_table.add_column("Topic", style="bold white")
    channels_table.add_column("Schema", style="blue")
    channels_table.add_column("Msgs", justify="right", style="green")
    channels_table.add_column("Hz", justify="right", style="yellow")
    if info.channel_sizes:
        channels_table.add_column("Size", justify="right", style="yellow")
        channels_table.add_column("B/s", justify="right", style="magenta")
        channels_table.add_column("B/msg", justify="right", style="magenta")

    for channel in sorted(summary.channels.values(), key=get_sort_key, reverse=args.reverse):
        channel_id = channel.id
        count = statistics.channel_message_counts.get(channel_id, 0)

        ch_duration_ns = channel_durations.get(channel_id, duration_ns)
        hz = count / (ch_duration_ns / 1_000_000_000) if ch_duration_ns > 0 else 0

        schema = summary.schemas.get(channel.schema_id)
        row = [
            str(channel_id),
            channel.topic,
            schema.name if schema else "[dim]unknown[/dim]",
            f"{count:,}",
            f"{hz:.2f}",
        ]
        if info.channel_sizes:
            row.append(bytes_to_human(info.channel_sizes.get(channel_id, 0)))
            bps = (
                info.channel_sizes.get(channel_id, 0) / (duration_ns / 1_000_000_000)
                if duration_ns > 0
                else 0
            )
            row.append(bytes_to_human(int(bps)))
            b_per_msg = info.channel_sizes.get(channel_id, 0) / count if count > 0 else 0
            row.append(bytes_to_human(int(b_per_msg)))
        channels_table.add_row(*row)

    console.print(channels_table)

    # Summary stats panel
    summary_info = Table.grid(padding=(0, 1))
    summary_info.add_column(style="bold blue")
    summary_info.add_column(justify="right")
    summary_info.add_row("Channels:", f"[green]{statistics.channel_count}[/green]")
    summary_info.add_row("Attachments:", f"[yellow]{statistics.attachment_count}[/yellow]")
    summary_info.add_row("Metadata:", f"[cyan]{statistics.metadata_count}[/cyan]")

    console.print(summary_info)
