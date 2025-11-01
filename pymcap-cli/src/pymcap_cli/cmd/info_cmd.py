from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from small_mcap import ChunkIndex, InvalidMagicError

from pymcap_cli.debug_wrapper import DebugStreamWrapper
from pymcap_cli.rebuild import read_info, rebuild_info
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

    return parser


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
    info_table.add_row(
        "Duration:",
        f"[yellow]{duration_ns / 1_000_000:.2f} ms[/yellow] [cyan]({duration_human})[/cyan]",
    )
    info_table.add_row("Start:", f"[cyan]{date_start}[/cyan]")
    info_table.add_row("End:", f"[cyan]{date_end}[/cyan]")

    console.print(info_table)

    # compression, count, compressed size, uncompressed size, min size, max size
    chunks: dict[str, tuple[int, int, int, int, int]] = {}
    for x in summary.chunk_indexes:
        count, compressed, uncompressed, min_size, max_size = chunks.get(
            x.compression, (0, 0, 0, 0, 0)
        )
        chunks[x.compression] = (
            count + 1,
            compressed + x.compressed_size,
            uncompressed + x.uncompressed_size,
            min(min_size, x.uncompressed_size) if min_size else x.uncompressed_size,
            max(max_size, x.uncompressed_size) if max_size else x.uncompressed_size,
        )

    compression_table = Table()
    compression_table.add_column("Type", style="bold cyan")
    compression_table.add_column("Chunks", justify="right", style="green")
    compression_table.add_column("Compressed", justify="right", style="yellow")
    compression_table.add_column("Uncompressed", justify="right", style="yellow")
    compression_table.add_column("Ratio", justify="right", style="magenta")
    compression_table.add_column("Min Size", justify="right", style="yellow")
    compression_table.add_column("Max Size", justify="right", style="yellow")

    chunks_count = len(summary.chunk_indexes)
    for compression_type, (
        count,
        compressed_size,
        uncompressed_size,
        min_size,
        max_size,
    ) in chunks.items():
        ratio = (
            f"{compressed_size / uncompressed_size * 100:.1f}%" if uncompressed_size > 0 else "N/A"
        )
        compression_table.add_row(
            compression_type,
            f"{count}/{chunks_count}",
            bytes_to_human(compressed_size),
            bytes_to_human(uncompressed_size),
            ratio,
            bytes_to_human(min_size),
            bytes_to_human(max_size),
        )

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

    # Channels table (improved)
    channels_table = Table()
    channels_table.add_column("Topic", style="bold white")
    channels_table.add_column("Schema", style="blue")
    channels_table.add_column("Msgs", justify="right", style="green")
    channels_table.add_column("Hz", justify="right", style="yellow")
    if info.channel_sizes:
        channels_table.add_column("Size", justify="right", style="yellow")
        channels_table.add_column("B/s", justify="right", style="magenta")
        channels_table.add_column("B/msg", justify="right", style="magenta")

    for channel in sorted(summary.channels.values(), key=lambda c: c.topic):
        channel_id = channel.id
        count = statistics.channel_message_counts.get(channel_id, 0)
        hz = count / (duration_ns / 1_000_000_000) if duration_ns > 0 else 0
        schema = summary.schemas.get(channel.schema_id)
        row = [
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
            row.append(bytes_to_human(int(bps)) + "/s")
            b_per_msg = info.channel_sizes.get(channel_id, 0) / count if count > 0 else 0
            row.append(bytes_to_human(int(b_per_msg)) + "/msg")
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
