from __future__ import annotations

import io
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from small_mcap import ChunkIndex, InvalidMagicError, RebuildInfo

from pymcap_cli.debug_wrapper import DebugStreamWrapper
from pymcap_cli.utils import read_info, rebuild_info

if TYPE_CHECKING:
    from small_mcap import Channel, Schema

    from pymcap_cli.types import (
        ChannelInfo,
        CompressionStats,
        McapInfoOutput,
        MessageDistribution,
        SchemaInfo,
        Stats,
    )


app = typer.Typer()


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


def _calculate_channel_durations(info: RebuildInfo) -> dict[int, int]:
    """Calculate per-channel duration in nanoseconds from message indexes.

    Returns a dict mapping channel_id -> duration_ns (last_msg_time - first_msg_time).
    Uses message index information from rebuild.
    """
    if not info.chunk_information:
        return {}

    # Track first and last message times per channel
    channel_times: dict[int, tuple[float, float]] = defaultdict(
        lambda: (float("inf"), float("-inf"))
    )

    for chunk_info_list in info.chunk_information.values():
        for chunk_info in chunk_info_list:
            if not chunk_info.records:
                continue

            timestamps = [record[0] for record in chunk_info.records]
            if not timestamps:
                continue

            msg_first, msg_last = min(timestamps), max(timestamps)
            prev_first, prev_last = channel_times[chunk_info.channel_id]
            channel_times[chunk_info.channel_id] = (
                min(prev_first, msg_first),
                max(prev_last, msg_last),
            )

    # Calculate durations
    return {
        channel_id: int(last_time - first_time)
        for channel_id, (first_time, last_time) in channel_times.items()
    }


def _calculate_optimal_bucket_count(duration_ns: int) -> int:
    """Calculate optimal bucket count to produce round time durations.

    Tries bucket counts from 20-80 and selects the one that produces
    the most "round" duration (1ms, 10ms, 100ms, 1s, 2s, 5s, 10s, 30s, 1min, etc.).

    Args:
        duration_ns: Total duration in nanoseconds

    Returns:
        Optimal bucket count between 20 and 80
    """
    # Target "round" durations in nanoseconds
    round_durations = [
        1_000_000,  # 1ms
        2_000_000,  # 2ms
        5_000_000,  # 5ms
        10_000_000,  # 10ms
        20_000_000,  # 20ms
        50_000_000,  # 50ms
        100_000_000,  # 100ms
        200_000_000,  # 200ms
        500_000_000,  # 500ms
        1_000_000_000,  # 1s
        2_000_000_000,  # 2s
        5_000_000_000,  # 5s
        10_000_000_000,  # 10s
        20_000_000_000,  # 20s
        30_000_000_000,  # 30s
        60_000_000_000,  # 1min
        120_000_000_000,  # 2min
        300_000_000_000,  # 5min
        600_000_000_000,  # 10min
        1_200_000_000_000,  # 20min
        1_800_000_000_000,  # 30min
        3_600_000_000_000,  # 1hr
    ]

    min_buckets = 20
    max_buckets = 80
    best_bucket_count = 50  # Default middle value
    min_error = float("inf")

    for bucket_count in range(min_buckets, max_buckets + 1):
        bucket_duration = duration_ns / bucket_count

        # Find the closest round duration
        for round_duration in round_durations:
            error = abs(bucket_duration - round_duration) / round_duration
            if error < min_error:
                min_error = error
                best_bucket_count = bucket_count

    return best_bucket_count


def _calculate_message_distribution(
    info: RebuildInfo, start_time: int, end_time: int
) -> MessageDistribution:
    """Calculate message distribution across dynamic time buckets.

    Uses 20-80 buckets, choosing the count that produces the most "round"
    time duration per bucket (1s, 10s, 1min, etc.).

    Args:
        info: Info object containing chunk_information with message records
        start_time: Global message start time in nanoseconds
        end_time: Global message end time in nanoseconds

    Returns:
        MessageDistribution with optimal bucket count and message counts
    """
    duration_ns = end_time - start_time

    # Handle edge case: zero duration
    if duration_ns <= 0:
        bucket_count = 20
        return {
            "bucket_count": bucket_count,
            "bucket_duration_ns": 0,
            "message_counts": [0] * bucket_count,
            "max_count": 0,
        }

    # Calculate optimal bucket count for round durations
    bucket_count = _calculate_optimal_bucket_count(duration_ns)

    # Initialize empty buckets
    message_counts = [0] * bucket_count
    bucket_duration_ns = duration_ns // bucket_count

    # Process all messages from chunk_information
    if info.chunk_information:
        for chunk_info_list in info.chunk_information.values():
            for chunk_info in chunk_info_list:
                if not chunk_info.records:
                    continue

                for timestamp, *_ in chunk_info.records:
                    # Calculate which bucket this message belongs to
                    offset = timestamp - start_time
                    bucket_idx = int(offset / bucket_duration_ns)
                    # Clamp to valid range (handle edge case where timestamp == end_time)
                    bucket_idx = min(bucket_idx, bucket_count - 1)
                    if bucket_idx >= 0:
                        message_counts[bucket_idx] += 1

    max_count = max(message_counts) if message_counts else 0

    return {
        "bucket_count": bucket_count,
        "bucket_duration_ns": bucket_duration_ns,
        "message_counts": message_counts,
        "max_count": max_count,
    }


def _calculate_per_channel_distributions(
    info: RebuildInfo, bucket_count: int, bucket_duration_ns: int, start_time: int
) -> dict[int, list[int]]:
    """Calculate message distribution per channel using global time buckets.

    Uses the same bucket parameters as the global distribution to allow
    direct comparison across channels.

    Args:
        info: Info object containing chunk_information with message records
        bucket_count: Number of time buckets (from global distribution)
        bucket_duration_ns: Duration of each bucket in nanoseconds
        start_time: Global message start time in nanoseconds

    Returns:
        Dictionary mapping channel_id to list of message counts per bucket
    """
    # Initialize empty distributions for all channels
    channel_distributions: dict[int, list[int]] = {}

    # Handle edge case: zero duration or no bucket duration
    if bucket_duration_ns <= 0:
        return channel_distributions

    # Process all messages from chunk_information
    if info.chunk_information:
        for chunk_info_list in info.chunk_information.values():
            for chunk_info in chunk_info_list:
                channel_id = chunk_info.channel_id

                # Initialize this channel's distribution if not seen yet
                if channel_id not in channel_distributions:
                    channel_distributions[channel_id] = [0] * bucket_count

                if not chunk_info.records:
                    continue

                for timestamp, *_ in chunk_info.records:
                    # Calculate which bucket this message belongs to
                    offset = timestamp - start_time
                    bucket_idx = int(offset / bucket_duration_ns)
                    # Clamp to valid range
                    bucket_idx = min(bucket_idx, bucket_count - 1)
                    if bucket_idx >= 0:
                        channel_distributions[channel_id][bucket_idx] += 1

    return channel_distributions


def info_to_dict(info: RebuildInfo, file_path: str, file_size: int) -> McapInfoOutput:
    """Transform MCAP Info object into a JSON-serializable dictionary.

    Args:
        info: RebuildInfo from small-mcap containing header, summary, and optional sizes
        file_path: Path to the MCAP file
        file_size: File size in bytes

    Returns:
        Dictionary containing complete MCAP statistics and metadata, ready for JSON serialization
    """
    header = info.header
    summary = info.summary
    if header is None:
        raise ValueError("MCAP header is missing")
    statistics = summary.statistics
    if statistics is None:
        raise ValueError("MCAP statistics are missing")

    # Calculate global duration
    duration_ns = statistics.message_end_time - statistics.message_start_time

    # Aggregate chunk statistics by compression type
    chunks: dict[str, ChunkStats] = defaultdict(ChunkStats)
    for x in summary.chunk_indexes:
        stats = chunks[x.compression]
        stats.count += 1
        stats.compressed_size += x.compressed_size
        stats.uncompressed_size += x.uncompressed_size
        stats.uncompressed_sizes.append(x.uncompressed_size)
        stats.durations_ns.append(x.message_end_time - x.message_start_time)
        if info.chunk_information and (cinfo := info.chunk_information.get(x.chunk_start_offset)):
            stats.message_count += sum(len(ci.records) for ci in cinfo)

    # Calculate chunk overlaps
    max_concurrent, overlap_size = _calculate_chunk_overlaps(summary.chunk_indexes)

    # Calculate per-channel durations (always, when available)
    channel_durations: dict[int, int] = {}
    if info.chunk_information:
        channel_durations = _calculate_channel_durations(info)

    # Calculate message distribution
    message_distribution = _calculate_message_distribution(
        info, statistics.message_start_time, statistics.message_end_time
    )

    # Calculate per-channel distributions using the same buckets
    per_channel_distributions = _calculate_per_channel_distributions(
        info,
        message_distribution["bucket_count"],
        message_distribution["bucket_duration_ns"],
        statistics.message_start_time,
    )

    # Build JSON output structure
    output: McapInfoOutput = {
        "file": {
            "path": file_path,
            "size_bytes": file_size,
        },
        "header": {
            "library": header.library,
            "profile": header.profile,
        },
        "statistics": {
            "message_count": statistics.message_count,
            "chunk_count": statistics.chunk_count,
            "channel_count": statistics.channel_count,
            "attachment_count": statistics.attachment_count,
            "metadata_count": statistics.metadata_count,
            "message_start_time": statistics.message_start_time,
            "message_end_time": statistics.message_end_time,
            "duration_ns": duration_ns,
        },
        "chunks": {
            "by_compression": {},
            "overlaps": {
                "max_concurrent": max_concurrent,
                "max_concurrent_bytes": overlap_size,
            },
            "indexes": [],
        },
        "channels": [],
        "schemas": [],
        "message_distribution": message_distribution,
    }

    # Add chunk statistics by compression type
    for compression_type, chunk_stats in chunks.items():
        output["chunks"]["by_compression"][compression_type] = _build_chunk_compression_stats(
            chunk_stats
        )

    # Add chunk indexes
    for chunk_index in summary.chunk_indexes:
        output["chunks"]["indexes"].append(
            {
                "compression": chunk_index.compression,
                "compressed_size": chunk_index.compressed_size,
                "uncompressed_size": chunk_index.uncompressed_size,
                "message_start_time": chunk_index.message_start_time,
                "message_end_time": chunk_index.message_end_time,
                "chunk_start_offset": chunk_index.chunk_start_offset,
            }
        )

    # Add channel information
    for channel in summary.channels.values():
        channel_id = channel.id
        count = statistics.channel_message_counts.get(channel_id, 0)
        schema = summary.schemas.get(channel.schema_id)
        channel_size = info.channel_sizes.get(channel_id) if info.channel_sizes else None
        ch_duration_ns = channel_durations.get(channel_id)
        ch_distribution = per_channel_distributions.get(channel_id, [])

        output["channels"].append(
            _build_channel_dict(
                channel=channel,
                message_count=count,
                schema_name=schema.name if schema else None,
                channel_size=channel_size,
                channel_duration_ns=ch_duration_ns,
                global_duration_ns=duration_ns,
                message_distribution=ch_distribution,
            )
        )

    # Add schema information
    for schema_id, schema in summary.schemas.items():
        output["schemas"].append(_build_schema_dict(schema_id, schema))

    return output


def _calculate_stats(values: list[int] | list[float]) -> Stats:
    """Calculate min, max, and average from a list of values."""
    if not values:
        return {"minimum": 0, "maximum": 0, "average": 0.0, "median": 0.0}
    return {
        "minimum": min(values),
        "maximum": max(values),
        "average": sum(values) / len(values),
        "median": sorted(values)[len(values) // 2] if values else 0,
    }


def _build_chunk_compression_stats(chunk_stats: ChunkStats) -> CompressionStats:
    """Build chunk compression statistics dict for JSON output."""
    return {
        "count": chunk_stats.count,
        "compressed_size": chunk_stats.compressed_size,
        "uncompressed_size": chunk_stats.uncompressed_size,
        "compression_ratio": (
            chunk_stats.compressed_size / chunk_stats.uncompressed_size
            if chunk_stats.uncompressed_size > 0
            else 0
        ),
        "message_count": chunk_stats.message_count,
        "size_stats": _calculate_stats(chunk_stats.uncompressed_sizes),
        "duration_stats": _calculate_stats(chunk_stats.durations_ns),
    }


def _build_channel_dict(
    channel: Channel,
    message_count: int,
    schema_name: str | None,
    channel_size: int | None,
    channel_duration_ns: int | None,
    global_duration_ns: int,
    message_distribution: list[int],
) -> ChannelInfo:
    """Build channel information dict for JSON output with calculated metrics."""
    # Calculate Hz metrics
    hz_global = (
        message_count / (global_duration_ns / 1_000_000_000) if global_duration_ns > 0 else 0
    )
    hz_channel = None
    if channel_duration_ns is not None and channel_duration_ns > 0:
        hz_channel = message_count / (channel_duration_ns / 1_000_000_000)

    # Calculate size-based metrics
    bps = None
    b_per_msg = None
    if channel_size is not None:
        if global_duration_ns > 0:
            bps = channel_size / (global_duration_ns / 1_000_000_000)
        if message_count > 0:
            b_per_msg = channel_size / message_count

    return {
        "id": channel.id,
        "topic": channel.topic,
        "schema_id": channel.schema_id,
        "schema_name": schema_name,
        "message_count": message_count,
        "size_bytes": channel_size,
        "duration_ns": channel_duration_ns,
        "hz": hz_global,
        "hz_channel": hz_channel,
        "bytes_per_second": bps,
        "bytes_per_message": b_per_msg,
        "message_distribution": message_distribution,
    }


def _build_schema_dict(schema_id: int, schema: Schema) -> SchemaInfo:
    """Build schema information dict for JSON output."""
    return {
        "id": schema_id,
        "name": schema.name,
    }


@dataclass(slots=True)
class ChunkStats:
    count: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
    uncompressed_sizes: list[int] = field(default_factory=list)

    message_count: int = 0

    durations_ns: list[float] = field(default_factory=list)


@app.command(name="info-json")
def info_json(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to the MCAP file to analyze",
        ),
    ],
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            "-r",
            help="Rebuild the MCAP file from scratch",
        ),
    ] = False,
    exact_sizes: Annotated[
        bool,
        typer.Option(
            "--exact-sizes",
            "-e",
            help="Use exact sizes for message data (may be slower, requires --rebuild)",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode",
        ),
    ] = False,
) -> None:
    """Output MCAP file statistics as JSON with all available data."""
    file_size = file.stat().st_size

    debug_wrapper = None
    with file.open("rb", buffering=0) as f_raw:
        if debug:
            debug_wrapper = DebugStreamWrapper(f_raw)
            f_buffered: io.BufferedReader = io.BufferedReader(debug_wrapper, buffer_size=1024)
        else:
            f_buffered = io.BufferedReader(f_raw, buffer_size=1024)

        if rebuild:
            info = rebuild_info(f_buffered, file_size, exact_sizes=exact_sizes)
        else:
            try:
                info = read_info(f_buffered)
            except InvalidMagicError:
                if not debug:
                    # Silently rebuild if invalid magic and not in debug mode
                    info = rebuild_info(f_buffered, file_size, exact_sizes=exact_sizes)
                else:
                    raise

    if debug_wrapper:
        debug_wrapper.print_stats(file_size)

    # Transform info to JSON-serializable dict
    output = info_to_dict(info, str(file), file_size)

    # Output JSON
    print(json.dumps(output, indent=2))  # noqa: T201
