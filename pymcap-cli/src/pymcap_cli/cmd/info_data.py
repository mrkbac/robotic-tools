"""Pure data logic for MCAP info computation.

Extracted from info_json_cmd.py — no CLI or Rich dependencies.
"""

import heapq
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from small_mcap import Channel, ChunkIndex, RebuildInfo, Schema

from pymcap_cli.info_types import (
    AttachmentInfo,
    ChannelInfo,
    CompressionStats,
    McapInfoOutput,
    MessageDistribution,
    MetadataInfo,
    PartialStats,
    SchemaInfo,
    Stats,
)


@dataclass(slots=True)
class ChunkStats:
    count: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
    uncompressed_sizes: list[int] = field(default_factory=list)

    message_count: int = 0

    durations_ns: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ChannelStatistics:
    """Collected statistics for a single channel from a single-pass iteration."""

    channel_id: int
    first_time: int = sys.maxsize
    last_time: int = 0
    message_count: int = 0
    # For interval calculation: track last timestamp and collect intervals directly
    chunk_last_timestamp: int = -1  # Last timestamp from previous chunk for this channel
    intervals: list[int] = field(default_factory=list)


def _calculate_chunk_overlaps(chunk_indexes: list[ChunkIndex]) -> tuple[int, int]:
    if len(chunk_indexes) <= 1:
        return 0, 0

    # Min-heap of (end_time, uncompressed_size) for active chunks
    active_heap: list[tuple[int, int]] = []
    max_concurrent = 0
    max_concurrent_bytes = 0

    for chunk in sorted(chunk_indexes, key=lambda c: c.message_start_time):
        # Remove chunks that have ended before this chunk starts
        while active_heap and active_heap[0][0] < chunk.message_start_time:
            heapq.heappop(active_heap)

        # Add current chunk
        heapq.heappush(active_heap, (chunk.message_end_time, chunk.uncompressed_size))

        max_concurrent = max(max_concurrent, len(active_heap))
        max_concurrent_bytes = max(max_concurrent_bytes, sum(size for _, size in active_heap))

    return max_concurrent, max_concurrent_bytes


def _collect_channel_statistics(
    info: RebuildInfo, start_time: int, bucket_count: int, bucket_duration_ns: int
) -> tuple[
    dict[int, int],  # channel_durations
    dict[int, list[int]],  # channel_intervals
    list[int],  # global message_counts per bucket
    dict[int, list[int]],  # per_channel_distributions
    dict[int, int],  # message_start_time
    dict[int, int],  # message_end_time
]:
    """Single-pass collection of all channel statistics and distributions.

    Computes intervals on-the-fly within each chunk (messages within chunks are
    already sorted by log_time in MCAP format). This avoids collecting all
    timestamps and sorting, providing O(n) performance instead of O(n log n).

    Returns:
        Tuple of (
            channel_durations,
            channel_intervals,
            message_counts,
            per_channel_distributions,
            message_start_time,
            message_end_time,
        )
    """
    # Initialize data structures
    channel_stats: dict[int, ChannelStatistics] = {}
    global_message_counts = [0] * bucket_count
    per_channel_distributions: dict[int, list[int]] = {}

    # Handle edge case
    if not info.chunk_information or bucket_duration_ns <= 0:
        return {}, {}, global_message_counts, {}, {}, {}

    # Single pass over all chunk information
    for msg_idx_list in info.chunk_information.values():
        for msg_idx in msg_idx_list:
            timestamps = msg_idx.timestamps
            if not timestamps:
                continue

            channel_id = msg_idx.channel_id

            # Initialize channel stats if first time seeing this channel
            if channel_id not in channel_stats:
                stats = ChannelStatistics(channel_id=channel_id)
                channel_stats[channel_id] = stats
                per_channel_distributions[channel_id] = [0] * bucket_count

            stats = channel_stats[channel_id]
            channel_dist = per_channel_distributions[channel_id]

            # Process first record separately
            first_timestamp = timestamps[0]
            stats.first_time = min(stats.first_time, first_timestamp)

            # Compute intervals within this chunk (timestamps are already sorted by log_time)
            prev_timestamp = first_timestamp
            for timestamp in timestamps:
                # Update last_time (will end up with the max)
                stats.last_time = max(stats.last_time, timestamp)

                # Compute interval from previous message in this chunk
                if timestamp > prev_timestamp:
                    stats.intervals.append(timestamp - prev_timestamp)
                prev_timestamp = timestamp

                stats.message_count += 1

                # Update distributions
                offset = timestamp - start_time
                bucket_idx = int(offset / bucket_duration_ns)
                if bucket_idx >= bucket_count:
                    bucket_idx = bucket_count - 1
                if bucket_idx >= 0:
                    global_message_counts[bucket_idx] += 1
                    channel_dist[bucket_idx] += 1

    # Calculate durations
    channel_durations = {
        channel_id: int(stats.last_time - stats.first_time)
        for channel_id, stats in channel_stats.items()
        if stats.first_time != sys.maxsize
    }

    # Collect intervals (already computed on-the-fly)
    channel_intervals: dict[int, list[int]] = {
        channel_id: stats.intervals
        for channel_id, stats in channel_stats.items()
        if stats.intervals
    }

    # Collect first and last times
    message_start_time = {
        channel_id: stats.first_time
        for channel_id, stats in channel_stats.items()
        if stats.first_time != sys.maxsize
    }
    message_end_time = {
        channel_id: stats.last_time
        for channel_id, stats in channel_stats.items()
        if stats.last_time != 0
    }

    return (
        channel_durations,
        channel_intervals,
        global_message_counts,
        per_channel_distributions,
        message_start_time,
        message_end_time,
    )


@dataclass(slots=True)
class IntervalStatsResult:
    hz_stats: dict[str, float]
    jitter_ns: float
    jitter_cv: float
    bps_stats: dict[str, float] | None = None


def _calculate_interval_stats(
    channel_intervals: dict[int, list[int]],
    channel_sizes: dict[int, int] | None,
    message_counts: dict[int, int],
) -> dict[int, IntervalStatsResult]:
    """Calculate statistical rate information from message intervals.

    Args:
        channel_intervals: Dict mapping channel_id -> list of intervals (ns)
        channel_sizes: Optional dict mapping channel_id -> total bytes
        message_counts: Dict mapping channel_id -> message count

    Returns:
        Dict mapping channel_id -> IntervalStatsResult with hz_stats,
        jitter_ns, jitter_cv, and optional bps_stats.
    """
    interval_stats: dict[int, IntervalStatsResult] = {}

    for channel_id, intervals in channel_intervals.items():
        if not intervals:
            continue

        # Convert intervals to Hz values (no sorting needed)
        hz_values = [1_000_000_000 / interval for interval in intervals]

        # Calculate Hz statistics using fast operations
        # Note: statistics.median() handles unsorted data efficiently
        hz_stats = {
            "minimum": min(hz_values),
            "maximum": max(hz_values),
            "average": sum(hz_values) / len(hz_values),
            "median": statistics.median(hz_values),
        }

        # Calculate jitter (stddev and coefficient of variation of intervals)
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((iv - mean_interval) ** 2 for iv in intervals) / len(intervals)
        jitter_ns = math.sqrt(variance)
        jitter_cv = jitter_ns / mean_interval if mean_interval > 0 else 0.0

        result = IntervalStatsResult(hz_stats=hz_stats, jitter_ns=jitter_ns, jitter_cv=jitter_cv)

        # Calculate bytes per second statistics if size data is available
        if channel_sizes and channel_id in channel_sizes:
            channel_size = channel_sizes[channel_id]
            message_count = message_counts.get(channel_id, 0)

            if message_count > 0:
                # bytes/sec = Hz * average bytes per message
                avg_bytes_per_msg = channel_size / message_count
                result.bps_stats = {
                    "minimum": hz_stats["minimum"] * avg_bytes_per_msg,
                    "maximum": hz_stats["maximum"] * avg_bytes_per_msg,
                    "average": hz_stats["average"] * avg_bytes_per_msg,
                    "median": hz_stats["median"] * avg_bytes_per_msg,
                }

        interval_stats[channel_id] = result

    return interval_stats


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
    statistics_rec = summary.statistics
    if statistics_rec is None:
        raise ValueError("MCAP statistics are missing")

    # Calculate global duration
    duration_ns = statistics_rec.message_end_time - statistics_rec.message_start_time

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
            stats.message_count += sum(len(ci.timestamps) for ci in cinfo)

    # Calculate chunk overlaps
    max_concurrent, overlap_size = _calculate_chunk_overlaps(summary.chunk_indexes)

    # Calculate optimal bucket count for distributions
    bucket_count = _calculate_optimal_bucket_count(duration_ns)
    bucket_duration_ns = duration_ns // bucket_count if duration_ns > 0 else 0

    # Single-pass collection of all channel statistics and distributions
    channel_durations: dict[int, int] = {}
    channel_intervals: dict[int, list[int]] = {}
    global_message_counts: list[int] = []
    per_channel_distributions: dict[int, list[int]] = {}
    message_start_time: dict[int, int] = {}
    message_end_time: dict[int, int] = {}
    interval_stats: dict[int, IntervalStatsResult] = {}

    if info.chunk_information:
        (
            channel_durations,
            channel_intervals,
            global_message_counts,
            per_channel_distributions,
            message_start_time,
            message_end_time,
        ) = _collect_channel_statistics(
            info,
            statistics_rec.message_start_time,
            bucket_count,
            bucket_duration_ns,
        )
        interval_stats = _calculate_interval_stats(
            channel_intervals, info.channel_sizes, statistics_rec.channel_message_counts
        )
    else:
        global_message_counts = [0] * bucket_count

    # Build message distribution from collected data
    max_count = max(global_message_counts) if global_message_counts else 0
    message_distribution: MessageDistribution = {
        "bucket_count": bucket_count,
        "bucket_duration_ns": bucket_duration_ns,
        "message_counts": global_message_counts,
        "max_count": max_count,
    }

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
            "message_count": statistics_rec.message_count,
            "chunk_count": statistics_rec.chunk_count,
            "message_index_count": sum(map(len, info.chunk_information.values()))
            if info.chunk_information
            else 0,
            "channel_count": statistics_rec.channel_count,
            "attachment_count": statistics_rec.attachment_count,
            "metadata_count": statistics_rec.metadata_count,
            "message_start_time": statistics_rec.message_start_time,
            "message_end_time": statistics_rec.message_end_time,
            "duration_ns": duration_ns,
        },
        "chunks": {
            "by_compression": {},
            "overlaps": {
                "max_concurrent": max_concurrent,
                "max_concurrent_bytes": overlap_size,
            },
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

    # Add channel information
    for channel in summary.channels.values():
        channel_id = channel.id
        count = statistics_rec.channel_message_counts.get(channel_id, 0)
        schema = summary.schemas.get(channel.schema_id)
        channel_size = info.channel_sizes.get(channel_id) if info.channel_sizes else None
        ch_duration_ns = channel_durations.get(channel_id)
        ch_distribution = per_channel_distributions.get(channel_id, [])
        ch_interval_stats = interval_stats.get(channel_id)
        ch_first_time = message_start_time.get(channel_id)
        ch_last_time = message_end_time.get(channel_id)

        output["channels"].append(
            _build_channel_dict(
                channel=channel,
                message_count=count,
                schema_name=schema.name if schema else None,
                channel_size=channel_size,
                estimated_sizes=info.estimated_channel_sizes,
                channel_duration_ns=ch_duration_ns,
                global_duration_ns=duration_ns,
                message_distribution=ch_distribution,
                interval_stats=ch_interval_stats,
                message_start_time=ch_first_time,
                message_end_time=ch_last_time,
            )
        )

    # Add schema information
    for schema_id, schema in summary.schemas.items():
        output["schemas"].append(_build_schema_dict(schema_id, schema))

    # Add metadata records
    output["metadata"] = [
        MetadataInfo(name=mi.name, metadata={}) for mi in summary.metadata_indexes
    ]

    # Add attachment information from indexes
    output["attachments"] = [
        AttachmentInfo(
            name=ai.name,
            media_type=ai.media_type,
            data_size=ai.data_size,
            log_time=ai.log_time,
            create_time=ai.create_time,
            offset=ai.offset,
            length=ai.length,
        )
        for ai in summary.attachment_indexes
    ]

    return output


def _calculate_stats(values: list[int] | list[float]) -> Stats:
    """Calculate min, max, average, and median from a list of values."""
    if not values:
        return {"minimum": 0, "maximum": 0, "average": 0.0, "median": 0.0}

    return {
        "minimum": min(values),
        "maximum": max(values),
        "average": statistics.mean(values),
        "median": statistics.median(values),
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
    estimated_sizes: bool,
    channel_duration_ns: int | None,
    global_duration_ns: int,
    message_distribution: list[int],
    interval_stats: IntervalStatsResult | None = None,
    message_start_time: int | None = None,
    message_end_time: int | None = None,
) -> ChannelInfo:
    """Build channel information dict for JSON output with calculated metrics."""
    # Calculate global Hz (average based on global duration)
    hz_global = (
        message_count / (global_duration_ns / 1_000_000_000) if global_duration_ns > 0 else 0
    )

    # Calculate Hz based on channel duration (first to last message)
    hz_channel = None
    if channel_duration_ns is not None and channel_duration_ns > 0:
        hz_channel = message_count / (channel_duration_ns / 1_000_000_000)

    # Build hz_stats (PartialStats)
    # Average is always available (from global duration)
    # Min/max/median only available from interval stats (rebuild mode)
    hz_stats: PartialStats = {
        "average": hz_global,
        "minimum": None,
        "maximum": None,
        "median": None,
    }
    if interval_stats:
        hz_stats["minimum"] = interval_stats.hz_stats["minimum"]
        hz_stats["maximum"] = interval_stats.hz_stats["maximum"]
        hz_stats["median"] = interval_stats.hz_stats["median"]

    # Calculate bytes per message
    b_per_msg = None
    if channel_size is not None and message_count > 0:
        b_per_msg = channel_size / message_count

    # Build bytes_per_second_stats (PartialStats | None)
    # Only available if we have channel size data
    bps_stats: PartialStats | None = None
    if channel_size is not None:
        bps_global = (
            channel_size / (global_duration_ns / 1_000_000_000) if global_duration_ns > 0 else 0
        )
        bps_stats = {
            "average": bps_global,
            "minimum": None,
            "maximum": None,
            "median": None,
        }
        if interval_stats and interval_stats.bps_stats:
            bps_stats["minimum"] = interval_stats.bps_stats["minimum"]
            bps_stats["maximum"] = interval_stats.bps_stats["maximum"]
            bps_stats["median"] = interval_stats.bps_stats["median"]

    # Extract jitter values from interval_stats (only available in rebuild mode)
    jitter_ns: float | None = None
    jitter_cv: float | None = None
    if interval_stats:
        jitter_ns = interval_stats.jitter_ns
        jitter_cv = interval_stats.jitter_cv

    return {
        "id": channel.id,
        "topic": channel.topic,
        "schema_id": channel.schema_id,
        "schema_name": schema_name,
        "message_count": message_count,
        "size_bytes": channel_size,
        "estimated_sizes": estimated_sizes,
        "duration_ns": channel_duration_ns,
        "hz_stats": hz_stats,
        "hz_channel": hz_channel,
        "bytes_per_second_stats": bps_stats,
        "bytes_per_message": b_per_msg,
        "message_distribution": message_distribution,
        "message_start_time": message_start_time,
        "message_end_time": message_end_time,
        "jitter_ns": jitter_ns,
        "jitter_cv": jitter_cv,
    }


def _build_schema_dict(schema_id: int, schema: Schema) -> SchemaInfo:
    """Build schema information dict for JSON output."""
    return {
        "id": schema_id,
        "name": schema.name,
        "encoding": schema.encoding,
        "data": schema.data.decode("utf-8", errors="replace"),
    }
