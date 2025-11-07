"""Type definitions for MCAP info structures."""

from __future__ import annotations

from typing import TypedDict


class FileInfo(TypedDict):
    """File metadata."""

    path: str
    size_bytes: int


class HeaderInfo(TypedDict):
    """MCAP header information."""

    library: str
    profile: str


class StatisticsInfo(TypedDict):
    """MCAP statistics."""

    message_count: int
    chunk_count: int
    channel_count: int
    attachment_count: int
    metadata_count: int
    message_start_time: int
    message_end_time: int
    duration_ns: int


class Stats(TypedDict):
    """Statistics with minimum, average, and maximum values."""

    minimum: float
    maximum: float
    average: float
    median: float


class CompressionStats(TypedDict):
    """Statistics for a specific compression type."""

    count: int
    compressed_size: int
    uncompressed_size: int
    compression_ratio: float
    message_count: int
    size_stats: Stats
    duration_stats: Stats


class ChunkIndexInfo(TypedDict):
    """Information about a chunk index."""

    compression: str
    compressed_size: int
    uncompressed_size: int
    message_start_time: int
    message_end_time: int
    chunk_start_offset: int


class ChunkOverlaps(TypedDict):
    """Chunk overlap information."""

    max_concurrent: int
    max_concurrent_bytes: int


class ChunksInfo(TypedDict):
    """Chunk-related information."""

    by_compression: dict[str, CompressionStats]
    overlaps: ChunkOverlaps
    indexes: list[ChunkIndexInfo]


class ChannelInfo(TypedDict):
    """Information about a channel."""

    id: int
    topic: str
    schema_id: int
    schema_name: str | None
    message_count: int
    size_bytes: int | None
    duration_ns: int | None
    hz: float
    hz_channel: float | None
    bytes_per_second: float | None
    bytes_per_message: float | None
    message_distribution: list[int]


class SchemaInfo(TypedDict):
    """Information about a schema."""

    id: int
    name: str


class MessageDistribution(TypedDict):
    """Message distribution across time buckets."""

    bucket_count: int
    bucket_duration_ns: int
    message_counts: list[int]
    max_count: int


class McapInfoOutput(TypedDict):
    """Complete MCAP info output structure."""

    file: FileInfo
    header: HeaderInfo
    statistics: StatisticsInfo
    chunks: ChunksInfo
    channels: list[ChannelInfo]
    schemas: list[SchemaInfo]
    message_distribution: MessageDistribution
