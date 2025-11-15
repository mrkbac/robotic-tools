"""Type definitions for MCAP info structures."""

from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict

from cyclopts import Group, Parameter


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


class ChunkOverlaps(TypedDict):
    """Chunk overlap information."""

    max_concurrent: int
    max_concurrent_bytes: int


class ChunksInfo(TypedDict):
    """Chunk-related information."""

    by_compression: dict[str, CompressionStats]
    overlaps: ChunkOverlaps


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
    hz_median: float | None
    bytes_per_second: float | None
    bytes_per_second_median: float | None
    bytes_per_message: float | None
    messages_per_second_median: float | None
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


class CompressionType(str, Enum):
    """Compression algorithm types."""

    ZSTD = "zstd"
    LZ4 = "lz4"
    NONE = "none"


# Common CLI parameter type aliases (with defaults included)

# MCAP processing constants
MIN_CHUNK_SIZE = 1024  # 1 KiB minimum chunk size
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB default chunk size
DEFAULT_COMPRESSION = CompressionType.ZSTD  # Default compression algorithm

# Parameter groups
OUTPUT_OPTIONS_GROUP = Group("Output Options")

ChunkSizeOption = Annotated[
    int,
    Parameter(
        name=["--chunk-size"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]

CompressionOption = Annotated[
    CompressionType,
    Parameter(
        name=["--compression"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]

OutputPathOption = Annotated[
    Path,
    Parameter(
        name=["-o", "--output"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]

ForceOverwriteOption = Annotated[
    bool,
    Parameter(
        name=["-f", "--force"],
        group=OUTPUT_OPTIONS_GROUP,
    ),
]
