# ruff: noqa: I001, E501
# AUTO-GENERATED from pymcap-cli/schemas/mcap_info.json - DO NOT EDIT


from typing import TypedDict, Union
from typing_extensions import Required


class AttachmentInfo(TypedDict, total=False):
    r"""
    AttachmentInfo.

    MCAP attachment record.
    """

    name: Required[str]
    r""" Required property """

    media_type: Required[str]
    r""" Required property """

    data_size: Required[int]
    r"""
    minimum: 0

    Required property
    """

    log_time: Required[int]
    r"""
    minimum: 0

    Required property
    """

    create_time: Required[int]
    r"""
    minimum: 0

    Required property
    """

    offset: Required[int]
    r"""
    minimum: 0

    Required property
    """

    length: Required[int]
    r"""
    minimum: 0

    Required property
    """


class ChannelInfo(TypedDict, total=False):
    r"""
    ChannelInfo.

    Information about a channel.
    """

    id: Required[int]
    r""" Required property """

    topic: Required[str]
    r""" Required property """

    schema_id: Required[int]
    r""" Required property """

    message_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    size_bytes: Required[int | None]
    r"""
    minimum: 0

    Required property
    """

    duration_ns: Required[int | None]
    r"""
    minimum: 0

    Required property
    """

    hz_stats: "_ChannelInfohzstats"
    r""" Aggregation type: anyOf """

    message_distribution: list["_ChannelInfomessagedistributionitem"]
    message_start_time: int | None
    r""" minimum: 0 """

    message_end_time: int | None
    r""" minimum: 0 """

    estimated_sizes: Required[bool]
    r"""
    Whether size_bytes is estimated from MessageIndex offsets (true) or measured from actual data (false).

    Required property
    """

    jitter_ns: int | float | None
    r""" Standard deviation of inter-message intervals in nanoseconds. """


class ChunkOverlaps(TypedDict, total=False):
    r"""
    ChunkOverlaps.

    Chunk overlap information.
    """

    max_concurrent: Required[int]
    r"""
    minimum: 0

    Required property
    """

    max_concurrent_bytes: Required[int]
    r"""
    minimum: 0

    Required property
    """


class ChunksInfo(TypedDict, total=False):
    r"""
    ChunksInfo.

    Chunk-related information.
    """

    by_compression: Required[dict[str, "CompressionStats"]]
    r""" Required property """

    overlaps: Required["ChunkOverlaps"]
    r"""
    ChunkOverlaps.

    Chunk overlap information.

    Required property
    """


class CompressionStats(TypedDict, total=False):
    r"""
    CompressionStats.

    Statistics for a specific compression type.
    """

    count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    compressed_size: Required[int]
    r"""
    minimum: 0

    Required property
    """

    uncompressed_size: Required[int]
    r"""
    minimum: 0

    Required property
    """

    compression_ratio: Required[int | float]
    r"""
    minimum: 0

    Required property
    """

    message_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    size_stats: Required["Stats"]
    r"""
    Stats.

    Statistics with minimum, average, and maximum values.

    Required property
    """

    duration_stats: Required["Stats"]
    r"""
    Stats.

    Statistics with minimum, average, and maximum values.

    Required property
    """


class FileInfo(TypedDict, total=False):
    r"""
    FileInfo.

    File metadata.
    """

    path: Required[str]
    r""" Required property """

    size_bytes: Required[int]
    r"""
    minimum: 0

    Required property
    """


class HeaderInfo(TypedDict, total=False):
    r"""
    HeaderInfo.

    MCAP header information.
    """

    library: Required[str]
    r""" Required property """

    profile: Required[str]
    r""" Required property """


class IntervalStats(TypedDict, total=False):
    r"""
    IntervalStats.

    Min/max/median statistics from per-interval measurements (rebuild-only).
    """

    minimum: Required[int | float | None]
    r""" Required property """

    maximum: Required[int | float | None]
    r""" Required property """

    median: Required[int | float | None]
    r""" Required property """


class McapInfoOutput(TypedDict, total=False):
    r"""
    McapInfoOutput.

    Complete MCAP info output structure.
    """

    file: Required["FileInfo"]
    r"""
    FileInfo.

    File metadata.

    Required property
    """

    header: Required["HeaderInfo"]
    r"""
    HeaderInfo.

    MCAP header information.

    Required property
    """

    statistics: Required["StatisticsInfo"]
    r"""
    StatisticsInfo.

    MCAP statistics.

    Required property
    """

    chunks: Required["ChunksInfo"]
    r"""
    ChunksInfo.

    Chunk-related information.

    Required property
    """

    channels: Required[list["ChannelInfo"]]
    r""" Required property """

    schemas: Required[list["SchemaInfo"]]
    r""" Required property """

    message_distribution: Required["MessageDistribution"]
    r"""
    MessageDistribution.

    Message distribution across time buckets.

    Required property
    """

    metadata: list["MetadataInfo"]
    attachments: list["AttachmentInfo"]
    thumbnail: str
    r""" Base64-encoded micro-thumbnail JPEG (48x36, q=0.5). """


class MessageDistribution(TypedDict, total=False):
    r"""
    MessageDistribution.

    Message distribution across time buckets.
    """

    bucket_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    bucket_duration_ns: Required[int]
    r"""
    minimum: 0

    Required property
    """

    message_counts: Required[list["_MessageDistributionmessagecountsitem"]]
    r""" Required property """

    max_count: Required[int]
    r"""
    minimum: 0

    Required property
    """


class MetadataInfo(TypedDict, total=False):
    r"""
    MetadataInfo.

    MCAP metadata record.
    """

    name: Required[str]
    r""" Required property """

    metadata: Required[dict[str, str]]
    r""" Required property """


class SchemaInfo(TypedDict, total=False):
    r"""
    SchemaInfo.

    Information about a schema.
    """

    id: Required[int]
    r""" Required property """

    name: Required[str]
    r""" Required property """

    encoding: Required[str]
    r""" Required property """

    data: Required[str]
    r""" Required property """


class StatisticsInfo(TypedDict, total=False):
    r"""
    StatisticsInfo.

    MCAP statistics.
    """

    message_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    chunk_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    message_index_count: Required[int | None]
    r"""
    minimum: 0

    Required property
    """

    channel_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    attachment_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    metadata_count: Required[int]
    r"""
    minimum: 0

    Required property
    """

    message_start_time: Required[int]
    r"""
    minimum: 0

    Required property
    """

    message_end_time: Required[int]
    r"""
    minimum: 0

    Required property
    """

    duration_ns: Required[int]
    r"""
    minimum: 0

    Required property
    """


class Stats(TypedDict, total=False):
    r"""
    Stats.

    Statistics with minimum, average, and maximum values.
    """

    minimum: Required[int | float]
    r"""
    minimum: 0

    Required property
    """

    maximum: Required[int | float]
    r"""
    minimum: 0

    Required property
    """

    average: Required[int | float]
    r"""
    minimum: 0

    Required property
    """

    median: Required[int | float]
    r"""
    minimum: 0

    Required property
    """


_ChannelInfohzstats = Union["IntervalStats", None]
r""" Aggregation type: anyOf """


_ChannelInfomessagedistributionitem = int
r""" minimum: 0 """


_MessageDistributionmessagecountsitem = int
r""" minimum: 0 """
