# ruff: noqa: I001, E501
# AUTO-GENERATED from pymcap-cli/schemas/mcap_info.json - DO NOT EDIT


from typing import TypedDict, Union
from typing_extensions import Required


class ChannelInfo(TypedDict, total=False):
    """
    ChannelInfo.

    Information about a channel.
    """

    id: Required[int]
    """ Required property """

    topic: Required[str]
    """ Required property """

    schema_id: Required[int]
    """ Required property """

    schema_name: str | None
    message_count: Required[int]
    """
    minimum: 0

    Required property
    """

    size_bytes: int | None
    """ minimum: 0 """

    duration_ns: int | None
    """ minimum: 0 """

    hz_stats: Required["PartialStats"]
    """
    PartialStats.

    Statistics where only average is always available; min/max/median are only available in rebuild mode.

    Required property
    """

    hz_channel: int | float | None
    """ minimum: 0 """

    bytes_per_second_stats: "_ChannelInfobytespersecondstats"
    """ Aggregation type: anyOf """

    bytes_per_message: int | float | None
    """ minimum: 0 """

    message_distribution: Required[list["_ChannelInfomessagedistributionitem"]]
    """ Required property """

    message_start_time: int | None
    """ minimum: 0 """

    message_end_time: int | None
    """ minimum: 0 """


class ChunkOverlaps(TypedDict, total=False):
    """
    ChunkOverlaps.

    Chunk overlap information.
    """

    max_concurrent: Required[int]
    """
    minimum: 0

    Required property
    """

    max_concurrent_bytes: Required[int]
    """
    minimum: 0

    Required property
    """


class ChunksInfo(TypedDict, total=False):
    """
    ChunksInfo.

    Chunk-related information.
    """

    by_compression: Required[dict[str, "CompressionStats"]]
    """ Required property """

    overlaps: Required["ChunkOverlaps"]
    """
    ChunkOverlaps.

    Chunk overlap information.

    Required property
    """


class CompressionStats(TypedDict, total=False):
    """
    CompressionStats.

    Statistics for a specific compression type.
    """

    count: Required[int]
    """
    minimum: 0

    Required property
    """

    compressed_size: Required[int]
    """
    minimum: 0

    Required property
    """

    uncompressed_size: Required[int]
    """
    minimum: 0

    Required property
    """

    compression_ratio: Required[int | float]
    """
    minimum: 0

    Required property
    """

    message_count: Required[int]
    """
    minimum: 0

    Required property
    """

    size_stats: Required["Stats"]
    """
    Stats.

    Statistics with minimum, average, and maximum values.

    Required property
    """

    duration_stats: Required["Stats"]
    """
    Stats.

    Statistics with minimum, average, and maximum values.

    Required property
    """


class FileInfo(TypedDict, total=False):
    """
    FileInfo.

    File metadata.
    """

    path: Required[str]
    """ Required property """

    size_bytes: Required[int]
    """
    minimum: 0

    Required property
    """


class HeaderInfo(TypedDict, total=False):
    """
    HeaderInfo.

    MCAP header information.
    """

    library: Required[str]
    """ Required property """

    profile: Required[str]
    """ Required property """


class McapInfoOutput(TypedDict, total=False):
    """
    McapInfoOutput.

    Complete MCAP info output structure.
    """

    file: Required["FileInfo"]
    """
    FileInfo.

    File metadata.

    Required property
    """

    header: Required["HeaderInfo"]
    """
    HeaderInfo.

    MCAP header information.

    Required property
    """

    statistics: Required["StatisticsInfo"]
    """
    StatisticsInfo.

    MCAP statistics.

    Required property
    """

    chunks: Required["ChunksInfo"]
    """
    ChunksInfo.

    Chunk-related information.

    Required property
    """

    channels: Required[list["ChannelInfo"]]
    """ Required property """

    schemas: Required[list["SchemaInfo"]]
    """ Required property """

    message_distribution: Required["MessageDistribution"]
    """
    MessageDistribution.

    Message distribution across time buckets.

    Required property
    """


class MessageDistribution(TypedDict, total=False):
    """
    MessageDistribution.

    Message distribution across time buckets.
    """

    bucket_count: Required[int]
    """
    minimum: 0

    Required property
    """

    bucket_duration_ns: Required[int]
    """
    minimum: 0

    Required property
    """

    message_counts: Required[list["_MessageDistributionmessagecountsitem"]]
    """ Required property """

    max_count: Required[int]
    """
    minimum: 0

    Required property
    """


class PartialStats(TypedDict, total=False):
    """
    PartialStats.

    Statistics where only average is always available; min/max/median are only available in rebuild
    mode.
    """

    average: Required[int | float]
    """
    minimum: 0

    Required property
    """

    minimum: int | float | None
    """ minimum: 0 """

    maximum: int | float | None
    """ minimum: 0 """

    median: int | float | None
    """ minimum: 0 """


class SchemaInfo(TypedDict, total=False):
    """
    SchemaInfo.

    Information about a schema.
    """

    id: Required[int]
    """ Required property """

    name: Required[str]
    """ Required property """


class StatisticsInfo(TypedDict, total=False):
    """
    StatisticsInfo.

    MCAP statistics.
    """

    message_count: Required[int]
    """
    minimum: 0

    Required property
    """

    chunk_count: Required[int]
    """
    minimum: 0

    Required property
    """

    message_index_count: int | None
    """ minimum: 0 """

    channel_count: Required[int]
    """
    minimum: 0

    Required property
    """

    attachment_count: Required[int]
    """
    minimum: 0

    Required property
    """

    metadata_count: Required[int]
    """
    minimum: 0

    Required property
    """

    message_start_time: Required[int]
    """
    minimum: 0

    Required property
    """

    message_end_time: Required[int]
    """
    minimum: 0

    Required property
    """

    duration_ns: Required[int]
    """
    minimum: 0

    Required property
    """


class Stats(TypedDict, total=False):
    """
    Stats.

    Statistics with minimum, average, and maximum values.
    """

    minimum: Required[int | float]
    """
    minimum: 0

    Required property
    """

    maximum: Required[int | float]
    """
    minimum: 0

    Required property
    """

    average: Required[int | float]
    """
    minimum: 0

    Required property
    """

    median: Required[int | float]
    """
    minimum: 0

    Required property
    """


_ChannelInfobytespersecondstats = Union["PartialStats", None]
""" Aggregation type: anyOf """


_ChannelInfomessagedistributionitem = int
""" minimum: 0 """


_MessageDistributionmessagecountsitem = int
""" minimum: 0 """
