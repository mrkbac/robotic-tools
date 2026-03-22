"""Tests for pymcap_cli.types.info_data — pure data transformation logic."""

from __future__ import annotations

import pytest
from pymcap_cli.types.info_data import (
    ChunkStats,
    IntervalStatsResult,
    _build_channel_dict,
    _build_chunk_compression_stats,
    _build_schema_dict,
    _calculate_chunk_overlaps,
    _calculate_interval_stats,
    _calculate_optimal_bucket_count,
    _calculate_stats,
    info_to_dict,
)
from small_mcap.rebuild import RebuildInfo
from small_mcap.records import (
    Channel,
    ChunkIndex,
    Header,
    MessageIndex,
    Schema,
    Statistics,
    Summary,
)

# ---------------------------------------------------------------------------
# _calculate_stats
# ---------------------------------------------------------------------------


class TestCalculateStats:
    def test_empty(self):
        result = _calculate_stats([])
        assert result == {"minimum": 0, "maximum": 0, "average": 0.0, "median": 0.0}

    def test_single_value(self):
        result = _calculate_stats([42])
        assert result["minimum"] == 42
        assert result["maximum"] == 42
        assert result["average"] == 42.0
        assert result["median"] == 42.0

    def test_multiple_values(self):
        result = _calculate_stats([10, 20, 30, 40])
        assert result["minimum"] == 10
        assert result["maximum"] == 40
        assert result["average"] == 25.0
        assert result["median"] == 25.0

    def test_odd_count_median(self):
        result = _calculate_stats([1, 3, 5])
        assert result["median"] == 3.0

    def test_floats(self):
        result = _calculate_stats([1.5, 2.5, 3.5])
        assert result["minimum"] == 1.5
        assert result["maximum"] == 3.5


# ---------------------------------------------------------------------------
# _calculate_chunk_overlaps
# ---------------------------------------------------------------------------


def _make_chunk_index(start: int, end: int, size: int = 1000) -> ChunkIndex:
    return ChunkIndex(
        message_start_time=start,
        message_end_time=end,
        chunk_start_offset=0,
        chunk_length=0,
        message_index_offsets={},
        message_index_length=0,
        compression="zstd",
        compressed_size=500,
        uncompressed_size=size,
    )


class TestCalculateChunkOverlaps:
    def test_empty(self):
        assert _calculate_chunk_overlaps([]) == (0, 0)

    def test_single_chunk(self):
        assert _calculate_chunk_overlaps([_make_chunk_index(0, 100)]) == (0, 0)

    def test_non_overlapping(self):
        chunks = [
            _make_chunk_index(0, 100),
            _make_chunk_index(200, 300),
            _make_chunk_index(400, 500),
        ]
        max_concurrent, max_bytes = _calculate_chunk_overlaps(chunks)
        assert max_concurrent == 1
        assert max_bytes == 1000  # single chunk's uncompressed_size

    def test_overlapping(self):
        chunks = [
            _make_chunk_index(0, 200, size=100),
            _make_chunk_index(50, 250, size=200),
            _make_chunk_index(100, 300, size=300),
        ]
        max_concurrent, max_bytes = _calculate_chunk_overlaps(chunks)
        assert max_concurrent == 3
        assert max_bytes == 600  # 100 + 200 + 300

    def test_partial_overlap(self):
        chunks = [
            _make_chunk_index(0, 100, size=100),
            _make_chunk_index(50, 150, size=200),
            _make_chunk_index(200, 300, size=300),
        ]
        max_concurrent, max_bytes = _calculate_chunk_overlaps(chunks)
        assert max_concurrent == 2
        assert max_bytes == 300  # 100 + 200


# ---------------------------------------------------------------------------
# _calculate_interval_stats
# ---------------------------------------------------------------------------


class TestCalculateIntervalStats:
    def test_empty(self):
        assert _calculate_interval_stats({}) == {}

    def test_empty_intervals_for_channel(self):
        result = _calculate_interval_stats({1: []})
        assert result == {}

    def test_uniform_intervals(self):
        # 10 Hz = 100ms intervals
        interval_ns = 100_000_000
        result = _calculate_interval_stats({1: [interval_ns] * 10})
        stats = result[1]
        assert stats.hz_stats["minimum"] == pytest.approx(10.0)
        assert stats.hz_stats["maximum"] == pytest.approx(10.0)
        assert stats.hz_stats["median"] == pytest.approx(10.0)
        assert stats.jitter_ns == pytest.approx(0.0)

    def test_variable_intervals(self):
        # Mix of 10 Hz and 20 Hz
        intervals = [100_000_000, 50_000_000]  # 10Hz, 20Hz
        result = _calculate_interval_stats({1: intervals})
        stats = result[1]
        assert stats.hz_stats["minimum"] == pytest.approx(10.0)
        assert stats.hz_stats["maximum"] == pytest.approx(20.0)
        # jitter = stddev of intervals: mean=75M, each deviates by 25M → stddev=25M
        assert stats.jitter_ns == pytest.approx(25_000_000.0)


# ---------------------------------------------------------------------------
# _calculate_optimal_bucket_count
# ---------------------------------------------------------------------------


class TestCalculateOptimalBucketCount:
    def test_10s_picks_20_buckets_for_500ms(self):
        # 10s / 20 = 500ms, which is a round duration in the list
        result = _calculate_optimal_bucket_count(10_000_000_000)
        assert result == 20
        assert 10_000_000_000 / result == 500_000_000  # 500ms

    def test_1s_picks_20_buckets_for_50ms(self):
        # 1s / 20 = 50ms, which is a round duration in the list
        result = _calculate_optimal_bucket_count(1_000_000_000)
        assert result == 20
        assert 1_000_000_000 / result == 50_000_000  # 50ms

    def test_1hr_picks_30_buckets_for_2min(self):
        # 1hr / 30 = 120s = 2min, which is a round duration in the list
        result = _calculate_optimal_bucket_count(3_600_000_000_000)
        assert result == 30
        assert 3_600_000_000_000 / result == 120_000_000_000  # 2min


# ---------------------------------------------------------------------------
# _build_chunk_compression_stats
# ---------------------------------------------------------------------------


class TestBuildChunkCompressionStats:
    def test_basic(self):
        stats = ChunkStats(
            count=2,
            compressed_size=500,
            uncompressed_size=1000,
            uncompressed_sizes=[400, 600],
            message_count=100,
            durations_ns=[1_000_000, 2_000_000],
        )
        result = _build_chunk_compression_stats(stats)
        assert result["count"] == 2
        assert result["compressed_size"] == 500
        assert result["uncompressed_size"] == 1000
        assert result["compression_ratio"] == pytest.approx(0.5)
        assert result["message_count"] == 100
        assert result["size_stats"]["minimum"] == 400
        assert result["size_stats"]["maximum"] == 600
        assert result["duration_stats"]["minimum"] == 1_000_000
        assert result["duration_stats"]["maximum"] == 2_000_000

    def test_zero_uncompressed(self):
        stats = ChunkStats(
            count=1,
            compressed_size=0,
            uncompressed_size=0,
            uncompressed_sizes=[0],
            message_count=0,
            durations_ns=[0],
        )
        result = _build_chunk_compression_stats(stats)
        assert result["compression_ratio"] == 0


# ---------------------------------------------------------------------------
# _build_channel_dict
# ---------------------------------------------------------------------------


class TestBuildChannelDict:
    def test_basic(self):
        channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="cdr", metadata={})
        result = _build_channel_dict(
            channel=channel,
            message_count=100,
            channel_size=5000,
            estimated_sizes=False,
            channel_duration_ns=1_000_000_000,
            message_distribution=[10, 20, 30],
        )
        assert result["id"] == 1
        assert result["topic"] == "/test"
        assert result["message_count"] == 100
        assert result["size_bytes"] == 5000
        assert result["duration_ns"] == 1_000_000_000

    def test_with_interval_stats(self):
        channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="cdr", metadata={})
        hz_stats = {"minimum": 10.0, "maximum": 30.0, "median": 20.0}
        interval = IntervalStatsResult(hz_stats=hz_stats, jitter_ns=500.0)
        result = _build_channel_dict(
            channel=channel,
            message_count=50,
            channel_size=None,
            estimated_sizes=True,
            channel_duration_ns=None,
            message_distribution=[],
            interval_stats=interval,
        )
        assert result["hz_stats"] == hz_stats
        assert result["jitter_ns"] == 500.0

    def test_without_interval_stats(self):
        channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="cdr", metadata={})
        result = _build_channel_dict(
            channel=channel,
            message_count=50,
            channel_size=None,
            estimated_sizes=True,
            channel_duration_ns=None,
            message_distribution=[],
        )
        assert "hz_stats" not in result
        assert "jitter_ns" not in result


# ---------------------------------------------------------------------------
# _build_schema_dict
# ---------------------------------------------------------------------------


class TestBuildSchemaDict:
    def test_basic(self):
        schema = Schema(id=1, name="sensor_msgs/msg/Image", encoding="ros2msg", data=b"uint8 data")
        result = _build_schema_dict(1, schema)
        assert result["id"] == 1
        assert result["name"] == "sensor_msgs/msg/Image"
        assert result["encoding"] == "ros2msg"
        assert result["data"] == "uint8 data"

    def test_non_utf8_data_uses_replacement_char(self):
        schema = Schema(id=2, name="test", encoding="protobuf", data=b"\xff\xfe")
        result = _build_schema_dict(2, schema)
        assert result["data"] == "\ufffd\ufffd"


# ---------------------------------------------------------------------------
# info_to_dict (integration)
# ---------------------------------------------------------------------------


def _make_rebuild_info(
    *,
    num_channels: int = 1,
    num_messages: int = 100,
    duration_ns: int = 10_000_000_000,
    num_chunks: int = 2,
) -> RebuildInfo:
    """Create a synthetic RebuildInfo for testing."""
    header = Header(profile="ros2", library="test-lib")

    schemas = {1: Schema(id=1, name="std_msgs/msg/String", encoding="ros2msg", data=b"string data")}
    channels = {}
    channel_message_counts = {}
    for i in range(num_channels):
        ch_id = i + 1
        channels[ch_id] = Channel(
            id=ch_id, schema_id=1, topic=f"/topic_{i}", message_encoding="cdr", metadata={}
        )
        channel_message_counts[ch_id] = num_messages // num_channels

    start_time = 1_000_000_000
    end_time = start_time + duration_ns

    statistics = Statistics(
        message_count=num_messages,
        schema_count=1,
        channel_count=num_channels,
        attachment_count=0,
        metadata_count=0,
        chunk_count=num_chunks,
        message_start_time=start_time,
        message_end_time=end_time,
        channel_message_counts=channel_message_counts,
    )

    # Create chunk indexes
    chunk_duration = duration_ns // num_chunks
    chunk_indexes = [
        ChunkIndex(
            message_start_time=start_time + i * chunk_duration,
            message_end_time=start_time + (i + 1) * chunk_duration,
            chunk_start_offset=i * 10000,
            chunk_length=5000,
            message_index_offsets={1: 0},
            message_index_length=100,
            compression="zstd",
            compressed_size=2000,
            uncompressed_size=5000,
        )
        for i in range(num_chunks)
    ]

    summary = Summary(
        statistics=statistics,
        schemas=schemas,
        channels=channels,
        chunk_indexes=chunk_indexes,
    )

    # Create chunk information with message indexes
    msgs_per_chunk = num_messages // num_chunks
    chunk_information: dict[int, list[MessageIndex]] = {}
    for i in range(num_chunks):
        offset = i * 10000
        timestamps = [
            start_time + i * chunk_duration + j * (chunk_duration // msgs_per_chunk)
            for j in range(msgs_per_chunk)
        ]
        chunk_information[offset] = [
            MessageIndex(channel_id=1, timestamps=timestamps, offsets=list(range(len(timestamps))))
        ]

    return RebuildInfo(
        header=header,
        summary=summary,
        channel_sizes={1: 50000},
        estimated_channel_sizes=False,
        chunk_information=chunk_information,
    )


class TestInfoToDict:
    def test_basic_structure(self):
        info = _make_rebuild_info()
        result = info_to_dict(info, "/test.mcap", 100000)

        assert result["file"]["path"] == "/test.mcap"
        assert result["file"]["size_bytes"] == 100000
        assert result["header"]["library"] == "test-lib"
        assert result["header"]["profile"] == "ros2"
        assert result["statistics"]["message_count"] == 100
        assert result["statistics"]["chunk_count"] == 2
        assert result["statistics"]["channel_count"] == 1

    def test_channels_populated(self):
        info = _make_rebuild_info(num_channels=2, num_messages=200)
        result = info_to_dict(info, "/test.mcap", 100000)
        assert len(result["channels"]) == 2

    def test_schemas_populated(self):
        info = _make_rebuild_info()
        result = info_to_dict(info, "/test.mcap", 100000)
        assert len(result["schemas"]) == 1
        assert result["schemas"][0]["name"] == "std_msgs/msg/String"

    def test_chunks_by_compression(self):
        info = _make_rebuild_info(num_chunks=3)
        result = info_to_dict(info, "/test.mcap", 100000)
        assert "zstd" in result["chunks"]["by_compression"]
        assert result["chunks"]["by_compression"]["zstd"]["count"] == 3

    def test_message_distribution(self):
        info = _make_rebuild_info()
        result = info_to_dict(info, "/test.mcap", 100000)
        dist = result["message_distribution"]
        assert dist["bucket_count"] >= 20
        assert dist["bucket_count"] <= 80
        assert sum(dist["message_counts"]) == 100

    def test_missing_header_raises(self):
        info = _make_rebuild_info()
        info.header = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="header"):
            info_to_dict(info, "/test.mcap", 0)

    def test_missing_statistics_raises(self):
        info = _make_rebuild_info()
        info.summary.statistics = None
        with pytest.raises(ValueError, match="statistics"):
            info_to_dict(info, "/test.mcap", 0)

    def test_no_chunk_information(self):
        info = _make_rebuild_info()
        info.chunk_information = None
        result = info_to_dict(info, "/test.mcap", 100000)
        assert result["statistics"]["message_count"] == 100
        # Distribution should still exist but be all zeros
        assert all(c == 0 for c in result["message_distribution"]["message_counts"])
