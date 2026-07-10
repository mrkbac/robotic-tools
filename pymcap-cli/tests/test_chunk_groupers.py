"""Unit tests for the OutputProcessor chunk-grouper implementations."""

from __future__ import annotations

import re

from pymcap_cli.core.processors.chunk_groupers import (
    PatternGrouper,
    PerChannelGrouper,
    SchemaCompressionGrouper,
)
from small_mcap import Channel, CompressionType, Schema


def _channel(channel_id: int, topic: str, schema_id: int = 1) -> Channel:
    return Channel(
        id=channel_id,
        schema_id=schema_id,
        topic=topic,
        message_encoding="raw",
        metadata={},
    )


def _schema(schema_id: int, name: str) -> Schema:
    return Schema(id=schema_id, name=name, encoding="raw", data=b"")


# ---------------------------------------------------------------------------
# PerChannelGrouper
# ---------------------------------------------------------------------------


class TestPerChannelGrouper:
    def test_uses_channel_id_as_key(self):
        grouper = PerChannelGrouper()
        ch_a = _channel(1, "/a")
        ch_b = _channel(2, "/b")

        assert grouper.chunk_group_key(0, ch_a, None) == 1
        assert grouper.chunk_group_key(0, ch_b, None) == 2

    def test_ignores_segment_key_and_schema(self):
        grouper = PerChannelGrouper()
        ch = _channel(7, "/topic")
        assert grouper.chunk_group_key(0, ch, None) == 7
        assert grouper.chunk_group_key("seg-x", ch, _schema(1, "Anything")) == 7


# ---------------------------------------------------------------------------
# PatternGrouper
# ---------------------------------------------------------------------------


class TestPatternGrouper:
    def test_first_topic_pattern_wins(self):
        grouper = PatternGrouper(
            topic_patterns=[re.compile("/camera"), re.compile("/lidar")],
        )
        cam = _channel(1, "/camera/front")
        lidar = _channel(2, "/lidar/top")

        assert grouper.chunk_group_key(0, cam, None) == 0
        assert grouper.chunk_group_key(0, lidar, None) == 1

    def test_no_match_returns_minus_one(self):
        grouper = PatternGrouper(topic_patterns=[re.compile("/camera")])
        ch = _channel(1, "/imu")
        assert grouper.chunk_group_key(0, ch, None) == -1

    def test_schema_pattern_offset_by_topic_count(self):
        grouper = PatternGrouper(
            topic_patterns=[re.compile("/camera")],
            schema_patterns=[re.compile("Image"), re.compile("PointCloud")],
        )
        ch = _channel(1, "/cam_a", schema_id=1)

        # Topic doesn't match — schema does.
        key = grouper.chunk_group_key(0, ch, _schema(1, "sensor_msgs/Image"))
        assert key == 1  # len(topic_patterns) + 0

        key = grouper.chunk_group_key(0, ch, _schema(1, "sensor_msgs/PointCloud2"))
        assert key == 2  # len(topic_patterns) + 1

    def test_topic_match_takes_precedence_over_schema_match(self):
        grouper = PatternGrouper(
            topic_patterns=[re.compile("/cam_a")],
            schema_patterns=[re.compile("Image")],
        )
        # /cam_a matches the topic pattern (index 0); its schema also matches
        # the schema pattern. Topic wins.
        ch = _channel(1, "/cam_a")
        assert grouper.chunk_group_key(0, ch, _schema(1, "Image")) == 0

    def test_no_schema_falls_through_to_minus_one(self):
        grouper = PatternGrouper(
            topic_patterns=[],
            schema_patterns=[re.compile("Image")],
        )
        ch = _channel(1, "/cam_a")
        assert grouper.chunk_group_key(0, ch, None) == -1


# ---------------------------------------------------------------------------
# SchemaCompressionGrouper
# ---------------------------------------------------------------------------


class TestSchemaCompressionGrouper:
    def test_matching_channels_share_one_group_by_default(self):
        grouper = SchemaCompressionGrouper([re.compile("CompressedVideo")])
        cam_a = _channel(1, "/cam_a")
        cam_b = _channel(2, "/cam_b")
        schema = _schema(1, "foxglove_msgs/CompressedVideo")

        key_a = grouper.chunk_group_key(0, cam_a, schema)
        key_b = grouper.chunk_group_key(0, cam_b, schema)
        assert key_a == key_b
        assert key_a is not None

    def test_non_matching_schema_returns_none(self):
        grouper = SchemaCompressionGrouper([re.compile("CompressedVideo")])
        imu = _channel(3, "/imu")
        assert grouper.chunk_group_key(0, imu, _schema(2, "sensor_msgs/Imu")) is None

    def test_per_channel_splits_matching_channels_into_distinct_groups(self):
        grouper = SchemaCompressionGrouper([re.compile("CompressedVideo")], per_channel=True)
        cam_a = _channel(1, "/cam_a")
        cam_b = _channel(2, "/cam_b")
        schema = _schema(1, "foxglove_msgs/CompressedVideo")

        key_a = grouper.chunk_group_key(0, cam_a, schema)
        key_b = grouper.chunk_group_key(0, cam_b, schema)
        assert key_a is not None
        assert key_b is not None
        assert key_a != key_b
        # Non-matching channels still opt out even with per_channel set.
        imu = _channel(3, "/imu")
        assert grouper.chunk_group_key(0, imu, _schema(2, "sensor_msgs/Imu")) is None

    def test_compression_override_applies_to_matching_regardless_of_per_channel(self):
        for per_channel in (False, True):
            grouper = SchemaCompressionGrouper(
                [re.compile("CompressedVideo")], per_channel=per_channel
            )
            cam = _channel(1, "/cam_a")
            assert (
                grouper.chunk_compression(0, cam, _schema(1, "x/CompressedVideo"))
                == CompressionType.NONE
            )
            assert grouper.chunk_compression(0, cam, _schema(2, "x/Imu")) is None
