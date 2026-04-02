"""Tests for core processor classes."""

from __future__ import annotations

import pytest
from pymcap_cli.core.processors import (
    Action,
    AlwaysDecodeProcessor,
    AttachmentFilterProcessor,
    ChunkDecision,
    MetadataFilterProcessor,
    TimeFilterProcessor,
    TopicFilterProcessor,
)
from small_mcap import Attachment, Channel, Message, Metadata
from small_mcap.records import LazyChunk


def _channel(topic: str) -> Channel:
    return Channel(id=1, schema_id=1, topic=topic, message_encoding="raw", metadata={})


def _message(log_time: int) -> Message:
    return Message(channel_id=1, sequence=0, log_time=log_time, publish_time=log_time, data=b"")


def _attachment(log_time: int) -> Attachment:
    return Attachment(log_time=log_time, create_time=0, name="a", media_type="text/plain", data=b"")


def _metadata() -> Metadata:
    return Metadata(name="info", metadata={})


def _lazy_chunk(start: int, end: int) -> LazyChunk:
    return LazyChunk(
        message_start_time=start,
        message_end_time=end,
        uncompressed_size=0,
        uncompressed_crc=0,
        compression="none",
        record_start=0,
        data_len=0,
    )


# ---------------------------------------------------------------------------
# TopicFilterProcessor
# ---------------------------------------------------------------------------


class TestTopicFilterProcessor:
    def test_no_filters_passes_all(self):
        proc = TopicFilterProcessor()
        assert proc.on_channel(_channel("/foo"), None) == Action.CONTINUE

    def test_include_matching_topic(self):
        proc = TopicFilterProcessor(include=["/foo"])
        assert proc.on_channel(_channel("/foo"), None) == Action.CONTINUE

    def test_include_non_matching_topic(self):
        proc = TopicFilterProcessor(include=["/foo"])
        assert proc.on_channel(_channel("/bar"), None) == Action.SKIP

    def test_include_partial_match_via_search(self):
        proc = TopicFilterProcessor(include=["foo"])
        assert proc.on_channel(_channel("/ns/foo/data"), None) == Action.CONTINUE

    def test_include_is_case_insensitive(self):
        proc = TopicFilterProcessor(include=["FOO"])
        assert proc.on_channel(_channel("/foo"), None) == Action.CONTINUE

    def test_exclude_matching_topic(self):
        proc = TopicFilterProcessor(exclude=["/foo"])
        assert proc.on_channel(_channel("/foo"), None) == Action.SKIP

    def test_exclude_non_matching_topic(self):
        proc = TopicFilterProcessor(exclude=["/foo"])
        assert proc.on_channel(_channel("/bar"), None) == Action.CONTINUE

    def test_include_takes_priority_over_exclude(self):
        # When include patterns are set, exclude is ignored
        proc = TopicFilterProcessor(include=["/foo"], exclude=["/foo"])
        assert proc.on_channel(_channel("/foo"), None) == Action.CONTINUE

    def test_multiple_include_patterns(self):
        proc = TopicFilterProcessor(include=["/cam", "/lidar"])
        assert proc.on_channel(_channel("/cam/front"), None) == Action.CONTINUE
        assert proc.on_channel(_channel("/lidar/top"), None) == Action.CONTINUE
        assert proc.on_channel(_channel("/imu"), None) == Action.SKIP

    def test_regex_include_pattern(self):
        proc = TopicFilterProcessor(include=[r"/cam/\d+"])
        assert proc.on_channel(_channel("/cam/0"), None) == Action.CONTINUE
        assert proc.on_channel(_channel("/cam/front"), None) == Action.SKIP

    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            TopicFilterProcessor(include=["[invalid"])


# ---------------------------------------------------------------------------
# MetadataFilterProcessor
# ---------------------------------------------------------------------------


class TestMetadataFilterProcessor:
    def test_include_true_passes(self):
        proc = MetadataFilterProcessor(include=True)
        assert proc.on_metadata(_metadata()) == Action.CONTINUE

    def test_include_false_skips(self):
        proc = MetadataFilterProcessor(include=False)
        assert proc.on_metadata(_metadata()) == Action.SKIP

    def test_default_includes(self):
        proc = MetadataFilterProcessor()
        assert proc.on_metadata(_metadata()) == Action.CONTINUE


# ---------------------------------------------------------------------------
# AttachmentFilterProcessor
# ---------------------------------------------------------------------------


class TestAttachmentFilterProcessor:
    def test_include_true_passes(self):
        proc = AttachmentFilterProcessor(include=True)
        assert proc.on_attachment(_attachment(0)) == Action.CONTINUE

    def test_include_false_skips(self):
        proc = AttachmentFilterProcessor(include=False)
        assert proc.on_attachment(_attachment(0)) == Action.SKIP

    def test_default_includes(self):
        proc = AttachmentFilterProcessor()
        assert proc.on_attachment(_attachment(0)) == Action.CONTINUE


# ---------------------------------------------------------------------------
# AlwaysDecodeProcessor
# ---------------------------------------------------------------------------


class TestAlwaysDecodeProcessor:
    def test_always_returns_decode(self):
        proc = AlwaysDecodeProcessor()
        chunk = _lazy_chunk(0, 100)
        assert proc.on_chunk(chunk, []) == ChunkDecision.DECODE


# ---------------------------------------------------------------------------
# TimeFilterProcessor — validation
# ---------------------------------------------------------------------------


class TestTimeFilterProcessorValidation:
    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match=r"start_ns.*must be less than.*end_ns"):
            TimeFilterProcessor(start_ns=100, end_ns=50)

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match=r"start_ns.*must be less than.*end_ns"):
            TimeFilterProcessor(start_ns=100, end_ns=100)

    def test_valid_range_accepted(self):
        proc = TimeFilterProcessor(start_ns=0, end_ns=100)
        assert proc.start == 0
        assert proc.end == 100

    def test_only_start_accepted(self):
        proc = TimeFilterProcessor(start_ns=500)
        assert proc.start == 500
        assert proc._has_start is True
        assert proc._has_end is False

    def test_only_end_accepted(self):
        proc = TimeFilterProcessor(end_ns=500)
        assert proc.end == 500
        assert proc._has_start is False
        assert proc._has_end is True

    def test_no_bounds_accepted(self):
        proc = TimeFilterProcessor()
        assert proc._has_start is False
        assert proc._has_end is False


# ---------------------------------------------------------------------------
# TimeFilterProcessor — on_message / on_attachment
# ---------------------------------------------------------------------------


class TestTimeFilterProcessorMessages:
    def test_message_within_range(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200)
        assert proc.on_message(_message(150)) == Action.CONTINUE

    def test_message_at_start_boundary(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200)
        assert proc.on_message(_message(100)) == Action.CONTINUE

    def test_message_before_start(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200)
        assert proc.on_message(_message(99)) == Action.SKIP

    def test_message_at_end_boundary_is_excluded(self):
        # end_ns is exclusive
        proc = TimeFilterProcessor(start_ns=100, end_ns=200)
        assert proc.on_message(_message(200)) == Action.SKIP

    def test_message_after_end(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200)
        assert proc.on_message(_message(201)) == Action.SKIP

    def test_message_only_start_bound(self):
        proc = TimeFilterProcessor(start_ns=100)
        assert proc.on_message(_message(100)) == Action.CONTINUE
        assert proc.on_message(_message(99)) == Action.SKIP

    def test_message_only_end_bound(self):
        proc = TimeFilterProcessor(end_ns=200)
        assert proc.on_message(_message(199)) == Action.CONTINUE
        assert proc.on_message(_message(200)) == Action.SKIP

    def test_attachment_within_range(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200)
        assert proc.on_attachment(_attachment(150)) == Action.CONTINUE

    def test_attachment_outside_range(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200)
        assert proc.on_attachment(_attachment(50)) == Action.SKIP


# ---------------------------------------------------------------------------
# TimeFilterProcessor — on_chunk
# ---------------------------------------------------------------------------


class TestTimeFilterProcessorChunks:
    def test_chunk_entirely_before_start(self):
        proc = TimeFilterProcessor(start_ns=500)
        chunk = _lazy_chunk(0, 400)
        assert proc.on_chunk(chunk, []) == ChunkDecision.SKIP

    def test_chunk_entirely_after_end(self):
        proc = TimeFilterProcessor(end_ns=500)
        chunk = _lazy_chunk(600, 900)
        assert proc.on_chunk(chunk, []) == ChunkDecision.SKIP

    def test_chunk_start_straddles_range_start(self):
        # chunk starts before range start → must decode per message
        proc = TimeFilterProcessor(start_ns=500)
        chunk = _lazy_chunk(400, 600)
        assert proc.on_chunk(chunk, []) == ChunkDecision.DECODE

    def test_chunk_end_straddles_range_end(self):
        # chunk ends at or after range end → must decode per message
        proc = TimeFilterProcessor(end_ns=500)
        chunk = _lazy_chunk(400, 600)
        assert proc.on_chunk(chunk, []) == ChunkDecision.DECODE

    def test_chunk_entirely_within_range(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=900)
        chunk = _lazy_chunk(200, 800)
        assert proc.on_chunk(chunk, []) == ChunkDecision.CONTINUE

    def test_chunk_no_bounds_always_continue(self):
        proc = TimeFilterProcessor()
        chunk = _lazy_chunk(0, 1_000_000)
        assert proc.on_chunk(chunk, []) == ChunkDecision.CONTINUE
