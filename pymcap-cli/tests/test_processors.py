"""Tests for core processor classes."""

from __future__ import annotations

import pytest
from pymcap_cli.core.processors.always_decode import AlwaysDecodeProcessor
from pymcap_cli.core.processors.attachment_filter import AttachmentFilterProcessor
from pymcap_cli.core.processors.base import Action, ChunkDecision
from pymcap_cli.core.processors.metadata_filter import MetadataFilterProcessor
from pymcap_cli.core.processors.time_filter import TimeFilterProcessor
from pymcap_cli.core.processors.topic_filter import TopicFilterProcessor
from small_mcap import Attachment, Channel, Message, Metadata

from tests.helpers import lazy_chunk


def _channel(topic: str) -> Channel:
    return Channel(id=1, schema_id=1, topic=topic, message_encoding="raw", metadata={})


def _message(log_time: int) -> Message:
    return Message(channel_id=1, sequence=0, log_time=log_time, publish_time=log_time, data=b"")


def _attachment(log_time: int) -> Attachment:
    return Attachment(log_time=log_time, create_time=0, name="a", media_type="text/plain", data=b"")


def _metadata() -> Metadata:
    return Metadata(name="info", metadata={})


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
        chunk = lazy_chunk(0, 100)
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
        assert proc.start_ns == 0
        assert proc.end_ns == 100

    def test_only_start_accepted(self):
        proc = TimeFilterProcessor(start_ns=500)
        assert proc.start_ns == 500
        assert proc.end_ns is None

    def test_only_end_accepted(self):
        proc = TimeFilterProcessor(end_ns=500)
        assert proc.start_ns is None
        assert proc.end_ns == 500

    def test_no_bounds_accepted(self):
        proc = TimeFilterProcessor()
        assert proc.start_ns is None
        assert proc.end_ns is None


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
        chunk = lazy_chunk(0, 400)
        assert proc.on_chunk(chunk, []) == ChunkDecision.SKIP

    def test_chunk_entirely_after_end(self):
        proc = TimeFilterProcessor(end_ns=500)
        chunk = lazy_chunk(600, 900)
        assert proc.on_chunk(chunk, []) == ChunkDecision.SKIP

    def test_chunk_start_straddles_range_start(self):
        # chunk starts before range start → must decode per message
        proc = TimeFilterProcessor(start_ns=500)
        chunk = lazy_chunk(400, 600)
        assert proc.on_chunk(chunk, []) == ChunkDecision.DECODE

    def test_chunk_end_straddles_range_end(self):
        # chunk ends at or after range end → must decode per message
        proc = TimeFilterProcessor(end_ns=500)
        chunk = lazy_chunk(400, 600)
        assert proc.on_chunk(chunk, []) == ChunkDecision.DECODE

    def test_chunk_entirely_within_range(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=900)
        chunk = lazy_chunk(200, 800)
        assert proc.on_chunk(chunk, []) == ChunkDecision.CONTINUE

    def test_chunk_no_bounds_always_continue(self):
        proc = TimeFilterProcessor()
        chunk = lazy_chunk(0, 1_000_000)
        assert proc.on_chunk(chunk, []) == ChunkDecision.CONTINUE


# ---------------------------------------------------------------------------
# TimeFilterProcessor — invert
# ---------------------------------------------------------------------------


class TestTimeFilterProcessorInvert:
    def test_message_inside_range_is_skipped(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200, invert=True)
        assert proc.on_message(_message(150)) == Action.SKIP

    def test_message_outside_range_passes(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200, invert=True)
        assert proc.on_message(_message(50)) == Action.CONTINUE
        assert proc.on_message(_message(250)) == Action.CONTINUE

    def test_attachment_inside_range_is_skipped(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=200, invert=True)
        assert proc.on_attachment(_attachment(150)) == Action.SKIP

    def test_chunk_fully_inside_skip(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=900, invert=True)
        chunk = lazy_chunk(200, 800)
        assert proc.on_chunk(chunk, []) == ChunkDecision.SKIP

    def test_chunk_inside_open_start_window_skips(self):
        proc = TimeFilterProcessor(start_ns=100, invert=True)
        chunk = lazy_chunk(200, 800)
        assert proc.on_chunk(chunk, []) == ChunkDecision.SKIP

    def test_chunk_inside_open_end_window_skips(self):
        proc = TimeFilterProcessor(end_ns=900, invert=True)
        chunk = lazy_chunk(200, 800)
        assert proc.on_chunk(chunk, []) == ChunkDecision.SKIP

    def test_chunk_fully_outside_continue(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=900, invert=True)
        chunk_before = lazy_chunk(0, 50)
        chunk_after = lazy_chunk(1000, 1200)
        assert proc.on_chunk(chunk_before, []) == ChunkDecision.CONTINUE
        assert proc.on_chunk(chunk_after, []) == ChunkDecision.CONTINUE

    def test_chunk_spanning_decode(self):
        proc = TimeFilterProcessor(start_ns=100, end_ns=900, invert=True)
        chunk = lazy_chunk(50, 200)
        assert proc.on_chunk(chunk, []) == ChunkDecision.DECODE


# ---------------------------------------------------------------------------
# TimeFilterProcessor — RelativeTime resolution
# ---------------------------------------------------------------------------


class TestTimeFilterProcessorRelativeTime:
    def _summary(self, start: int, end: int) -> object:
        # Minimal stub matching the only attributes initialize() reads.
        from dataclasses import dataclass  # noqa: PLC0415

        @dataclass
        class _Stats:
            message_start_time: int
            message_end_time: int

        @dataclass
        class _Summary:
            statistics: _Stats

        return _Summary(_Stats(start, end))

    def test_relative_start_resolved(self):
        from pymcap_cli.utils import RelativeTime  # noqa: PLC0415

        proc = TimeFilterProcessor(start_ns=RelativeTime("start", 5))
        proc.initialize([self._summary(100, 1000)])
        assert proc.start_ns == 105
        assert proc.end_ns is None

    def test_relative_start_preserves_zero_global_start(self):
        from pymcap_cli.utils import RelativeTime  # noqa: PLC0415

        proc = TimeFilterProcessor(start_ns=RelativeTime("start", 5))
        proc.initialize([self._summary(0, 1000), self._summary(50, 2000)])
        assert proc.start_ns == 5

    def test_relative_requires_summary_statistics(self):
        from pymcap_cli.utils import RelativeTime  # noqa: PLC0415

        proc = TimeFilterProcessor(start_ns=RelativeTime("start", 5))
        with pytest.raises(ValueError, match="summary statistics"):
            proc.initialize([None])

    def test_relative_end_resolved(self):
        from pymcap_cli.utils import RelativeTime  # noqa: PLC0415

        proc = TimeFilterProcessor(end_ns=RelativeTime("end", -10))
        proc.initialize([self._summary(100, 1000)])
        assert proc.end_ns == 990

    def test_relative_inverted_after_resolve_raises(self):
        from pymcap_cli.utils import RelativeTime  # noqa: PLC0415

        proc = TimeFilterProcessor(
            start_ns=RelativeTime("end", -1),
            end_ns=RelativeTime("start", 1),
        )
        with pytest.raises(ValueError, match=r"resolved start_ns"):
            proc.initialize([self._summary(100, 1000)])


# ---------------------------------------------------------------------------
# TopicFilterProcessor — invert
# ---------------------------------------------------------------------------


class TestTopicFilterProcessorInvert:
    def test_invert_include_skips_match(self):
        proc = TopicFilterProcessor(include=["/foo"], invert=True)
        assert proc.on_channel(_channel("/foo"), None) == Action.SKIP
        assert proc.on_channel(_channel("/bar"), None) == Action.CONTINUE

    def test_invert_exclude_passes_match(self):
        proc = TopicFilterProcessor(exclude=["/foo"], invert=True)
        # Without invert, /foo would be SKIP. With invert, CONTINUE.
        assert proc.on_channel(_channel("/foo"), None) == Action.CONTINUE
        assert proc.on_channel(_channel("/bar"), None) == Action.SKIP


# ---------------------------------------------------------------------------
# Topic glob → regex via InputOptions.from_args
# ---------------------------------------------------------------------------


class TestTopicGlobConversion:
    def test_glob_translated_into_include_topics(self):
        from pymcap_cli.core.mcap_processor import InputOptions  # noqa: PLC0415

        opts = InputOptions.from_args(include_topic_glob=["sensors/*"])
        # fnmatch.translate produces a Python regex string
        assert any("sensors" in regex for regex in opts.include_topics)
        # That regex compiles and matches as expected
        topic_filter = next(p for p in opts.processors if isinstance(p, TopicFilterProcessor))
        assert topic_filter.on_channel(_channel("sensors/lidar"), None) == Action.CONTINUE
        assert topic_filter.on_channel(_channel("/prefix/sensors/lidar"), None) == Action.SKIP
        assert topic_filter.on_channel(_channel("/scan"), None) == Action.SKIP

    def test_exclude_glob_translated(self):
        from pymcap_cli.core.mcap_processor import InputOptions  # noqa: PLC0415

        opts = InputOptions.from_args(exclude_topic_glob=["debug/*"])
        topic_filter = next(p for p in opts.processors if isinstance(p, TopicFilterProcessor))
        assert topic_filter.on_channel(_channel("debug/log"), None) == Action.SKIP
        assert topic_filter.on_channel(_channel("/prefix/debug/log"), None) == Action.CONTINUE
        assert topic_filter.on_channel(_channel("/scan"), None) == Action.CONTINUE
