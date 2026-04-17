"""Tests for split processor classes."""

# ruff: noqa: ARG001 ARG005

from __future__ import annotations

import pytest
from pymcap_cli.core.processors import (
    SPLIT_REQUIRED,
    ChunkDecision,
    DurationSplitProcessor,
    ExpressionSplitProcessor,
    TimestampSplitProcessor,
    global_time_range,
)
from small_mcap import Channel, Message, Summary
from small_mcap import Statistics as SummaryStatistics
from small_mcap.records import LazyChunk


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


def _message(log_time: int) -> Message:
    return Message(channel_id=1, sequence=0, log_time=log_time, publish_time=log_time, data=b"")


def _make_summary(start_ns: int, end_ns: int, msg_count: int = 100) -> Summary:
    return Summary(
        schemas={},
        channels={},
        statistics=SummaryStatistics(
            message_count=msg_count,
            schema_count=0,
            channel_count=0,
            attachment_count=0,
            metadata_count=0,
            chunk_count=10,
            message_start_time=start_ns,
            message_end_time=end_ns,
            channel_message_counts={},
        ),
    )


# ---------------------------------------------------------------------------
# global_time_range helper
# ---------------------------------------------------------------------------


class TestGlobalTimeRange:
    def test_empty_list_returns_none(self):
        assert global_time_range([]) is None

    def test_all_none_summaries(self):
        assert global_time_range([None, None]) is None

    def test_single_summary(self):
        summaries = [_make_summary(100, 500)]
        assert global_time_range(summaries) == (100, 500)

    def test_multiple_summaries(self):
        summaries = [
            _make_summary(100, 300),
            _make_summary(50, 500),
            _make_summary(200, 400),
        ]
        assert global_time_range(summaries) == (50, 500)

    def test_mixed_none_and_valid(self):
        summaries = [None, _make_summary(100, 200), None]
        assert global_time_range(summaries) == (100, 200)


# ---------------------------------------------------------------------------
# DurationSplitProcessor
# ---------------------------------------------------------------------------


class TestDurationSplitProcessorValidation:
    def test_zero_duration_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DurationSplitProcessor(0)

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DurationSplitProcessor(-100)

    def test_positive_duration_ok(self):
        proc = DurationSplitProcessor(1_000_000_000)
        assert proc.duration_ns == 1_000_000_000


class TestDurationSplitProcessorInitialization:
    def test_no_summaries_no_boundaries(self):
        proc = DurationSplitProcessor(1_000_000_000)
        proc.initialize([])
        assert proc.boundaries == []
        assert proc.n_segments == 0

    def test_none_summaries_no_boundaries(self):
        proc = DurationSplitProcessor(1_000_000_000)
        proc.initialize([None])
        assert proc.boundaries == []

    def test_creates_boundaries_from_range(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        # Boundaries: 0, 100, 200, 251
        assert proc.boundaries == [0, 100, 200, 251]
        assert proc.n_segments == 3

    def test_single_segment(self):
        proc = DurationSplitProcessor(1000)
        proc.initialize([_make_summary(0, 500)])
        # Duration exceeds range, single segment: [0, 501]
        assert proc.boundaries == [0, 501]
        assert proc.n_segments == 1


class TestDurationSplitProcessorSegmentIndex:
    def test_first_segment(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc._segment_index(0) == 0
        assert proc._segment_index(50) == 0
        assert proc._segment_index(99) == 0

    def test_middle_segment(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc._segment_index(100) == 1
        assert proc._segment_index(150) == 1

    def test_last_segment(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc._segment_index(200) == 2
        assert proc._segment_index(250) == 2


class TestDurationSplitProcessorOnChunk:
    def test_no_boundaries_returns_continue(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([])
        assert proc.on_chunk(_lazy_chunk(0, 100), []) == ChunkDecision.CONTINUE

    def test_chunk_within_single_segment(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.on_chunk(_lazy_chunk(10, 50), []) == ChunkDecision.CONTINUE

    def test_chunk_spans_boundary(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        # Chunk from 50 to 150 spans boundary at 100
        assert proc.on_chunk(_lazy_chunk(50, 150), []) == ChunkDecision.DECODE

    def test_chunk_at_boundary_start(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        # Chunk from 100 to 199 is within segment 1
        assert proc.on_chunk(_lazy_chunk(100, 199), []) == ChunkDecision.CONTINUE


class TestDurationSplitProcessorRouteChunk:
    def test_no_boundaries_returns_zero(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([])
        assert proc.route_chunk(_lazy_chunk(0, 100)) == 0

    def test_returns_segment_key(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.route_chunk(_lazy_chunk(10, 50)) == 0
        assert proc.route_chunk(_lazy_chunk(100, 150)) == 1

    def test_returns_split_required(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.route_chunk(_lazy_chunk(50, 150)) is SPLIT_REQUIRED


class TestDurationSplitProcessorRouteMessage:
    def test_routes_to_correct_segment(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.route_message(_message(50)) == 0
        assert proc.route_message(_message(150)) == 1
        assert proc.route_message(_message(250)) == 2


class TestDurationSplitProcessorOutputKeys:
    def test_returns_keys_when_initialized(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.output_keys() == [0, 1, 2]

    def test_returns_empty_when_not_initialized(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([])
        assert proc.output_keys() == []


# ---------------------------------------------------------------------------
# TimestampSplitProcessor
# ---------------------------------------------------------------------------


class TestTimestampSplitProcessorValidation:
    def test_empty_split_points_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            TimestampSplitProcessor([])

    def test_valid_split_points_ok(self):
        proc = TimestampSplitProcessor([100, 200])
        assert proc.split_points == [100, 200]


class TestTimestampSplitProcessorInitialization:
    def test_no_summaries_no_boundaries(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([])
        assert proc.boundaries == []

    def test_creates_boundaries_from_range(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        # Boundaries: 0, 100, 200, 301
        assert proc.boundaries == [0, 100, 200, 301]
        assert proc.n_segments == 3

    def test_filters_split_points_outside_range(self):
        proc = TimestampSplitProcessor([100, 500, 1000])
        proc.initialize([_make_summary(0, 300)])
        # Only 100 is within range
        assert proc.boundaries == [0, 100, 301]
        assert proc.n_segments == 2

    def test_sorts_split_points(self):
        proc = TimestampSplitProcessor([200, 100])
        proc.initialize([_make_summary(0, 300)])
        assert proc.boundaries == [0, 100, 200, 301]


class TestTimestampSplitProcessorOnChunk:
    def test_no_boundaries_returns_continue(self):
        proc = TimestampSplitProcessor([100])
        proc.initialize([])
        assert proc.on_chunk(_lazy_chunk(0, 50), []) == ChunkDecision.CONTINUE

    def test_chunk_within_segment(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.on_chunk(_lazy_chunk(10, 50), []) == ChunkDecision.CONTINUE

    def test_chunk_spans_boundary(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.on_chunk(_lazy_chunk(50, 150), []) == ChunkDecision.DECODE


class TestTimestampSplitProcessorRouteChunk:
    def test_returns_segment_key(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.route_chunk(_lazy_chunk(10, 50)) == 0
        assert proc.route_chunk(_lazy_chunk(110, 150)) == 1
        assert proc.route_chunk(_lazy_chunk(210, 250)) == 2

    def test_returns_split_required(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.route_chunk(_lazy_chunk(50, 150)) is SPLIT_REQUIRED


class TestTimestampSplitProcessorRouteMessage:
    def test_routes_to_correct_segment(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.route_message(_message(50)) == 0
        assert proc.route_message(_message(150)) == 1
        assert proc.route_message(_message(250)) == 2


class TestTimestampSplitProcessorOutputKeys:
    def test_returns_keys_when_initialized(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.output_keys() == [0, 1, 2]


# ---------------------------------------------------------------------------
# ExpressionSplitProcessor
# ---------------------------------------------------------------------------


class TestExpressionSplitProcessor:
    def test_always_decodes(self):
        proc = ExpressionSplitProcessor(lambda msg, ch: 0)
        assert proc.on_chunk(_lazy_chunk(0, 100), []) == ChunkDecision.DECODE

    def test_route_chunk_always_split_required(self):
        proc = ExpressionSplitProcessor(lambda msg, ch: 0)
        assert proc.route_chunk(_lazy_chunk(0, 100)) is SPLIT_REQUIRED

    def test_route_message_calls_fn(self):
        channels: dict[int, Channel] = {}
        call_log = []

        def fn(msg: Message, ch: dict[int, Channel]) -> int:
            call_log.append(msg.log_time)
            return 0 if msg.log_time < 100 else 1

        proc = ExpressionSplitProcessor(fn, channels)
        assert proc.route_message(_message(50)) == 0
        assert proc.route_message(_message(150)) == 1
        assert call_log == [50, 150]

    def test_output_keys_returns_none(self):
        proc = ExpressionSplitProcessor(lambda msg, ch: 0)
        assert proc.output_keys() is None

    def test_string_keys(self):
        proc = ExpressionSplitProcessor(lambda msg, ch: "segment_a")
        assert proc.route_message(_message(0)) == "segment_a"
