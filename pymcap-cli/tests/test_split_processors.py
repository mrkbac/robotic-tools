"""Tests for split processor classes."""

from __future__ import annotations

import pytest
from pymcap_cli.core.processors.base import SPLIT_REQUIRED, ChunkDecision
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from pymcap_cli.core.processors.utils import global_time_range
from small_mcap import Channel, Message, MessageIndex, Summary
from small_mcap import Statistics as SummaryStatistics

from tests.helpers import lazy_chunk


def _message(log_time: int, channel_id: int = 1, data: bytes = b"") -> Message:
    return Message(
        channel_id=channel_id,
        sequence=0,
        log_time=log_time,
        publish_time=log_time,
        data=data,
    )


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
        assert proc.on_chunk(lazy_chunk(0, 100), []) == ChunkDecision.CONTINUE

    def test_chunk_within_single_segment(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.on_chunk(lazy_chunk(10, 50), []) == ChunkDecision.CONTINUE

    def test_chunk_spans_boundary(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        # Chunk from 50 to 150 spans boundary at 100
        assert proc.on_chunk(lazy_chunk(50, 150), []) == ChunkDecision.DECODE

    def test_chunk_at_boundary_start(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        # Chunk from 100 to 199 is within segment 1
        assert proc.on_chunk(lazy_chunk(100, 199), []) == ChunkDecision.CONTINUE


class TestDurationSplitProcessorRouteChunk:
    def test_no_boundaries_returns_zero(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([])
        assert proc.route_chunk(lazy_chunk(0, 100)) == 0

    def test_returns_segment_key(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.route_chunk(lazy_chunk(10, 50)) == 0
        assert proc.route_chunk(lazy_chunk(100, 150)) == 1

    def test_returns_split_required(self):
        proc = DurationSplitProcessor(100)
        proc.initialize([_make_summary(0, 250)])
        assert proc.route_chunk(lazy_chunk(50, 150)) is SPLIT_REQUIRED


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
        assert proc.on_chunk(lazy_chunk(0, 50), []) == ChunkDecision.CONTINUE

    def test_chunk_within_segment(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.on_chunk(lazy_chunk(10, 50), []) == ChunkDecision.CONTINUE

    def test_chunk_spans_boundary(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.on_chunk(lazy_chunk(50, 150), []) == ChunkDecision.DECODE


class TestTimestampSplitProcessorRouteChunk:
    def test_returns_segment_key(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.route_chunk(lazy_chunk(10, 50)) == 0
        assert proc.route_chunk(lazy_chunk(110, 150)) == 1
        assert proc.route_chunk(lazy_chunk(210, 250)) == 2

    def test_returns_split_required(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize([_make_summary(0, 300)])
        assert proc.route_chunk(lazy_chunk(50, 150)) is SPLIT_REQUIRED


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


def _make_expression_proc(path: str = "/t.field") -> ExpressionSplitProcessor:
    """Construct an ExpressionSplitProcessor with a pre-wired fake decoder.

    Bypasses CDR/JSON decoding so tests can drive the processor with synthetic
    ``Message.data`` payloads — the fake decoder treats the bytes as a UTF-8
    string and returns an object exposing that string as ``.field``.
    """
    proc = ExpressionSplitProcessor(path)
    proc.channels[1] = Channel(id=1, schema_id=1, topic="/t", message_encoding="json", metadata={})
    proc._decoders[1] = lambda data: type("M", (), {"field": bytes(data).decode()})()
    return proc


class TestExpressionSplitProcessor:
    def test_invalid_path_raises(self):
        # ros-parser raises a lark UnexpectedToken (subclass of LarkError);
        # users see whatever the underlying parser surfaces — just assert the
        # processor does not silently swallow bogus input.
        with pytest.raises(Exception, match="Unexpected"):
            ExpressionSplitProcessor("not a valid path !!!")

    def test_decodes_chunks_with_target_topic(self):
        proc = _make_expression_proc()
        assert (
            proc.on_chunk(
                lazy_chunk(0, 100),
                [MessageIndex(channel_id=1, timestamps=[], offsets=[])],
            )
            == ChunkDecision.DECODE
        )

    def test_fast_copies_chunks_without_target_topic(self):
        proc = _make_expression_proc()
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )
        assert (
            proc.on_chunk(
                lazy_chunk(0, 100),
                [MessageIndex(channel_id=2, timestamps=[], offsets=[])],
            )
            == ChunkDecision.CONTINUE
        )

    def test_route_chunk_returns_current_segment_index(self):
        proc = _make_expression_proc()
        assert proc.route_chunk(lazy_chunk(0, 100)) == 0
        # Advance past the first target message (no transition yet)…
        proc.route_message(_message(0, channel_id=1, data=b"alpha"))
        assert proc.route_chunk(lazy_chunk(0, 100)) == 0
        # …then on value change the sticky segment advances.
        proc.route_message(_message(1, channel_id=1, data=b"beta"))
        assert proc.route_chunk(lazy_chunk(0, 100)) == 1

    def test_value_runs_share_one_segment(self):
        proc = _make_expression_proc()
        assert proc.route_message(_message(0, channel_id=1, data=b"alpha")) == 0
        assert proc.route_message(_message(1, channel_id=1, data=b"alpha")) == 0
        assert proc.route_message(_message(2, channel_id=1, data=b"alpha")) == 0

    def test_value_change_triggers_new_segment(self):
        proc = _make_expression_proc()
        assert proc.route_message(_message(0, channel_id=1, data=b"alpha")) == 0
        assert proc.route_message(_message(1, channel_id=1, data=b"beta")) == 1
        assert proc.route_message(_message(2, channel_id=1, data=b"beta")) == 1
        assert proc.route_message(_message(3, channel_id=1, data=b"alpha")) == 2

    def test_sticky_between_target_messages(self):
        proc = _make_expression_proc()
        proc.route_message(_message(0, channel_id=1, data=b"alpha"))
        proc.route_message(_message(1, channel_id=1, data=b"beta"))  # now in seg 1
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )
        assert proc.route_message(_message(10, channel_id=2)) == 1

    def test_starts_at_zero_before_any_target_message(self):
        proc = _make_expression_proc()
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )
        assert proc.route_message(_message(0, channel_id=2)) == 0

    def test_output_keys_returns_none(self):
        proc = _make_expression_proc()
        assert proc.output_keys() is None

    def test_falls_back_to_decode_when_channels_unknown(self):
        """Before any channel is seen, chunks must DECODE to stay correct."""
        proc = ExpressionSplitProcessor("/t.field")
        assert (
            proc.on_chunk(
                lazy_chunk(0, 100),
                [MessageIndex(channel_id=1, timestamps=[], offsets=[])],
            )
            == ChunkDecision.DECODE
        )
