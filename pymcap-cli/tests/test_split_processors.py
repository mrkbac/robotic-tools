"""Tests for split processor classes."""

from __future__ import annotations

import json

import pytest
from pymcap_cli.core.processors.base import SPLIT_REQUIRED, ChunkDecision
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from pymcap_cli.core.processors.utils import global_time_range
from small_mcap import Channel, Message, MessageIndex, Schema, Summary
from small_mcap import Statistics as SummaryStatistics

from tests.helpers import chunk_context, lazy_chunk, message_context, pipeline_context


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


def _on_chunk(proc, chunk, indexes=()) -> ChunkDecision:
    return proc.on_chunk(chunk_context(indexes), chunk)


def _route_chunk(proc, chunk):
    result = proc.route_chunk(chunk_context(), chunk)
    if result is SPLIT_REQUIRED:
        return SPLIT_REQUIRED
    routes = list(result)
    assert len(routes) == 1
    return routes[0]


def _routes(proc, message: Message) -> list[int | str]:
    return list(proc.route_message(message_context(message), message))


def _route_message(proc, message: Message) -> int | str:
    routes = _routes(proc, message)
    assert routes
    return routes[0]


# ---------------------------------------------------------------------------
# global_time_range helper
# ---------------------------------------------------------------------------


class TestGlobalTimeRange:
    def test_empty_list_returns_none(self):
        assert global_time_range(pipeline_context([])) is None

    def test_all_none_summaries(self):
        assert global_time_range(pipeline_context([None, None])) is None

    def test_single_summary(self):
        summaries = [_make_summary(100, 500)]
        assert global_time_range(pipeline_context(summaries)) == (100, 500)

    def test_multiple_summaries(self):
        summaries = [
            _make_summary(100, 300),
            _make_summary(50, 500),
            _make_summary(200, 400),
        ]
        assert global_time_range(pipeline_context(summaries)) == (50, 500)

    def test_mixed_none_and_valid(self):
        summaries = [None, _make_summary(100, 200), None]
        assert global_time_range(pipeline_context(summaries)) == (100, 200)


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


class TestDurationSplitProcessorAnchoring:
    # DurationSplit is streaming-anchor — no summary required. The first
    # timestamp it sees becomes the origin and segments grow from there.

    def test_anchor_is_set_on_first_timestamp_seen(self):
        proc = DurationSplitProcessor(100)
        # _segment_index lazily anchors on first call.
        assert _route_message(proc, _message(0)) == 0
        assert proc._anchor_ns == 0

    def test_anchor_is_first_seen_even_when_nonzero(self):
        proc = DurationSplitProcessor(100)
        # First message arrives at t=1000 — that becomes the anchor.
        assert _route_message(proc, _message(1000)) == 0
        assert _route_message(proc, _message(1050)) == 0
        assert _route_message(proc, _message(1100)) == 1
        assert _route_message(proc, _message(1250)) == 2

    def test_no_initialize_required_to_route(self):
        # Crucial property: works without ever calling initialize.
        proc = DurationSplitProcessor(100)
        assert _on_chunk(proc, lazy_chunk(0, 50)) == ChunkDecision.CONTINUE
        assert _on_chunk(proc, lazy_chunk(50, 150)) == ChunkDecision.DECODE


class TestDurationSplitProcessorOnChunk:
    def test_chunk_within_single_segment(self):
        proc = DurationSplitProcessor(100)
        assert _on_chunk(proc, lazy_chunk(0, 0)) == ChunkDecision.CONTINUE  # anchor
        assert _on_chunk(proc, lazy_chunk(10, 50)) == ChunkDecision.CONTINUE

    def test_chunk_spans_boundary(self):
        proc = DurationSplitProcessor(100)
        _on_chunk(proc, lazy_chunk(0, 0))  # anchor at 0
        # Chunk from 50 to 150 spans boundary at 100
        assert _on_chunk(proc, lazy_chunk(50, 150)) == ChunkDecision.DECODE

    def test_chunk_at_boundary_start(self):
        proc = DurationSplitProcessor(100)
        _on_chunk(proc, lazy_chunk(0, 0))  # anchor at 0
        # Chunk from 100 to 199 is within segment 1
        assert _on_chunk(proc, lazy_chunk(100, 199)) == ChunkDecision.CONTINUE


class TestDurationSplitProcessorRouteChunk:
    def test_returns_segment_key(self):
        proc = DurationSplitProcessor(100)
        _on_chunk(proc, lazy_chunk(0, 0))  # anchor at 0
        assert _route_chunk(proc, lazy_chunk(10, 50)) == 0
        assert _route_chunk(proc, lazy_chunk(100, 150)) == 1

    def test_returns_split_required(self):
        proc = DurationSplitProcessor(100)
        _on_chunk(proc, lazy_chunk(0, 0))  # anchor at 0
        assert _route_chunk(proc, lazy_chunk(50, 150)) is SPLIT_REQUIRED


class TestDurationSplitProcessorRouteMessage:
    def test_routes_to_correct_segment(self):
        proc = DurationSplitProcessor(100)
        assert _route_message(proc, _message(0)) == 0  # anchors at 0
        assert _route_message(proc, _message(150)) == 1
        assert _route_message(proc, _message(250)) == 2


class TestDurationSplitProcessorOutputKeys:
    def test_dynamic_segments(self):
        # Segments are created lazily — no upfront key list.
        proc = DurationSplitProcessor(100)
        assert proc.output_segments() is None


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
    def test_no_summaries_falls_back_to_sentinel_bounds(self):
        # Without summary statistics, the processor still works — user-
        # supplied absolute split_points are used between sentinel bounds.
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([]))
        assert proc.boundaries[0] == 0
        assert proc.boundaries[1:-1] == [100, 200]
        # Trailing sentinel is huge so segment routing covers the whole int range.
        assert proc.boundaries[-1] > 200
        assert proc.n_segments == 3

    def test_creates_boundaries_from_range(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        # Boundaries: 0, 100, 200, 301
        assert proc.boundaries == [0, 100, 200, 301]
        assert proc.n_segments == 3

    def test_filters_split_points_outside_range(self):
        proc = TimestampSplitProcessor([100, 500, 1000])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        # Only 100 is within range
        assert proc.boundaries == [0, 100, 301]
        assert proc.n_segments == 2

    def test_sorts_split_points(self):
        proc = TimestampSplitProcessor([200, 100])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        assert proc.boundaries == [0, 100, 200, 301]


class TestTimestampSplitProcessorOnChunk:
    def test_chunk_within_first_segment_under_sentinel_bounds_continues(self):
        # No summary → sentinel bounds. Chunk fully inside the first segment
        # ([0, 100)) fast-copies.
        proc = TimestampSplitProcessor([100])
        proc.initialize(pipeline_context([]))
        assert _on_chunk(proc, lazy_chunk(0, 50)) == ChunkDecision.CONTINUE

    def test_chunk_within_segment(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        assert _on_chunk(proc, lazy_chunk(10, 50)) == ChunkDecision.CONTINUE

    def test_chunk_spans_boundary(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        assert _on_chunk(proc, lazy_chunk(50, 150)) == ChunkDecision.DECODE


class TestTimestampSplitProcessorRouteChunk:
    def test_returns_segment_key(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        assert _route_chunk(proc, lazy_chunk(10, 50)) == 0
        assert _route_chunk(proc, lazy_chunk(110, 150)) == 1
        assert _route_chunk(proc, lazy_chunk(210, 250)) == 2

    def test_returns_split_required(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        assert _route_chunk(proc, lazy_chunk(50, 150)) is SPLIT_REQUIRED


class TestTimestampSplitProcessorRouteMessage:
    def test_routes_to_correct_segment(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        assert _route_message(proc, _message(50)) == 0
        assert _route_message(proc, _message(150)) == 1
        assert _route_message(proc, _message(250)) == 2


class TestTimestampSplitProcessorOutputKeys:
    def test_returns_keys_when_initialized(self):
        proc = TimestampSplitProcessor([100, 200])
        proc.initialize(pipeline_context([_make_summary(0, 300)]))
        segments = proc.output_segments()
        assert segments is not None
        assert [segment.key for segment in segments] == [0, 1, 2]


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

    def test_stream_modifier_path_raises(self):
        with pytest.raises(ValueError, match="not supported in split expressions"):
            ExpressionSplitProcessor("/t.field.@@delta")

    def test_variables_are_passed_to_expression(self) -> None:
        proc = ExpressionSplitProcessor("/t.field{==$expected}", variables={"expected": "alpha"})
        proc.channels[1] = Channel(
            id=1, schema_id=1, topic="/t", message_encoding="json", metadata={}
        )
        proc._decoders[1] = lambda data: type("M", (), {"field": bytes(data).decode()})()

        assert _route_message(proc, _message(0, channel_id=1, data=b"alpha")) == 0
        assert _route_message(proc, _message(1, channel_id=1, data=b"beta")) == 1

    def test_predicate_results_are_normalized_to_boolean(self) -> None:
        proc = ExpressionSplitProcessor("/t.items[:]{score>0.8}")
        proc.channels[1] = Channel(
            id=1, schema_id=1, topic="/t", message_encoding="json", metadata={}
        )
        proc._decoders[1] = lambda data: json.loads(bytes(data))

        first_match = b'{"items":[{"score":0.9,"name":"first"}]}'
        different_match = b'{"items":[{"score":0.95,"name":"second"}]}'
        no_match = b'{"items":[{"score":0.2,"name":"third"}]}'

        assert _route_message(proc, _message(0, data=first_match)) == 0
        assert _route_message(proc, _message(1, data=different_match)) == 0
        assert _route_message(proc, _message(2, data=no_match)) == 1
        assert proc.template_fields(0) == {"value": True}
        assert proc.template_fields(1) == {"value": False}

    def test_complex_non_predicate_result_is_rejected(self) -> None:
        proc = ExpressionSplitProcessor("/t.items[:]")
        proc.channels[1] = Channel(
            id=1, schema_id=1, topic="/t", message_encoding="json", metadata={}
        )
        proc._decoders[1] = lambda data: json.loads(bytes(data))

        with pytest.raises(ValueError, match="primitive"):
            _routes(proc, _message(0, data=b'{"items":[1,2]}'))

    def test_complex_non_predicate_schema_is_rejected_before_messages(self) -> None:
        proc = ExpressionSplitProcessor("/t.items")
        schema = Schema(
            id=1,
            name="example_msgs/msg/State",
            encoding="ros2msg",
            data=b"int8[] items\n",
        )

        with pytest.raises(ValueError, match="primitive"):
            proc._validate_path(schema)

    def test_skip_value_suppresses_matching_runs(self) -> None:
        proc = ExpressionSplitProcessor("/t.field", skip_values=("neutral",))
        proc.channels[1] = Channel(
            id=1, schema_id=1, topic="/t", message_encoding="json", metadata={}
        )
        proc._decoders[1] = lambda data: type("M", (), {"field": bytes(data).decode()})()

        assert _routes(proc, _message(0, data=b"neutral")) == []
        assert _routes(proc, _message(1, data=b"forward")) == [1]
        assert _routes(proc, _message(2, data=b"neutral")) == []
        assert _routes(proc, _message(3, data=b"reverse")) == [3]
        assert proc.template_fields(1) == {"value": "forward"}
        assert proc.template_fields(3) == {"value": "reverse"}

    def test_skip_numeric_zero_does_not_skip_boolean_false(self) -> None:
        proc = ExpressionSplitProcessor("/t.field", skip_values=(0,))
        proc.channels[1] = Channel(
            id=1, schema_id=1, topic="/t", message_encoding="json", metadata={}
        )
        proc._decoders[1] = lambda _data: type("M", (), {"field": False})()

        assert _routes(proc, _message(0)) == [0]

    def test_value_required_drops_messages_before_first_expression_value(self) -> None:
        proc = ExpressionSplitProcessor("/t.field", require_value=True)
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )

        assert _routes(proc, _message(0, channel_id=2)) == []

    def test_decodes_chunks_with_target_topic(self):
        proc = _make_expression_proc()
        indexes = [MessageIndex(channel_id=1, timestamps=[], offsets=[])]
        assert _on_chunk(proc, lazy_chunk(0, 100), indexes) == ChunkDecision.DECODE

    def test_fast_copies_chunks_without_target_topic(self):
        proc = _make_expression_proc()
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )
        indexes = [MessageIndex(channel_id=2, timestamps=[], offsets=[])]
        assert _on_chunk(proc, lazy_chunk(0, 100), indexes) == ChunkDecision.CONTINUE

    def test_route_chunk_returns_current_segment_index(self):
        proc = _make_expression_proc()
        assert _route_chunk(proc, lazy_chunk(0, 100)) == 0
        # Advance past the first target message (no transition yet)…
        _route_message(proc, _message(0, channel_id=1, data=b"alpha"))
        assert _route_chunk(proc, lazy_chunk(0, 100)) == 0
        # …then on value change the sticky segment advances.
        _route_message(proc, _message(1, channel_id=1, data=b"beta"))
        assert _route_chunk(proc, lazy_chunk(0, 100)) == 1

    def test_value_runs_share_one_segment(self):
        proc = _make_expression_proc()
        assert _route_message(proc, _message(0, channel_id=1, data=b"alpha")) == 0
        assert _route_message(proc, _message(1, channel_id=1, data=b"alpha")) == 0
        assert _route_message(proc, _message(2, channel_id=1, data=b"alpha")) == 0

    def test_value_change_triggers_new_segment(self):
        proc = _make_expression_proc()
        assert _route_message(proc, _message(0, channel_id=1, data=b"alpha")) == 0
        assert _route_message(proc, _message(1, channel_id=1, data=b"beta")) == 1
        assert _route_message(proc, _message(2, channel_id=1, data=b"beta")) == 1
        assert _route_message(proc, _message(3, channel_id=1, data=b"alpha")) == 2

    def test_sticky_between_target_messages(self):
        proc = _make_expression_proc()
        _route_message(proc, _message(0, channel_id=1, data=b"alpha"))
        _route_message(proc, _message(1, channel_id=1, data=b"beta"))  # now in seg 1
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )
        assert _route_message(proc, _message(10, channel_id=2)) == 1

    def test_starts_at_zero_before_any_target_message(self):
        proc = _make_expression_proc()
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )
        assert _route_message(proc, _message(0, channel_id=2)) == 0

    def test_output_keys_returns_none(self):
        proc = _make_expression_proc()
        assert proc.output_segments() is None

    def test_falls_back_to_decode_when_channels_unknown(self):
        """Before any channel is seen, chunks must DECODE to stay correct."""
        proc = ExpressionSplitProcessor("/t.field")
        indexes = [MessageIndex(channel_id=1, timestamps=[], offsets=[])]
        assert _on_chunk(proc, lazy_chunk(0, 100), indexes) == ChunkDecision.DECODE


def _make_hysteresis_proc(
    *,
    hysteresis_ns: int | None = None,
    hysteresis_count: int | None = None,
    trailing_context_ns: int | None = None,
    trailing_context_count: int | None = None,
) -> ExpressionSplitProcessor:
    proc = ExpressionSplitProcessor(
        "/t.field",
        hysteresis_ns=hysteresis_ns,
        hysteresis_count=hysteresis_count,
        trailing_context_ns=trailing_context_ns,
        trailing_context_count=trailing_context_count,
    )
    proc.channels[1] = Channel(id=1, schema_id=1, topic="/t", message_encoding="json", metadata={})
    proc._decoders[1] = lambda data: type("M", (), {"field": bytes(data).decode()})()
    return proc


# ---------------------------------------------------------------------------
# ExpressionSplitProcessor — hysteresis
# ---------------------------------------------------------------------------


class TestExpressionSplitHysteresis:
    def test_count_hysteresis_holds_segment(self):
        # Need 3 sustained reads of "beta" before transition fires.
        proc = _make_hysteresis_proc(hysteresis_count=3)
        assert _route_message(proc, _message(0, data=b"alpha")) == 0
        # First "beta" is just a candidate; segment stays at 0.
        assert _route_message(proc, _message(1, data=b"beta")) == 0
        assert _route_message(proc, _message(2, data=b"beta")) == 0
        # Third "beta" hits the threshold → commit, segment becomes 1.
        assert _route_message(proc, _message(3, data=b"beta")) == 1

    def test_count_hysteresis_resets_on_flap(self):
        proc = _make_hysteresis_proc(hysteresis_count=3)
        assert _route_message(proc, _message(0, data=b"alpha")) == 0
        assert _route_message(proc, _message(1, data=b"beta")) == 0
        assert _route_message(proc, _message(2, data=b"beta")) == 0
        # Flap back to alpha: candidate cleared.
        assert _route_message(proc, _message(3, data=b"alpha")) == 0
        # Next beta starts fresh.
        assert _route_message(proc, _message(4, data=b"beta")) == 0
        assert _route_message(proc, _message(5, data=b"beta")) == 0
        assert _route_message(proc, _message(6, data=b"beta")) == 1

    def test_time_hysteresis_holds_segment(self):
        # 500ms time threshold.
        proc = _make_hysteresis_proc(hysteresis_ns=500)
        assert _route_message(proc, _message(0, data=b"alpha")) == 0
        assert _route_message(proc, _message(100, data=b"beta")) == 0
        assert _route_message(proc, _message(400, data=b"beta")) == 0
        assert _route_message(proc, _message(600, data=b"beta")) == 1  # 600 - 100 >= 500

    def test_time_and_count_both_required(self):
        # Both must be satisfied.
        proc = _make_hysteresis_proc(hysteresis_ns=500, hysteresis_count=2)
        assert _route_message(proc, _message(0, data=b"alpha")) == 0
        # First "beta": candidate count=1 (need 2), time=0 (need 500). Neither.
        assert _route_message(proc, _message(200, data=b"beta")) == 0
        # Second beta: count=2 OK, but time=600-200=400 < 500 → still hold.
        assert _route_message(proc, _message(600, data=b"beta")) == 0
        # Third beta: count=3, time=700-200=500 → both OK, commit.
        assert _route_message(proc, _message(700, data=b"beta")) == 1

    def test_no_hysteresis_commits_immediately(self):
        proc = _make_hysteresis_proc()
        assert _route_message(proc, _message(0, data=b"alpha")) == 0
        assert _route_message(proc, _message(1, data=b"beta")) == 1


# ---------------------------------------------------------------------------
# ExpressionSplitProcessor — trailing context
# ---------------------------------------------------------------------------


class TestExpressionSplitTrailingContext:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"hysteresis_ns": 0},
            {"hysteresis_count": 0},
            {"trailing_context_ns": 0},
            {"trailing_context_count": 0},
        ],
    )
    def test_rejects_non_positive_thresholds(self, kwargs: dict[str, int]):
        with pytest.raises(ValueError, match="positive"):
            _make_hysteresis_proc(**kwargs)

    def test_trailing_count_duplicates_into_previous_segment(self):
        proc = _make_hysteresis_proc(trailing_context_count=2)
        assert _route_message(proc, _message(0, data=b"alpha")) == 0
        # Transition at t=10: now in segment 1, and also written to segment 0.
        assert _routes(proc, _message(10, data=b"beta")) == [1, 0]
        msg1 = _message(11, data=b"beta")
        assert _routes(proc, msg1) == [1, 0]
        msg2 = _message(12, data=b"beta")
        assert _routes(proc, msg2) == [1]
        # Window exhausted.
        msg3 = _message(13, data=b"beta")
        assert _routes(proc, msg3) == [1]

    def test_trailing_time_duplicates_until_window_closes(self):
        proc = _make_hysteresis_proc(trailing_context_ns=100)
        assert _route_message(proc, _message(0, data=b"alpha")) == 0
        # Transition at t=10: tail until 110.
        assert _routes(proc, _message(10, data=b"beta")) == [1, 0]
        msg = _message(50, data=b"beta")
        assert _routes(proc, msg) == [1, 0]
        # 110 is the cutoff (>110 closes).
        msg = _message(120, data=b"beta")
        assert _routes(proc, msg) == [1]

    def test_trailing_only_target_topic_duplicates(self):
        proc = _make_hysteresis_proc(trailing_context_count=5)
        proc.channels[2] = Channel(
            id=2, schema_id=1, topic="/other", message_encoding="json", metadata={}
        )
        _route_message(proc, _message(0, data=b"alpha"))
        _route_message(proc, _message(10, data=b"beta"))  # enter seg 1
        # Non-target message: no duplication.
        other = _message(11, channel_id=2)
        assert _routes(proc, other) == [1]
        # Target message: duplicates into seg 0.
        target = _message(12, channel_id=1, data=b"beta")
        assert _routes(proc, target) == [1, 0]

    def test_overlapping_trailing_windows_keep_each_previous_segment(self):
        proc = _make_hysteresis_proc(trailing_context_count=2)
        assert _route_message(proc, _message(0, data=b"alpha")) == 0

        first_transition = _message(10, data=b"beta")
        assert _routes(proc, first_transition) == [1, 0]

        second_transition = _message(20, data=b"gamma")
        assert _routes(proc, second_transition) == [2, 0, 1]

        followup = _message(30, data=b"gamma")
        assert _routes(proc, followup) == [2, 1]

        exhausted = _message(40, data=b"gamma")
        assert _routes(proc, exhausted) == [2]

    def test_no_trailing_when_disabled(self):
        proc = _make_hysteresis_proc()
        _route_message(proc, _message(0, data=b"alpha"))
        _route_message(proc, _message(10, data=b"beta"))
        msg = _message(11, data=b"beta")
        assert _routes(proc, msg) == [1]

    def test_no_trailing_for_first_segment_commit(self):
        # The very first commit (UNSET → first value) should not arm a tail
        # because there is no previous segment.
        proc = _make_hysteresis_proc(trailing_context_count=10)
        _route_message(proc, _message(0, data=b"alpha"))  # first commit
        msg = _message(1, data=b"alpha")
        assert _routes(proc, msg) == [0]
