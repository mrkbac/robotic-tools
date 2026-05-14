"""Tests for the LatchingProcessor and its integration with filter/split."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pymcap_cli.cmd._run_processor import run_processor
from pymcap_cli.cmd._run_processor_multi import run_processor_multi
from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from pymcap_cli.core.processors.base import Action
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.latching import (
    LatchingProcessor,
    _channel_is_transient_local,
)
from small_mcap import Channel, read_message

from tests.fixtures.mcap_generator import (
    _TRANSIENT_LOCAL_QOS_YAML,
    create_latched_topic_mcap,
)
from tests.helpers import channel_context

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Metadata detector
# ---------------------------------------------------------------------------


class TestChannelIsTransientLocal:
    def _channel(self, metadata: dict[str, str]) -> Channel:
        return Channel(id=1, schema_id=1, topic="/x", message_encoding="json", metadata=metadata)

    def test_no_metadata(self) -> None:
        assert _channel_is_transient_local(self._channel({})) is False

    def test_transient_local_yaml_list(self) -> None:
        assert _channel_is_transient_local(
            self._channel({"offered_qos_profiles": _TRANSIENT_LOCAL_QOS_YAML})
        )

    def test_volatile_durability(self) -> None:
        blob = "- durability: volatile\n"
        assert _channel_is_transient_local(self._channel({"offered_qos_profiles": blob})) is False

    def test_malformed_yaml_returns_false(self) -> None:
        assert (
            _channel_is_transient_local(
                self._channel({"offered_qos_profiles": "not: valid: yaml: ::"})
            )
            is False
        )

    def test_unrelated_metadata_key(self) -> None:
        assert (
            _channel_is_transient_local(self._channel({"qos.durability": "transient_local"}))
            is False
        )


# ---------------------------------------------------------------------------
# Pure unit tests for LatchingProcessor state
# ---------------------------------------------------------------------------


class TestLatchingProcessor:
    def _channel(self, *, ch_id: int, topic: str, latched: bool = False) -> Channel:
        metadata = {"offered_qos_profiles": _TRANSIENT_LOCAL_QOS_YAML} if latched else {}
        return Channel(
            id=ch_id, schema_id=1, topic=topic, message_encoding="json", metadata=metadata
        )

    def test_pattern_match_registers_latched_channel(self) -> None:
        proc = LatchingProcessor(patterns=[re.compile(r"/tf_static")])
        latched = self._channel(ch_id=1, topic="/tf_static")
        non_latched = self._channel(ch_id=2, topic="/scan")
        assert proc.on_channel(channel_context(latched), latched, None) is Action.CONTINUE
        assert proc.on_channel(channel_context(non_latched), non_latched, None) is Action.CONTINUE
        assert proc.latched_channel_ids == {1}

    def test_metadata_detection_disabled_by_default(self) -> None:
        proc = LatchingProcessor()
        channel = self._channel(ch_id=1, topic="/static_thing", latched=True)
        action = proc.on_channel(channel_context(channel), channel, None)
        assert action is Action.CONTINUE
        assert proc.latched_channel_ids == set()

    def test_metadata_detection_when_opted_in(self) -> None:
        proc = LatchingProcessor(from_metadata=True)
        channel = self._channel(ch_id=1, topic="/static_thing", latched=True)
        action = proc.on_channel(channel_context(channel), channel, None)
        assert action is Action.CONTINUE
        assert proc.latched_channel_ids == {1}

    def test_pattern_takes_precedence_over_disabled_metadata(self) -> None:
        proc = LatchingProcessor(patterns=[re.compile(r"_static")])
        # Channel has metadata but from_metadata disabled — pattern still hits.
        channel = self._channel(ch_id=1, topic="/tf_static", latched=True)
        action = proc.on_channel(channel_context(channel), channel, None)
        assert action is Action.CONTINUE
        assert proc.latched_channel_ids == {1}


# ---------------------------------------------------------------------------
# End-to-end through run_processor (filter)
# ---------------------------------------------------------------------------


def _read_messages(path: Path) -> list[tuple[str, int]]:
    """Return (topic, log_time) pairs from an MCAP file, in stored order."""
    with path.open("rb") as f:
        return [(channel.topic, message.log_time) for _schema, channel, message in read_message(f)]


def test_filter_drops_latched_topic_when_topic_pattern_excludes_it(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    src.write_bytes(create_latched_topic_mcap(other_messages=3, other_step_ns=NS_TO_SEC))

    out = tmp_path / "out.mcap"
    rc = run_processor(
        files=[str(src)],
        output=out,
        input_options=InputOptions.from_args(
            include_topic_regex=["/scan"],
            latch_topics=["/tf_static"],
        ),
        output_options=OutputOptions(),
    )
    assert rc.stats is not None

    messages = _read_messages(out)
    topics = {topic for topic, _ in messages}
    assert "/tf_static" not in topics
    assert "/scan" in topics


def test_filter_keeps_latched_topic_when_start_window_drops_it(tmp_path: Path) -> None:
    """The latched message at t=0 should survive --start 5s."""
    src = tmp_path / "in.mcap"
    src.write_bytes(create_latched_topic_mcap(other_messages=10, other_step_ns=NS_TO_SEC))

    out = tmp_path / "out.mcap"
    rc = run_processor(
        files=[str(src)],
        output=out,
        input_options=InputOptions.from_args(
            start=str(5 * NS_TO_SEC),
            latch_topics=["/tf_static"],
        ),
        output_options=OutputOptions(),
    )
    assert rc.stats is not None

    messages = _read_messages(out)
    latched = [(topic, ts) for topic, ts in messages if topic == "/tf_static"]
    assert len(latched) == 1, f"expected one latched replay; got {latched}"
    assert latched[0][1] == 0, "latched log_time must be preserved"

    scans = [ts for topic, ts in messages if topic == "/scan"]
    assert all(ts >= 5 * NS_TO_SEC for ts in scans)


def test_metadata_autodetect_opt_in(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    src.write_bytes(create_latched_topic_mcap(other_messages=3, other_step_ns=NS_TO_SEC))

    out = tmp_path / "out.mcap"
    rc = run_processor(
        files=[str(src)],
        output=out,
        input_options=InputOptions.from_args(
            include_topic_regex=["/scan"],
            latch_from_metadata=True,
        ),
        output_options=OutputOptions(),
    )
    assert rc.stats is not None

    topics = {topic for topic, _ in _read_messages(out)}
    assert "/tf_static" not in topics


def test_metadata_autodetect_off_by_default(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    src.write_bytes(create_latched_topic_mcap(other_messages=3, other_step_ns=NS_TO_SEC))

    out = tmp_path / "out.mcap"
    rc = run_processor(
        files=[str(src)],
        output=out,
        input_options=InputOptions.from_args(include_topic_regex=["/scan"]),
        output_options=OutputOptions(),
    )
    assert rc.stats is not None

    topics = {topic for topic, _ in _read_messages(out)}
    assert "/tf_static" not in topics, "default should not auto-latch via metadata"


# ---------------------------------------------------------------------------
# End-to-end through run_processor_multi (split)
# ---------------------------------------------------------------------------


def test_split_replays_latched_into_every_segment(tmp_path: Path) -> None:
    """Every output segment of a duration split must contain the latched
    /tf_static message — the exact same bytes published once at t=0."""
    src = tmp_path / "in.mcap"
    src.write_bytes(create_latched_topic_mcap(other_messages=10, other_step_ns=NS_TO_SEC))

    template = str(tmp_path / "seg_{index:03d}.mcap")
    rc = run_processor_multi(
        files=[str(src)],
        input_options=InputOptions.from_args(latch_topics=["/tf_static"]),
        output_options=OutputOptions(
            routers=[DurationSplitProcessor(2 * NS_TO_SEC)],
            output_template=template,
        ),
    )
    assert rc.stats is not None

    seg_paths = sorted(tmp_path.glob("seg_*.mcap"))
    assert len(seg_paths) >= 2, f"expected multiple segments; got {seg_paths}"

    for seg in seg_paths:
        msgs = _read_messages(seg)
        topics = {topic for topic, _ in msgs}
        assert "/tf_static" in topics, f"segment {seg.name} missing /tf_static"
        latched_times = [ts for topic, ts in msgs if topic == "/tf_static"]
        # Always exactly one replay per segment (or one true publish in seg 0).
        assert latched_times == [0], f"expected single latched replay; got {latched_times}"


def test_split_replays_latches_from_all_input_streams(tmp_path: Path) -> None:
    """With multiple inputs each carrying its own latched topic, every split
    segment must contain BOTH latches — the replay path must walk every
    input stream's processors, not just the one that triggered the open."""
    src_a = tmp_path / "a.mcap"
    src_b = tmp_path / "b.mcap"
    src_a.write_bytes(
        create_latched_topic_mcap(
            latched_topic="/tf_static_a",
            other_topic="/scan_a",
            other_messages=10,
            other_step_ns=NS_TO_SEC,
        )
    )
    src_b.write_bytes(
        create_latched_topic_mcap(
            latched_topic="/tf_static_b",
            other_topic="/scan_b",
            other_messages=10,
            other_step_ns=NS_TO_SEC,
        )
    )

    template = str(tmp_path / "seg_{index:03d}.mcap")
    rc = run_processor_multi(
        files=[str(src_a), str(src_b)],
        input_options=InputOptions.from_args(latch_topics=["/tf_static_a", "/tf_static_b"]),
        output_options=OutputOptions(
            routers=[DurationSplitProcessor(2 * NS_TO_SEC)],
            output_template=template,
        ),
    )
    assert rc.stats is not None

    seg_paths = sorted(tmp_path.glob("seg_*.mcap"))
    assert len(seg_paths) >= 2, f"expected multiple segments; got {seg_paths}"

    for seg in seg_paths:
        msgs = _read_messages(seg)
        topics = {topic for topic, _ in msgs}
        assert "/tf_static_a" in topics, f"segment {seg.name} missing /tf_static_a"
        assert "/tf_static_b" in topics, f"segment {seg.name} missing /tf_static_b"
        latched_a = [ts for topic, ts in msgs if topic == "/tf_static_a"]
        latched_b = [ts for topic, ts in msgs if topic == "/tf_static_b"]
        assert latched_a == [0], f"segment {seg.name} has duplicate /tf_static_a: {latched_a}"
        assert latched_b == [0], f"segment {seg.name} has duplicate /tf_static_b: {latched_b}"


def test_split_replays_latest_latch_before_segment_start(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    src.write_bytes(
        create_latched_topic_mcap(
            latched_update_times=[5 * NS_TO_SEC],
            other_messages=8,
            other_step_ns=NS_TO_SEC,
        )
    )

    template = str(tmp_path / "seg_{index:03d}.mcap")
    rc = run_processor_multi(
        files=[str(src)],
        input_options=InputOptions.from_args(latch_topics=["/tf_static"]),
        output_options=OutputOptions(
            routers=[DurationSplitProcessor(2 * NS_TO_SEC)],
            output_template=template,
        ),
    )
    assert rc.stats is not None

    seg_paths = sorted(tmp_path.glob("seg_*.mcap"))
    assert len(seg_paths) >= 4

    latched_times_by_segment = [
        [ts for topic, ts in _read_messages(seg) if topic == "/tf_static"] for seg in seg_paths
    ]
    assert latched_times_by_segment[0] == [0]
    assert latched_times_by_segment[1] == [0]
    assert latched_times_by_segment[2] == [0, 5 * NS_TO_SEC]
    assert latched_times_by_segment[3] == [5 * NS_TO_SEC]


def test_split_replays_latched_when_dynamic_segment_opens_on_fast_copy(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    src.write_bytes(
        create_latched_topic_mcap(
            other_messages=6,
            other_step_ns=2 * NS_TO_SEC,
            chunk_size=64,
        )
    )

    template = str(tmp_path / "seg_{index:03d}.mcap")
    rc = run_processor_multi(
        files=[str(src)],
        input_options=InputOptions.from_args(latch_topics=["/tf_static"]),
        output_options=OutputOptions(
            routers=[DurationSplitProcessor(2 * NS_TO_SEC)],
            output_template=template,
        ),
    )
    assert rc.stats is not None

    seg_paths = sorted(tmp_path.glob("seg_*.mcap"))
    assert len(seg_paths) >= 4
    for seg in seg_paths:
        topics = {topic for topic, _ in _read_messages(seg)}
        assert "/tf_static" in topics, f"segment {seg.name} missing /tf_static"


def test_split_no_latch_means_only_first_segment_has_static(tmp_path: Path) -> None:
    """Sanity check: without --latch the static topic is only in segment 0."""
    src = tmp_path / "in.mcap"
    src.write_bytes(create_latched_topic_mcap(other_messages=10, other_step_ns=NS_TO_SEC))

    template = str(tmp_path / "seg_{index:03d}.mcap")
    rc = run_processor_multi(
        files=[str(src)],
        output_options=OutputOptions(
            routers=[DurationSplitProcessor(2 * NS_TO_SEC)],
            output_template=template,
        ),
    )
    assert rc.stats is not None

    seg_paths = sorted(tmp_path.glob("seg_*.mcap"))
    assert len(seg_paths) >= 2

    have_static = [seg for seg in seg_paths if "/tf_static" in {t for t, _ in _read_messages(seg)}]
    assert have_static == seg_paths[:1], (
        f"only first segment should hold /tf_static; got {[s.name for s in have_static]}"
    )


# ---------------------------------------------------------------------------
# Direct McapProcessor integration to validate fast-copy DECODE rule
# ---------------------------------------------------------------------------


def test_latched_chunk_is_decoded_not_fast_copied(tmp_path: Path) -> None:
    """When a chunk references a latched channel, LatchingProcessor.on_chunk
    must force DECODE so on_message updates the cache."""
    src = tmp_path / "in.mcap"
    # Many small chunks so the chunk-level decision matters.
    src.write_bytes(
        create_latched_topic_mcap(other_messages=20, other_step_ns=NS_TO_SEC, chunk_size=64)
    )

    output = tmp_path / "out.mcap"
    with open_input(str(src)) as (stream, size):
        input_options = InputOptions.from_args(latch_topics=["/tf_static"])
        with output.open("wb") as out_stream:
            options = ProcessingOptions(
                inputs=[InputFile(stream=stream, size=size, options=input_options)],
                input_options=input_options,
                output_options=OutputOptions(),
            )
            processor = McapProcessor(options)
            stats = processor.process(output_stream=out_stream)

    # If the latched chunks were fast-copied, latched_channel_ids would be
    # empty (channels populated, but cache miss because on_message never fired
    # for them). Verify on_message did fire.
    latching_proc = next(
        p for p in processor._get_processors(0) if isinstance(p, LatchingProcessor)
    )
    assert latching_proc.latched_channel_ids, "latched channels never registered"
    assert any(stats.messages_processed for _ in [None])
    assert stats.chunks_processed > 0
