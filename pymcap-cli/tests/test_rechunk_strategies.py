"""Tests for rechunk strategy updates.

Covers:
- The default ``rechunk`` strategy is ``ALL``.
- ``--rechunk-schema-pattern`` and ``--rechunk-max-groups`` propagate from the
  CLI through to ``OutputOptions``.
- The streaming-safe routing semantics: schema patterns route after topic
  patterns; the max-groups cap is a hard upper bound on
  ``segment.chunk_groups`` per output segment.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pymcap_cli.cmd import process_cmd, rechunk_cmd
from pymcap_cli.cmd._rechunk_strategy import RechunkStrategy
from pymcap_cli.cmd._run_processor import run_processor
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.core.processors.chunk_groupers import (
    PatternGrouper,
    PerChannelGrouper,
    SchemaCompressionGrouper,
)
from small_mcap import CompressionType, McapWriter

from tests.helpers import empty_processor_result

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# CLI propagation: recorder-based tests for the new flags.
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.output_options: OutputOptions | None = None


def _patch(monkeypatch: pytest.MonkeyPatch, module, rec: _Recorder) -> None:
    def fake_run_processor(*, files, output, input_options, output_options):
        _ = files, output, input_options
        rec.output_options = output_options
        return empty_processor_result()

    monkeypatch.setattr(module, "run_processor", fake_run_processor)


def test_rechunk_default_strategy_is_all(monkeypatch: pytest.MonkeyPatch):
    rec = _Recorder()
    _patch(monkeypatch, rechunk_cmd, rec)

    rechunk_cmd.rechunk(file="in.mcap", output=Path("out.mcap"))

    assert rec.output_options is not None
    assert len(rec.output_options.output_processors) == 1
    assert isinstance(rec.output_options.output_processors[0], PerChannelGrouper)


def test_rechunk_pattern_strategy_accepts_schema_pattern_alone(monkeypatch: pytest.MonkeyPatch):
    rec = _Recorder()
    _patch(monkeypatch, rechunk_cmd, rec)

    exit_code = rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.PATTERN,
        schema_pattern=["sensor_msgs/.*Image.*"],
    )

    assert exit_code == 0
    assert rec.output_options is not None
    assert len(rec.output_options.output_processors) == 1
    grouper = rec.output_options.output_processors[0]
    assert isinstance(grouper, PatternGrouper)
    assert len(grouper.schema_patterns) == 1


def test_rechunk_pattern_strategy_rejects_no_patterns():
    exit_code = rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.PATTERN,
    )
    assert exit_code == 1


def test_rechunk_max_groups_propagates(monkeypatch: pytest.MonkeyPatch):
    rec = _Recorder()
    _patch(monkeypatch, rechunk_cmd, rec)

    rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.ALL,
        max_groups=4,
    )

    assert rec.output_options is not None
    assert rec.output_options.max_chunk_groups == 4


def test_rechunk_max_groups_rejects_zero():
    exit_code = rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.ALL,
        max_groups=0,
    )
    assert exit_code == 1


def test_rechunk_max_memory_propagates(monkeypatch: pytest.MonkeyPatch):
    rec = _Recorder()
    _patch(monkeypatch, rechunk_cmd, rec)

    rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.ALL,
        max_memory="1MB",
    )

    assert rec.output_options is not None
    assert rec.output_options.max_chunk_memory_bytes == 1_000_000


def test_rechunk_max_memory_rejects_unparsable():
    exit_code = rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.ALL,
        max_memory="not-a-size",
    )
    assert exit_code == 1


def test_rechunk_incompressible_schema_pattern_alone_with_strategy_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """--incompressible-schema-pattern activates grouping even with --strategy=none."""
    rec = _Recorder()
    _patch(monkeypatch, rechunk_cmd, rec)

    exit_code = rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.NONE,
        incompressible_schema_pattern=["CompressedVideo"],
    )

    assert exit_code == 0
    assert rec.output_options is not None
    assert len(rec.output_options.output_processors) == 1
    grouper = rec.output_options.output_processors[0]
    assert isinstance(grouper, SchemaCompressionGrouper)
    assert len(grouper.schema_patterns) == 1


def test_rechunk_incompressible_schema_pattern_composes_with_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    """--incompressible-schema-pattern adds to, rather than replaces, --strategy's grouper."""
    rec = _Recorder()
    _patch(monkeypatch, rechunk_cmd, rec)

    exit_code = rechunk_cmd.rechunk(
        file="in.mcap",
        output=Path("out.mcap"),
        strategy=RechunkStrategy.ALL,
        incompressible_schema_pattern=["CompressedVideo"],
    )

    assert exit_code == 0
    assert rec.output_options is not None
    assert len(rec.output_options.output_processors) == 2
    assert isinstance(rec.output_options.output_processors[0], PerChannelGrouper)
    assert isinstance(rec.output_options.output_processors[1], SchemaCompressionGrouper)


def test_process_propagates_schema_pattern_and_max_groups(monkeypatch: pytest.MonkeyPatch):
    rec = _Recorder()
    _patch(monkeypatch, process_cmd, rec)

    process_cmd.process(
        file=["in.mcap"],
        output=Path("out.mcap"),
        rechunk_strategy=RechunkStrategy.PATTERN,
        rechunk_schema_pattern=["sensor_msgs/.*Image.*"],
        rechunk_max_groups=3,
        rechunk_max_memory="512KB",
    )

    assert rec.output_options is not None
    assert len(rec.output_options.output_processors) == 1
    grouper = rec.output_options.output_processors[0]
    assert isinstance(grouper, PatternGrouper)
    assert len(grouper.schema_patterns) == 1
    assert rec.output_options.max_chunk_groups == 3
    assert rec.output_options.max_chunk_memory_bytes == 512_000


# ---------------------------------------------------------------------------
# End-to-end: routing + max-groups cap on a real (tiny) MCAP fixture.
# ---------------------------------------------------------------------------


def _write_fixture(
    path: Path,
    channels: list[tuple[str, str]],
    *,
    messages_per_channel: int = 2,
    payload_size: int = 16,
) -> None:
    """Write a small MCAP with the given (topic, schema_name) channels.

    Each channel gets ``messages_per_channel`` messages with ``payload_size``
    bytes of payload. Schemas are deduplicated by name.
    """
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=4096)
    writer.start()

    schema_ids: dict[str, int] = {}
    next_schema = 1
    for _, schema_name in channels:
        if schema_name not in schema_ids:
            writer.add_schema(schema_id=next_schema, name=schema_name, encoding="raw", data=b"")
            schema_ids[schema_name] = next_schema
            next_schema += 1

    for idx, (topic, schema_name) in enumerate(channels, start=1):
        writer.add_channel(
            channel_id=idx,
            topic=topic,
            message_encoding="raw",
            schema_id=schema_ids[schema_name],
        )
        for t in range(messages_per_channel):
            writer.add_message(
                channel_id=idx,
                log_time=idx * 1000 + t,
                publish_time=idx * 1000 + t,
                data=b"\x00" * payload_size,
            )
    writer.finish()
    path.write_bytes(buf.getvalue())


def _run(input_path: Path, output_path: Path, **output_opts):
    return run_processor(
        files=[str(input_path)],
        output=output_path,
        input_options=InputOptions.from_args(),
        output_options=OutputOptions(
            overwrite_policy=OverwriteCollisionPolicy.OVERWRITE,
            **output_opts,
        ),
    )


class TestEndToEndRechunk:
    def test_max_groups_caps_segment_group_count(self, tmp_path: Path):
        """PerChannelGrouper with 5 channels and max_chunk_groups=2 → exactly 2 groups."""
        inp = tmp_path / "in.mcap"
        out = tmp_path / "out.mcap"
        _write_fixture(
            inp,
            [(f"/topic{i}", "Schema") for i in range(5)],
        )

        result = _run(
            inp,
            out,
            output_processors=[PerChannelGrouper()],
            max_chunk_groups=2,
        )

        assert result.processor.output_manager is not None
        segments = list(result.processor.output_manager.segments.values())
        assert len(segments) == 1
        assert len(segments[0].chunk_groups) == 2

    def test_schema_pattern_routes_matching_channels_together(self, tmp_path: Path):
        """Channels sharing a schema-pattern match collapse into one group.

        Two channels use ``ImageSchema``; one uses ``OtherSchema``. With
        ``schema_patterns=['Image']``, the two Image channels share a group;
        the other lands in the unmatched (-1) bucket.
        """
        inp = tmp_path / "in.mcap"
        out = tmp_path / "out.mcap"
        _write_fixture(
            inp,
            [
                ("/cam_a", "ImageSchema"),
                ("/cam_b", "ImageSchema"),
                ("/imu", "OtherSchema"),
            ],
        )

        result = _run(
            inp,
            out,
            output_processors=[
                PatternGrouper(
                    topic_patterns=[],
                    schema_patterns=[re.compile("Image")],
                )
            ],
        )

        assert result.processor.output_manager is not None
        segment = next(iter(result.processor.output_manager.segments.values()))
        # 2 groups total: the schema-matched pool + the unmatched bucket.
        assert len(segment.chunk_groups) == 2
        # Both image channels point at the same group; the imu channel does not.
        cam_a_group = segment.channel_to_group[1]
        cam_b_group = segment.channel_to_group[2]
        imu_group = segment.channel_to_group[3]
        assert cam_a_group is cam_b_group
        assert imu_group is not cam_a_group

    def test_max_memory_forces_premature_chunk_flushes(self, tmp_path: Path):
        """With a tight memory cap the writer produces more chunks than without.

        Sets a memory cap well below what would naturally accumulate before
        chunk_size is reached, so the cap is the trigger. The unconstrained
        run produces one chunk per group; the capped run produces several.
        """
        inp = tmp_path / "in.mcap"
        out_uncapped = tmp_path / "uncapped.mcap"
        out_capped = tmp_path / "capped.mcap"
        # 4 channels x 50 messages x 256 bytes each ~= 50KB total across builders.
        _write_fixture(
            inp,
            [(f"/topic{i}", "Schema") for i in range(4)],
            messages_per_channel=50,
            payload_size=256,
        )

        unconstrained = _run(
            inp,
            out_uncapped,
            output_processors=[PerChannelGrouper()],
        )
        capped = _run(
            inp,
            out_capped,
            output_processors=[PerChannelGrouper()],
            max_chunk_memory_bytes=1024,
        )

        # Capped run must emit strictly more chunks than the unconstrained run.
        uncapped_chunks = unconstrained.stats.writer_statistics.chunk_count
        capped_chunks = capped.stats.writer_statistics.chunk_count
        assert capped_chunks > uncapped_chunks, (
            f"memory cap should trigger premature flushes "
            f"(uncapped={uncapped_chunks}, capped={capped_chunks})"
        )
        # Message count is preserved either way.
        assert (
            capped.stats.writer_statistics.message_count
            == unconstrained.stats.writer_statistics.message_count
        )

    def test_topic_pattern_takes_precedence_over_schema_pattern(self, tmp_path: Path):
        """When both a topic pattern and a schema pattern would match a channel,
        the topic pattern wins (first in the chain)."""
        inp = tmp_path / "in.mcap"
        out = tmp_path / "out.mcap"
        _write_fixture(
            inp,
            [
                ("/cam_a", "ImageSchema"),
                ("/cam_b", "ImageSchema"),
            ],
        )

        result = _run(
            inp,
            out,
            output_processors=[
                PatternGrouper(
                    topic_patterns=[re.compile("cam_a")],  # only matches cam_a's topic
                    schema_patterns=[re.compile("Image")],  # matches both schemas
                )
            ],
        )

        assert result.processor.output_manager is not None
        segment = next(iter(result.processor.output_manager.segments.values()))
        # cam_a → topic pattern (key 0); cam_b → schema pattern (key 1).
        # Two distinct groups.
        cam_a_group = segment.channel_to_group[1]
        cam_b_group = segment.channel_to_group[2]
        assert cam_a_group is not cam_b_group

    def test_incompressible_schema_grouper_overrides_compression(self, tmp_path: Path):
        """Matching-schema channels land in a shared, uncompressed group.

        Two CompressedVideo channels + one telemetry channel. With the
        default (zstd) run compression, the video channels' group must use
        CompressionType.NONE while the telemetry channel's group keeps the
        run default.
        """
        inp = tmp_path / "in.mcap"
        out = tmp_path / "out.mcap"
        _write_fixture(
            inp,
            [
                ("/cam_a/compressed", "foxglove_msgs/msg/CompressedVideo"),
                ("/cam_b/compressed", "foxglove_msgs/msg/CompressedVideo"),
                ("/imu", "sensor_msgs/msg/Imu"),
            ],
        )

        result = _run(
            inp,
            out,
            output_processors=[SchemaCompressionGrouper([re.compile("CompressedVideo")])],
            compression="zstd",
        )

        assert result.processor.output_manager is not None
        segment = next(iter(result.processor.output_manager.segments.values()))
        cam_a_group = segment.channel_to_group[1]
        cam_b_group = segment.channel_to_group[2]
        imu_group = segment.channel_to_group[3]

        # Both video channels share one group; the imu channel gets its own.
        assert cam_a_group is cam_b_group
        assert imu_group is not cam_a_group

        assert cam_a_group.chunk_builder.compression == CompressionType.NONE
        assert imu_group.chunk_builder.compression == CompressionType.ZSTD

    def test_incompressible_schema_grouper_preserves_message_count(self, tmp_path: Path):
        """The compression split changes chunk layout, not message content."""
        inp = tmp_path / "in.mcap"
        out = tmp_path / "out.mcap"
        _write_fixture(
            inp,
            [
                ("/cam/compressed", "foxglove_msgs/msg/CompressedVideo"),
                ("/imu", "sensor_msgs/msg/Imu"),
            ],
            messages_per_channel=10,
        )

        result = _run(
            inp,
            out,
            output_processors=[SchemaCompressionGrouper([re.compile("CompressedVideo")])],
            compression="zstd",
        )

        assert result.stats.writer_statistics.message_count == 20
