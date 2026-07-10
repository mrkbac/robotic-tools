"""Chunk layout of roscompress-style output: overlap and span bounds.

Two regressions are guarded here:

- Routing every already-compressed topic into ONE shared chunk group made the
  group's chunks span huge, mutually-overlapping time ranges (the video encoder
  emits each topic with a per-topic *frame-count* lag, so heterogeneous-fps
  topics arrive badly out of log-time order). Per-topic grouping collapses the
  overlap to the benign "one active chunk per concurrent topic".
- A max chunk time-span keeps low-byte-rate groups from accumulating a single
  4 MB chunk that covers a very wide time range.

The overlap oracle is ``_calculate_chunk_overlaps`` (used by the ``info``
command), which is independent of the code under test.
"""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING

import pytest
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd._run_processor import run_processor
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.core.processors.chunk_groupers import (
    PerChannelGrouper,
    SchemaCompressionGrouper,
)
from pymcap_cli.core.processors.video_compress import VideoCompressProcessor
from pymcap_cli.types.info_data import _calculate_chunk_overlaps
from small_mcap import McapWriter, get_summary, read_message

from tests.fixtures.image_mcap_generator import (
    SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA,
    create_jpeg_frame,
)

if TYPE_CHECKING:
    from pathlib import Path

_COMPRESSED_VIDEO_PATTERN = re.compile("CompressedVideo")

_W, _H = 64, 48


def _write_cameras_multirate(path: Path, topics: dict[str, tuple[int, int]]) -> None:
    """Write image topics, each with its own (step_ns, count).

    Messages are written in global log-time order (a valid input MCAP); the
    per-topic frame-count lag inside VideoCompressProcessor is what later
    reorders them relative to each other.
    """
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=1 << 20, encoder_factory=ROS2EncoderFactory())
    writer.start(profile="ros2")
    writer.add_schema(
        1,
        "sensor_msgs/msg/CompressedImage",
        "ros2msg",
        SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode(),
    )
    channel_ids = {topic: cid for cid, topic in enumerate(topics, start=1)}
    for topic, cid in channel_ids.items():
        writer.add_channel(cid, topic, "cdr", 1)

    records: list[tuple[int, int, int]] = [
        (i * step_ns + cid, cid, i)  # (log_time, channel_id, frame_idx)
        for topic, (step_ns, count) in topics.items()
        for i in range(count)
        for cid in (channel_ids[topic],)
    ]
    records.sort()

    for log_time, cid, frame_idx in records:
        header = {"stamp": {"sec": frame_idx, "nanosec": cid}, "frame_id": f"cam{cid}"}
        writer.add_message_encode(
            cid,
            log_time,
            {"header": header, "format": "jpeg", "data": create_jpeg_frame(_W, _H, frame_idx)},
            log_time,
        )
    writer.finish()
    path.write_bytes(buf.getvalue())


def _overlaps(path: Path) -> int:
    with path.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    max_concurrent, _bytes = _calculate_chunk_overlaps(summary.chunk_indexes)
    return max_concurrent


def _per_channel_times(path: Path) -> dict[str, list[int]]:
    times: dict[str, list[int]] = {}
    with path.open("rb") as f:
        for _s, ch, m in read_message(f):
            times.setdefault(ch.topic, []).append(m.log_time)
    return times


@pytest.mark.parametrize("per_channel", [False, True])
def test_roscompress_compressed_group_overlap(tmp_path: Path, per_channel: bool):
    pytest.importorskip("av")
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    # Two cameras at different frame rates, both exceeding _INFLIGHT_PER_TOPIC
    # (128) and overlapping in input time, so the encoder emits them badly out
    # of log-time order into the shared group.
    _write_cameras_multirate(
        src,
        {"/cam_fast": (10_000_000, 600), "/cam_slow": (20_000_000, 300)},
    )

    run_processor(
        files=[str(src)],
        output=out,
        input_options=InputOptions.from_args(
            extra_processors=[VideoCompressProcessor(encoder="libx264")]
        ),
        output_options=OutputOptions(
            overwrite_policy=OverwriteCollisionPolicy.OVERWRITE,
            output_processors=[
                SchemaCompressionGrouper([_COMPRESSED_VIDEO_PATTERN], per_channel=per_channel)
            ],
            chunk_size=8192,
        ),
    )

    # Message count preserved.
    times = _per_channel_times(out)
    assert {t: len(v) for t, v in times.items()} == {"/cam_fast": 600, "/cam_slow": 300}
    # Per-topic output stays monotonic.
    for topic_times in times.values():
        assert topic_times == sorted(topic_times)

    max_concurrent = _overlaps(out)
    if per_channel:
        # One active chunk per genuinely-concurrent topic; no deep overlap.
        assert max_concurrent <= 2
    else:
        # The shared arrival-ordered group overlaps far more than 2 deep.
        assert max_concurrent > 2


def _write_low_rate_topic(path: Path, *, count: int, step_ns: int, payload: int) -> None:
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=1 << 20)
    writer.start()
    writer.add_schema(1, "Schema", "raw", b"")
    writer.add_channel(1, "/slow", "raw", 1)
    for i in range(count):
        writer.add_message(
            channel_id=1,
            log_time=i * step_ns,
            publish_time=i * step_ns,
            data=b"\x00" * payload,
        )
    writer.finish()
    path.write_bytes(buf.getvalue())


def test_max_chunk_span_bounds_chunk_duration(tmp_path: Path):
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    # 200 messages 1 s apart = 200 s of data, tiny payloads that never approach
    # the 1 MB chunk size. Without a span cap this becomes one giant chunk.
    cap_ns = 10 * 1_000_000_000
    _write_low_rate_topic(src, count=200, step_ns=1_000_000_000, payload=8)

    run_processor(
        files=[str(src)],
        output=out,
        input_options=InputOptions.from_args(),
        output_options=OutputOptions(
            overwrite_policy=OverwriteCollisionPolicy.OVERWRITE,
            output_processors=[PerChannelGrouper()],
            chunk_size=1 << 20,
            max_chunk_span_ns=cap_ns,
        ),
    )

    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    assert len(summary.chunk_indexes) > 1  # the cap actually split the data
    for ci in summary.chunk_indexes:
        assert ci.message_end_time - ci.message_start_time <= cap_ns
