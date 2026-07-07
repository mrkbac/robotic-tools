"""VideoCompressProcessor: CompressedImage → CompressedVideo in the pipeline.

Uses the software encoder (libx264) for determinism/portability. Verifies
per-topic count + timestamp preservation, the single-clean-channel outcome,
composition with topic-drop, and that the emitted H.264 decodes back.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd._run_processor import run_processor
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.core.processors.video_compress import VideoCompressProcessor
from small_mcap import McapWriter, get_summary, read_message, read_message_decoded

from tests.fixtures.image_mcap_generator import (
    SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA,
    create_jpeg_frame,
)

if TYPE_CHECKING:
    from pathlib import Path

_W, _H = 160, 120
_VIDEO_SCHEMA = "foxglove_msgs/msg/CompressedVideo"


def _write_cameras(path: Path, topics: list[str], n: int) -> None:
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=1 << 20, encoder_factory=ROS2EncoderFactory())
    writer.start(profile="ros2")
    writer.add_schema(
        1,
        "sensor_msgs/msg/CompressedImage",
        "ros2msg",
        SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode(),
    )
    for cid, topic in enumerate(topics, start=1):
        writer.add_channel(cid, topic, "cdr", 1)
    step = 1_000_000
    for i in range(n):
        for cid, _topic in enumerate(topics, start=1):
            log_time = i * step + cid
            header = {"stamp": {"sec": i, "nanosec": cid}, "frame_id": f"cam{cid}"}
            writer.add_message_encode(
                cid,
                log_time,
                {"header": header, "format": "jpeg", "data": create_jpeg_frame(_W, _H, i)},
                log_time,
            )
    writer.finish()
    path.write_bytes(buf.getvalue())


def _run(src: Path, out: Path, *, extra_processors, exclude=None) -> None:
    run_processor(
        files=[str(src)],
        output=out,
        input_options=InputOptions.from_args(
            extra_processors=extra_processors, exclude_topic_regex=exclude or []
        ),
        output_options=OutputOptions(overwrite_policy=OverwriteCollisionPolicy.OVERWRITE),
    )


def _counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("rb") as f:
        for _s, ch, _m in read_message(f):
            counts[ch.topic] = counts.get(ch.topic, 0) + 1
    return counts


def test_video_processor_transcodes_to_compressed_video(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front", "/cam/rear"], n=12)

    _run(src, out, extra_processors=[VideoCompressProcessor(encoder="libx264")])

    # Per-topic counts preserved (no frames lost to encoder buffering / flush).
    assert _counts(out) == {"/cam/front": 12, "/cam/rear": 12}

    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    for topic in ("/cam/front", "/cam/rear"):
        chans = [c for c in summary.channels.values() if c.topic == topic]
        assert len(chans) == 1  # no orphaned CompressedImage channel left behind
        assert summary.schemas[chans[0].schema_id].name == _VIDEO_SCHEMA


def test_video_processor_preserves_per_channel_timestamp_order(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front"], n=20)

    _run(src, out, extra_processors=[VideoCompressProcessor(encoder="libx264")])

    with out.open("rb") as f:
        times = [m.log_time for _s, c, m in read_message(f) if c.topic == "/cam/front"]
    assert times == sorted(times)
    assert len(times) == 20


def test_video_processor_composes_with_topic_drop(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front", "/cam/debug"], n=8)

    _run(
        src,
        out,
        extra_processors=[VideoCompressProcessor(encoder="libx264")],
        exclude=[r"/cam/debug"],
    )

    counts = _counts(out)
    assert "/cam/debug" not in counts
    assert counts == {"/cam/front": 8}


def test_video_processor_output_is_a_valid_bitstream(tmp_path: Path):
    """The emitted H.264 packets decode cleanly back to the original frame count.

    Decodes the raw bitstream directly with PyAV's h264 decoder — the true
    correctness check for the encoder output, independent of the higher-level
    decompressor's (separately buggy) MJPEG re-encode path.
    """
    av = pytest.importorskip("av")

    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front"], n=10)

    _run(src, out, extra_processors=[VideoCompressProcessor(encoder="libx264")])

    with out.open("rb") as f:
        packets = [
            bytes(m.decoded_message.data)
            for m in read_message_decoded(f, decoder_factories=[DecoderFactory()])
            if m.channel.topic == "/cam/front"
        ]
    assert len(packets) == 10

    ctx = av.CodecContext.create("h264", "r")
    decoded = 0
    for data in packets:
        for _frame in ctx.decode(av.Packet(data)):
            decoded += 1
    for _frame in ctx.decode(None):  # flush
        decoded += 1
    assert decoded == 10
