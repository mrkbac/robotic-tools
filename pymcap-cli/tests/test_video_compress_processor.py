"""VideoCompressProcessor: CompressedImage → CompressedVideo in the pipeline.

Uses the software encoder (libx264) for determinism/portability. Verifies
per-topic count + timestamp preservation, the single-clean-channel outcome,
composition with topic-drop, and that the emitted H.264 decodes back.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pymcap_cli.core.processors.video_compress as video_compress
import pytest
from mcap_codec_support.video import EncoderMode, VideoEncoderError, get_software_encoder
from mcap_codec_support.video.common import EncoderConfig
from mcap_codec_support.video.ffmpeg import check_encoder_cli, find_ffmpeg
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
    from types import ModuleType

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


def _pillow_frame(
    image_module: ModuleType, width: int, height: int, frame_idx: int, image_format: str
) -> bytes:
    data = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            data[idx] = (x * 255 // width + frame_idx * 10) % 256
            data[idx + 1] = (y * 255 // height) % 256
            data[idx + 2] = (frame_idx * 20) % 256
    img = image_module.frombytes("RGB", (width, height), bytes(data))
    output = io.BytesIO()
    img.save(output, format=image_format.upper())
    return output.getvalue()


def _write_pillow_compressed_camera(path: Path, n: int, image_format: str) -> None:
    image_module = pytest.importorskip("PIL.Image")
    image_module.init()
    pil_format = image_format.upper()
    if pil_format not in image_module.SAVE:
        pytest.skip(f"Pillow build cannot write {pil_format}")

    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=1 << 20, encoder_factory=ROS2EncoderFactory())
    writer.start(profile="ros2")
    writer.add_schema(
        1,
        "sensor_msgs/msg/CompressedImage",
        "ros2msg",
        SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode(),
    )
    writer.add_channel(1, f"/cam/{image_format}", "cdr", 1)
    step = 1_000_000
    for i in range(n):
        log_time = i * step
        header = {"stamp": {"sec": i, "nanosec": 0}, "frame_id": "cam"}
        writer.add_message_encode(
            1,
            log_time,
            {
                "header": header,
                "format": image_format,
                "data": _pillow_frame(image_module, _W, _H, i, image_format),
            },
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


@pytest.mark.skipif(
    find_ffmpeg() is None or not check_encoder_cli("libx264"),
    reason="ffmpeg/libx264 not available",
)
@pytest.mark.parametrize("image_format", ["webp", "tiff"])
def test_video_processor_ffmpeg_cli_transcodes_fallback_compressed_images(
    tmp_path: Path, image_format: str
):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_pillow_compressed_camera(src, n=4, image_format=image_format)

    _run(
        src,
        out,
        extra_processors=[
            VideoCompressProcessor(encoder="libx264", backend=EncoderMode.FFMPEG_CLI)
        ],
    )

    topic = f"/cam/{image_format}"
    assert _counts(out) == {topic: 4}
    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    chans = [c for c in summary.channels.values() if c.topic == topic]
    assert len(chans) == 1
    assert summary.schemas[chans[0].schema_id].name == _VIDEO_SCHEMA


# --------------------------------------------------------------------------
# Hardware-encoder crash → software fallback (regression for frame-loss shift)
# --------------------------------------------------------------------------


class _CrashingHwEncoder:
    """Buffers ``buffer_depth`` frames (returns None), then the process dies."""

    def __init__(self, buffer_depth: int) -> None:
        self.config = EncoderConfig(width=_W, height=_H, codec_name="hw_enc")
        self._calls = 0
        self._buffer_depth = buffer_depth
        self.closed = False

    def encode(self, _frame: object) -> bytes | None:
        self._calls += 1
        if self._calls <= self._buffer_depth:
            return None  # frame accepted but held in the encoder's buffer
        raise VideoEncoderError("hardware encoder crashed mid-stream")

    def flush_packets(self) -> list[bytes]:
        raise VideoEncoderError("encoder is dead")  # a crashed encoder cannot flush

    def close(self) -> None:
        self.closed = True


class _StubSoftwareEncoder:
    def __init__(self) -> None:
        self.config = EncoderConfig(width=_W, height=_H, codec_name=get_software_encoder("h264"))

    def encode(self, _frame: object) -> bytes:
        return b"pkt"

    def flush_packets(self) -> list[bytes]:
        return []

    def close(self) -> None:
        pass


class _FallbackBackend:
    """Fake backend: a hardware encoder that crashes, then a software one."""

    label = "fake"
    prefetch_supported = False

    def __init__(self, buffer_depth: int) -> None:
        self._buffer_depth = buffer_depth

    def resolve_encoder(self, _codec: str) -> str:
        return "hw_enc"

    def get_pix_fmt(self, _topic: str) -> str | None:
        return None

    def decode_image(self, _dm: object, _schema_name: str) -> tuple[bytes, int, int]:
        return b"frame", _W, _H

    def create_encoder(
        self, _w: int, _h: int, codec_name: str, _quality: int, **_kwargs: object
    ) -> object:
        if codec_name == "hw_enc":
            return _CrashingHwEncoder(self._buffer_depth)
        return _StubSoftwareEncoder()


class _CreateFailsHwBackend:
    """Fake backend whose hardware encoder cannot be created (probes available,

    fails to open — like ``h264_nvenc`` on a GPU-less CI host). The software
    encoder works, so the topic must still transcode to CompressedVideo.
    """

    label = "fake"
    prefetch_supported = False

    def resolve_encoder(self, _codec: str) -> str:
        return "hw_enc"

    def get_pix_fmt(self, _topic: str) -> str | None:
        return None

    def decode_image(self, _dm: object, _schema_name: str) -> tuple[bytes, int, int]:
        return b"frame", _W, _H

    def create_encoder(
        self, _w: int, _h: int, codec_name: str, _quality: int, **_kwargs: object
    ) -> object:
        if codec_name == "hw_enc":
            raise VideoEncoderError("Failed to open encoder hw_enc: Operation not permitted")
        return _StubSoftwareEncoder()


def test_video_processor_falls_back_to_software_when_hw_create_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A hardware encoder that fails to open must not silently pass frames raw.

    Regression for CI where ``h264_nvenc`` probes available but ``avcodec_open2``
    fails: the topic must fall back to the software encoder and still emit
    CompressedVideo rather than leaving the input CompressedImage untouched.
    """
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front"], n=4)

    monkeypatch.setattr(
        video_compress,
        "create_video_compression_backend",
        lambda *_a, **_k: _CreateFailsHwBackend(),
    )
    _run(src, out, extra_processors=[VideoCompressProcessor(codec="h264")])

    assert _counts(out) == {"/cam/front": 4}
    with out.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    chans = [c for c in summary.channels.values() if c.topic == "/cam/front"]
    assert len(chans) == 1
    assert summary.schemas[chans[0].schema_id].name == _VIDEO_SCHEMA


def test_video_processor_fallback_after_hw_crash_keeps_timestamps_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A crashed hardware encoder must not shift surviving packets' timestamps.

    The encoder accepts (buffers) 2 frames, then dies. Those 2 frames are
    unrecoverable, but the remaining 3 must keep THEIR OWN log_times — before
    the fix their metadata was reused for the wrong frames, shifting every
    output timestamp by the buffer depth.
    """
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front"], n=5)

    monkeypatch.setattr(
        video_compress,
        "create_video_compression_backend",
        lambda *_a, **_k: _FallbackBackend(buffer_depth=2),
    )
    _run(src, out, extra_processors=[VideoCompressProcessor(codec="h264")])

    with out.open("rb") as f:
        times = [m.log_time for _s, c, m in read_message(f) if c.topic == "/cam/front"]

    # _write_cameras stamps frame i (single topic, cid=1) at log_time i*step + 1.
    step = 1_000_000
    assert times == [2 * step + 1, 3 * step + 1, 4 * step + 1]
