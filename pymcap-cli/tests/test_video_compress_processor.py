"""VideoCompressProcessor: CompressedImage → CompressedVideo in the pipeline.

Uses the software encoder (libx264) for determinism/portability. Verifies
per-topic count + timestamp preservation, the single-clean-channel outcome,
composition with topic-drop, and that the emitted H.264 decodes back.
"""

from __future__ import annotations

import io
from hashlib import sha256
from typing import TYPE_CHECKING

import pymcap_cli.core.processors.video_compress as video_compress
import pytest
from mcap_codec_support.video import (
    CorruptImageError,
    EncoderMode,
    VideoEncoderError,
    get_software_encoder,
)
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
from pymcap_cli.core.message_filter import TopicSelection
from pymcap_cli.core.processors.video_compress import (
    VideoCompressProcessor,
    VideoEncoderSession,
    split_decode_workers,
)
from small_mcap import McapWriter, get_summary, read_message, read_message_decoded

from tests.fixtures.image_mcap_generator import (
    SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA,
    write_camera_mcap,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

_W, _H = 160, 120
_VIDEO_SCHEMA = "foxglove_msgs/msg/CompressedVideo"


def _write_cameras(path: Path, topics: list[str], n: int) -> None:
    write_camera_mcap(path, topics, n, width=_W, height=_H)


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


def test_video_processor_transcodes_only_its_selected_topics(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/CAM_FRONT/image", "/CAM_BACK/image"], n=8)

    processor = VideoCompressProcessor(
        encoder="libx264",
        topics=TopicSelection.from_patterns(include=["/CAM_FRONT/image"]),
    )
    _run(src, out, extra_processors=[processor])

    with out.open("rb") as stream:
        summary = get_summary(stream)
    assert summary is not None
    schemas_by_topic = {
        channel.topic: summary.schemas[channel.schema_id].name
        for channel in summary.channels.values()
    }
    assert schemas_by_topic == {
        "/CAM_FRONT/image": _VIDEO_SCHEMA,
        "/CAM_BACK/image": "sensor_msgs/msg/CompressedImage",
    }
    assert processor.matched_topics == frozenset({"/CAM_FRONT/image"})


def test_video_processor_transcodes_every_topic_matching_a_regex(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/CAM_FRONT/image", "/CAM_BACK/image", "/LIDAR_TOP/image"], n=4)

    processor = VideoCompressProcessor(
        encoder="libx264",
        topics=TopicSelection.from_patterns(include=[r"/CAM_.*/image"]),
    )
    _run(src, out, extra_processors=[processor])

    assert processor.matched_topics == frozenset({"/CAM_FRONT/image", "/CAM_BACK/image"})


def test_video_processor_reports_no_matched_topics_when_nothing_selected(tmp_path: Path):
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/CAM_FRONT/image"], n=2)

    processor = VideoCompressProcessor(
        encoder="libx264",
        topics=TopicSelection.from_patterns(include=["/CAM_FRNT/image"]),
    )
    _run(src, out, extra_processors=[processor])

    assert processor.matched_topics == frozenset()


def test_split_decode_workers_divides_the_pool_across_a_processor_chain() -> None:
    single = video_compress._decode_pool_size()

    assert split_decode_workers(1) is None
    assert split_decode_workers(4) == max(1, single // 4)


def test_split_decode_workers_divides_an_explicit_worker_budget() -> None:
    assert split_decode_workers(1, worker_budget=6) == 6
    assert split_decode_workers(4, worker_budget=6) == 1


def test_split_decode_workers_rejects_non_positive_processor_count() -> None:
    with pytest.raises(ValueError, match="processor_count must be positive"):
        split_decode_workers(0)


def test_video_processor_rejects_non_positive_decode_workers() -> None:
    with pytest.raises(ValueError, match="decode_workers must be positive"):
        VideoCompressProcessor(decode_workers=0)


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


class _BufferedImageEncoder:
    def __init__(self) -> None:
        self.config = EncoderConfig(width=_W, height=_H, codec_name=get_software_encoder("h264"))
        self._pending: bytes | None = None

    def encode(self, frame: bytes) -> bytes | None:
        previous, self._pending = self._pending, frame
        return previous

    def flush_packets(self) -> list[bytes]:
        return [self._pending] if self._pending is not None else []

    def close(self) -> None:
        pass


class _CorruptImageBackend:
    label = "fake"

    def __init__(self, corrupt_frames: set[int], *, prefetch_supported: bool) -> None:
        self.prefetch_supported = prefetch_supported
        self._corrupt_frames = corrupt_frames

    def resolve_encoder(self, _codec: str) -> str:
        return get_software_encoder("h264")

    def get_pix_fmt(self, _topic: str) -> str | None:
        return None

    def decode_image(self, dm, _schema_name: str) -> tuple[bytes, int, int]:
        index = dm.decoded_message.header.stamp.sec
        if index in self._corrupt_frames:
            raise CorruptImageError("invalid JPEG entropy data")
        return f"packet-{index}".encode(), _W, _H

    def create_encoder(self, *_args, **_kwargs) -> _BufferedImageEncoder:
        return _BufferedImageEncoder()


def _write_timestamped_camera(path: Path, n: int, *, images: list[bytes] | None = None) -> None:
    output = io.BytesIO()
    writer = McapWriter(output, encoder_factory=ROS2EncoderFactory())
    writer.start(profile="ros2")
    writer.add_schema(
        1,
        "sensor_msgs/msg/CompressedImage",
        "ros2msg",
        SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode(),
    )
    writer.add_channel(1, "/CAM_FRONT/image", "cdr", 1)
    for index in range(n):
        writer.add_message_encode(
            1,
            1_000_000 + index * 100,
            {
                "header": {
                    "stamp": {"sec": index, "nanosec": index + 7},
                    "frame_id": f"camera-{index}",
                },
                "format": "jpeg",
                "data": images[index] if images is not None else f"source-{index}".encode(),
            },
            2_000_000 + index * 100,
        )
    writer.finish()
    path.write_bytes(output.getvalue())


@pytest.mark.parametrize("prefetch_supported", [False, True])
@pytest.mark.parametrize("corrupt_frames", [{1}, {0}, {2}, {0, 1}, {0, 1, 2}])
@pytest.mark.parametrize("max_inflight", [1, 128])
def test_video_processor_corrupt_frames_preserve_buffered_frame_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefetch_supported: bool,
    corrupt_frames: set[int],
    max_inflight: int,
) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_timestamped_camera(src, n=3)
    monkeypatch.setattr(
        video_compress,
        "create_video_compression_backend",
        lambda *_args, **_kwargs: _CorruptImageBackend(
            corrupt_frames, prefetch_supported=prefetch_supported
        ),
    )
    monkeypatch.setattr(video_compress, "_DEFAULT_INFLIGHT_PER_TOPIC", max_inflight)

    processor = VideoCompressProcessor(codec="h264")
    _run(src, out, extra_processors=[processor])

    with out.open("rb") as stream:
        messages = list(read_message_decoded(stream, decoder_factories=[DecoderFactory()]))
    assert [
        (
            message.message.log_time,
            message.message.publish_time,
            message.decoded_message.timestamp.sec,
            message.decoded_message.timestamp.nanosec,
            message.decoded_message.frame_id,
            bytes(message.decoded_message.data),
        )
        for message in messages
    ] == [
        (
            1_000_000 + index * 100,
            2_000_000 + index * 100,
            index,
            index + 7,
            f"camera-{index}",
            f"packet-{index}".encode(),
        )
        for index in range(3)
        if index not in corrupt_frames
    ]
    assert isinstance(processor.dropped_frames, tuple)
    assert [
        (
            dropped.topic,
            dropped.log_time,
            dropped.publish_time,
            dropped.stamp_sec,
            dropped.stamp_nanosec,
            dropped.frame_id,
            dropped.reason,
            dropped.sha256,
        )
        for dropped in processor.dropped_frames
    ] == [
        (
            "/CAM_FRONT/image",
            1_000_000 + index * 100,
            2_000_000 + index * 100,
            index,
            index + 7,
            f"camera-{index}",
            "invalid JPEG entropy data",
            sha256(f"source-{index}".encode()).hexdigest(),
        )
        for index in sorted(corrupt_frames)
    ]


@pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
@pytest.mark.parametrize("damage", ["entropy", "quantization"])
def test_video_processor_corrupt_jpeg_between_healthy_images_is_audited_and_dropped(
    tmp_path: Path, backend: EncoderMode, damage: str
) -> None:
    av = pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    if backend is EncoderMode.FFMPEG_CLI and (
        find_ffmpeg() is None or not check_encoder_cli("libx264")
    ):
        pytest.skip("ffmpeg/libx264 not available")
    images = [_pillow_frame(image_module, _W, _H, index, "JPEG") for index in range(3)]
    if damage == "entropy":
        scan = images[1].index(b"\xff\xda")
        entropy_start = scan + 2 + int.from_bytes(images[1][scan + 2 : scan + 4], "big")
        images[1] = images[1][: entropy_start + 4] + b"\xff\xd9"
    else:
        table = images[1].index(b"\xff\xdb")
        end = table + 2 + int.from_bytes(images[1][table + 2 : table + 4], "big")
        images[1] = images[1][:table] + images[1][end:]
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_timestamped_camera(src, n=3, images=images)
    processor = VideoCompressProcessor(encoder="libx264", backend=backend)

    _run(src, out, extra_processors=[processor])

    with out.open("rb") as stream:
        messages = list(read_message_decoded(stream, decoder_factories=[DecoderFactory()]))
    assert [message.message.log_time for message in messages] == [1_000_000, 1_000_200]
    assert [message.message.publish_time for message in messages] == [2_000_000, 2_000_200]
    assert [message.decoded_message.timestamp.sec for message in messages] == [0, 2]
    assert [message.decoded_message.timestamp.nanosec for message in messages] == [7, 9]
    assert [message.decoded_message.frame_id for message in messages] == ["camera-0", "camera-2"]
    assert len(processor.dropped_frames) == 1
    assert processor.dropped_frames[0].sha256 == sha256(images[1]).hexdigest()
    assert processor.dropped_frames[0].log_time == 1_000_100
    assert processor.dropped_frames[0].reason
    decoder = av.CodecContext.create("h264", "r")
    decoded_frames = sum(
        len(decoder.decode(av.Packet(bytes(message.decoded_message.data)))) for message in messages
    )
    decoded_frames += len(decoder.decode(None))
    assert decoded_frames == 2


class _LostFlushEncoder(_BufferedImageEncoder):
    def flush_packets(self) -> list[bytes]:
        return []


class _FailingFlushEncoder(_BufferedImageEncoder):
    def flush_packets(self) -> list[bytes]:
        raise VideoEncoderError("failed to drain encoder")


class _ExtraFlushEncoder(_BufferedImageEncoder):
    def flush_packets(self) -> list[bytes]:
        return [*super().flush_packets(), b"unexpected-extra-packet"]


class _FailingEncodeEncoder(_BufferedImageEncoder):
    def encode(self, _frame: bytes) -> bytes | None:
        raise VideoEncoderError("encoder rejected frame")


@pytest.mark.parametrize(
    "encoder_class",
    [_LostFlushEncoder, _FailingFlushEncoder, _ExtraFlushEncoder, _FailingEncodeEncoder],
)
def test_video_processor_unexplained_encoder_loss_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoder_class
) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_timestamped_camera(src, n=3)
    backend = _CorruptImageBackend(set(), prefetch_supported=False)
    monkeypatch.setattr(backend, "create_encoder", lambda *_args, **_kwargs: encoder_class())
    monkeypatch.setattr(
        video_compress, "create_video_compression_backend", lambda *_args, **_kwargs: backend
    )

    with pytest.raises(VideoEncoderError):
        _run(src, out, extra_processors=[VideoCompressProcessor(codec="h264")])


@pytest.mark.parametrize("prefetch_supported", [False, True])
@pytest.mark.parametrize("error_index", [0, 1])
@pytest.mark.parametrize("error_type", [RuntimeError, VideoEncoderError])
def test_video_processor_unexpected_decode_error_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefetch_supported: bool,
    error_index: int,
    error_type: type[Exception],
) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_timestamped_camera(src, n=3)
    backend = _CorruptImageBackend(set(), prefetch_supported=prefetch_supported)
    decode_image = backend.decode_image

    def decode_with_failure(dm, schema_name: str) -> tuple[bytes, int, int]:
        if dm.decoded_message.header.stamp.sec == error_index:
            raise error_type("decoder dependency unavailable")
        return decode_image(dm, schema_name)

    monkeypatch.setattr(backend, "decode_image", decode_with_failure)
    monkeypatch.setattr(
        video_compress, "create_video_compression_backend", lambda *_args, **_kwargs: backend
    )

    with pytest.raises(error_type, match="decoder dependency unavailable"):
        _run(src, out, extra_processors=[VideoCompressProcessor(codec="h264")])


def test_video_session_does_not_fallback_while_aborting():
    backend = _FallbackBackend(buffer_depth=0)
    session = VideoEncoderSession(
        backend=backend,
        encoder_name="hw_enc",
        codec="h264",
        quality=28,
        topic="/cam/front",
        scale=None,
    )
    session.ensure_encoder(_W, _H)

    session.abort()
    outcome = session.encode_with_fallback(b"frame", _W, _H)

    assert outcome.failed
    assert not outcome.swapped


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


def test_video_processor_hw_crash_with_buffered_frames_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A hardware crash cannot silently discard accepted, buffered images."""
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front"], n=5)

    monkeypatch.setattr(
        video_compress,
        "create_video_compression_backend",
        lambda *_a, **_k: _FallbackBackend(buffer_depth=2),
    )
    with pytest.raises(VideoEncoderError):
        _run(src, out, extra_processors=[VideoCompressProcessor(codec="h264")])


def test_video_processor_fallback_without_buffered_frames_preserves_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_cameras(src, ["/cam/front"], n=5)

    monkeypatch.setattr(
        video_compress,
        "create_video_compression_backend",
        lambda *_args, **_kwargs: _FallbackBackend(buffer_depth=0),
    )
    _run(src, out, extra_processors=[VideoCompressProcessor(codec="h264")])

    with out.open("rb") as f:
        times = [m.log_time for _s, c, m in read_message(f) if c.topic == "/cam/front"]

    # _write_cameras stamps frame i (single topic, cid=1) at log_time i*step + 1.
    step = 1_000_000
    assert times == [index * step + 1 for index in range(5)]
