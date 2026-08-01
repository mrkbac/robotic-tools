"""Hermetic coverage for PyAV probing, image, and lifecycle edges."""

from types import SimpleNamespace

import av
import mcap_codec_support.video.pyav as pyav_module
import pytest
from mcap_codec_support.video.common import DecompressedFrame, VideoEncoderError


class _FakeContext:
    def __init__(self, codec: str, direction: str) -> None:
        self.codec = codec
        self.direction = direction
        self.width = 0
        self.height = 0
        self.pix_fmt = "yuv420p"
        self.encode_calls = 0
        self.decode_calls = 0
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def encode(self, _frame):
        self.encode_calls += 1
        return [b"packet"]

    def decode(self, _packet):
        self.decode_calls += 1
        return []


def test_test_encoder_reports_available_context(monkeypatch) -> None:
    def create(name: str, direction: str) -> _FakeContext:
        return _FakeContext(name, direction)

    monkeypatch.setattr(
        pyav_module.av.CodecContext,
        "create",
        create,
    )

    assert pyav_module.test_encoder("libx264") is True


@pytest.mark.parametrize("error", [ValueError("bad"), av.error.FFmpegError(1, "bad")])
def test_test_encoder_reports_creation_errors(monkeypatch, error) -> None:
    def create(*_args):
        raise error

    monkeypatch.setattr(pyav_module.av.CodecContext, "create", create)

    assert pyav_module.test_encoder("broken") is False


def test_pyav_resolution_wrappers_delegate(monkeypatch) -> None:
    calls = []

    def resolve(codec, *, test_fn, use_hardware):
        calls.append((codec, test_fn, use_hardware))
        return "libx264"

    monkeypatch.setattr(pyav_module, "_resolve_encoder", resolve)
    assert pyav_module.resolve_encoder("h264", use_hardware=False) == "libx264"
    assert calls[0][0] == "h264"
    assert calls[0][2] is False

    backend_calls = []

    def resolve_backend(codec, backend, *, test_fn):
        backend_calls.append((codec, backend, test_fn))
        return "libx264"

    monkeypatch.setattr(pyav_module, "_resolve_encoder_for_backend", resolve_backend)
    assert pyav_module.resolve_encoder_for_backend("h264", "software") == "libx264"
    assert backend_calls[0][:2] == ("h264", "software")


def test_image_decoder_contexts_are_thread_local_and_reused(monkeypatch) -> None:
    created = []

    def create(name: str, direction: str):
        context = _FakeContext(name, direction)
        created.append(context)
        return context

    monkeypatch.setattr(pyav_module.av.CodecContext, "create", create)
    monkeypatch.setattr(pyav_module._decoder_local, "mjpeg_ctx", None, raising=False)
    monkeypatch.setattr(pyav_module._decoder_local, "png_ctx", None, raising=False)

    mjpeg = pyav_module._get_mjpeg_ctx()
    assert pyav_module._get_mjpeg_ctx() is mjpeg
    png = pyav_module._get_png_ctx()
    assert pyav_module._get_png_ctx() is png
    assert [context.codec for context in created] == ["mjpeg", "png"]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\xff\xd8payload", "mjpeg"),
        (b"\x89PNG\r\n\x1a\npayload", "png"),
        (b"not an image", "unknown"),
    ],
)
def test_detect_image_format(data: bytes, expected: str) -> None:
    assert pyav_module._detect_image_format(data) == expected


def test_decode_via_container_returns_first_frame(monkeypatch) -> None:
    frame = object()

    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, video: int):
            assert video == 0
            return iter([frame])

    monkeypatch.setattr(pyav_module.av, "open", lambda _data: Container())

    assert pyav_module._decode_via_container(b"container") is frame


def test_decode_via_container_wraps_ffmpeg_error(monkeypatch) -> None:
    def open_container(_data):
        raise av.error.InvalidDataError(1, "bad input")

    monkeypatch.setattr(pyav_module.av, "open", open_container)

    with pytest.raises(VideoEncoderError, match="Failed to decode compressed image"):
        pyav_module._decode_via_container(b"bad")


def test_decode_via_container_rejects_empty_container(monkeypatch) -> None:
    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, video: int):
            del video
            return iter(())

    monkeypatch.setattr(pyav_module.av, "open", lambda _data: Container())

    with pytest.raises(VideoEncoderError, match="Decoder produced no frames"):
        pyav_module._decode_via_container(b"empty")


def test_decode_compressed_frame_uses_jpeg_context(monkeypatch) -> None:
    frame = object()
    context = _FakeContext("mjpeg", "r")
    context.decode = lambda _packet: [frame]
    monkeypatch.setattr(pyav_module, "_get_mjpeg_ctx", lambda: context)

    assert pyav_module.decode_compressed_frame(b"\xff\xd8jpeg") is frame


def test_decode_compressed_frame_uses_png_context(monkeypatch) -> None:
    frame = object()
    context = _FakeContext("png", "r")
    context.decode = lambda _packet: [frame]
    monkeypatch.setattr(pyav_module, "_get_png_ctx", lambda: context)

    assert pyav_module.decode_compressed_frame(b"\x89PNG\r\n\x1a\nimage") is frame


def test_decode_compressed_frame_uses_container_for_unknown_format(monkeypatch) -> None:
    frame = object()
    monkeypatch.setattr(pyav_module, "_decode_via_container", lambda _data: frame)

    assert pyav_module.decode_compressed_frame(b"webp") is frame


def test_decode_compressed_frame_wraps_decoder_error(monkeypatch) -> None:
    context = _FakeContext("mjpeg", "r")

    def decode(_packet):
        raise av.error.InvalidDataError(1, "bad image")

    context.decode = decode
    monkeypatch.setattr(pyav_module, "_get_mjpeg_ctx", lambda: context)

    with pytest.raises(VideoEncoderError, match="Failed to decode compressed image"):
        pyav_module.decode_compressed_frame(b"\xff\xd8bad")


def test_decode_compressed_frame_rejects_decoder_without_frames(monkeypatch) -> None:
    context = _FakeContext("mjpeg", "r")
    monkeypatch.setattr(pyav_module, "_get_mjpeg_ctx", lambda: context)

    with pytest.raises(VideoEncoderError, match="Decoder produced no frames"):
        pyav_module.decode_compressed_frame(b"\xff\xd8empty")


def test_raw_image_to_frame_uses_pyav_stride_without_repacking() -> None:
    probe = pyav_module.VideoFrame(5, 3, "rgb24")
    step = probe.planes[0].line_size
    message = SimpleNamespace(
        width=5,
        height=3,
        encoding="rgb8",
        step=step,
        data=bytes(step * 3),
    )

    frame = pyav_module.raw_image_to_frame(message)

    assert frame.planes[0].line_size == step


def test_raw_image_to_frame_reports_impossibly_short_pyav_stride(monkeypatch) -> None:
    class Plane:
        line_size = 1

    class Frame:
        def __init__(self) -> None:
            self.planes = [Plane()]

    monkeypatch.setattr(pyav_module, "VideoFrame", lambda *_args: Frame())
    message = SimpleNamespace(width=2, height=1, encoding="rgb8", step=6, data=b"123456")

    with pytest.raises(VideoEncoderError, match="plane stride 1 is smaller"):
        pyav_module.raw_image_to_frame(message)


def test_video_frame_to_rgb_bytes_trims_tightly_packed_plane() -> None:
    class Plane:
        line_size = 3

        def __bytes__(self):
            return b"abcEXTRA"

    class Frame:
        width = 1
        height = 1

        def __init__(self) -> None:
            self.planes = [Plane()]

        def reformat(self, *, format: str):
            assert format == "rgb24"
            return self

    assert pyav_module.video_frame_to_rgb_bytes(Frame()) == b"abc"


def test_video_encoder_sets_hardware_bitrate(monkeypatch) -> None:
    context = _FakeContext("h264_videotoolbox", "w")
    monkeypatch.setattr(pyav_module.av.CodecContext, "create", lambda *_args: context)

    encoder = pyav_module.VideoEncoder(1920, 1080, "h264_videotoolbox")
    try:
        assert context.bit_rate == 5_000_000
    finally:
        encoder.close()


def test_video_encoder_wraps_open_error(monkeypatch) -> None:
    context = _FakeContext("broken", "w")

    def fail_open() -> None:
        raise av.error.InvalidDataError(1, "cannot open")

    context.open = fail_open
    monkeypatch.setattr(pyav_module.av.CodecContext, "create", lambda *_args: context)

    with pytest.raises(VideoEncoderError, match="Failed to open encoder broken"):
        pyav_module.VideoEncoder(16, 16, "broken")


def test_video_encoder_wraps_encode_error(monkeypatch) -> None:
    context = _FakeContext("libx264", "w")

    def fail_encode(_frame) -> None:
        raise av.error.InvalidDataError(1, "bad frame")

    context.encode = fail_encode
    monkeypatch.setattr(pyav_module.av.CodecContext, "create", lambda *_args: context)
    encoder = pyav_module.VideoEncoder(16, 16, "libx264")
    frame = SimpleNamespace(
        width=16,
        height=16,
        format=SimpleNamespace(name="yuv420p"),
        pts=None,
    )
    try:
        with pytest.raises(VideoEncoderError, match="Encoding error"):
            encoder.encode(frame)
    finally:
        encoder.close()


def test_video_encoder_flush_returns_empty_when_context_reports_error(monkeypatch) -> None:
    context = _FakeContext("libx264", "w")

    def fail_encode(_frame) -> None:
        raise av.error.InvalidDataError(1, "flush failed")

    context.encode = fail_encode
    monkeypatch.setattr(pyav_module.av.CodecContext, "create", lambda *_args: context)
    encoder = pyav_module.VideoEncoder(16, 16, "libx264")
    try:
        assert encoder.flush_packets() == []
    finally:
        encoder.close()


def test_pyav_decompressor_falls_back_between_av1_decoders(monkeypatch) -> None:
    created = []

    def create(name: str, direction: str):
        created.append(name)
        if name == "libaom-av1":
            raise av.error.InvalidDataError(1, "decoder unavailable")
        return _FakeContext(name, direction)

    monkeypatch.setattr(pyav_module.av.CodecContext, "create", create)
    decompressor = pyav_module.PyAVVideoDecompressor()

    assert decompressor._ensure_decoder("av1").codec == "libdav1d"
    assert created == ["libaom-av1", "libdav1d"]


def test_pyav_decompressor_reports_missing_decoder(monkeypatch) -> None:
    monkeypatch.setattr(
        pyav_module.av.CodecContext,
        "create",
        lambda *_args: (_ for _ in ()).throw(av.error.InvalidDataError(1, "missing")),
    )

    with pytest.raises(VideoEncoderError, match="No usable decoder for av1"):
        pyav_module.PyAVVideoDecompressor()._ensure_decoder("av1")


def test_pyav_decompressor_flushes_raw_pending_frames(monkeypatch) -> None:
    class Frame:
        width = 2
        height = 1

        def reformat(self, *, format: str):
            assert format == "rgb24"
            return self

    class Decoder:
        def decode(self, packet):
            assert packet is None
            return [Frame()]

    decompressor = pyav_module.PyAVVideoDecompressor(video_format="raw")
    decompressor._decoder = Decoder()
    decompressor._decoder_codecs = ("h264",)
    monkeypatch.setattr(pyav_module, "video_frame_to_rgb_bytes", lambda _: b"rgb")

    assert decompressor.flush() == [
        DecompressedFrame(data=b"rgb", width=2, height=1, is_jpeg=False)
    ]


def test_pyav_decompressor_returns_none_when_packet_is_buffered() -> None:
    class Decoder:
        def decode(self, _packet):
            return []

    decompressor = pyav_module.PyAVVideoDecompressor()
    decompressor._decoder = Decoder()
    decompressor._decoder_codecs = ("h264",)

    assert decompressor.decompress(b"buffered", "h264") is None


def test_pyav_decompressor_context_manager_releases_contexts() -> None:
    decompressor = pyav_module.PyAVVideoDecompressor()
    decompressor._decoder = object()
    decompressor._decoder_codecs = ("h264",)
    decompressor._pending_frames.append(object())
    decompressor._jpeg_encoder = object()

    with decompressor as context:
        assert context is decompressor

    assert decompressor._decoder is None
    assert decompressor._decoder_codecs is None
    assert not decompressor._pending_frames
    assert decompressor._jpeg_encoder is None


def test_pyav_decompressor_flushes_compressed_pending_frames(monkeypatch) -> None:
    frame = object()

    class Decoder:
        def decode(self, packet):
            assert packet is None
            return [frame]

    decompressor = pyav_module.PyAVVideoDecompressor()
    decompressor._decoder = Decoder()
    decompressor._decoder_codecs = ("h264",)
    monkeypatch.setattr(
        decompressor,
        "_frame_to_jpeg",
        lambda _: DecompressedFrame(data=b"jpeg", width=1, height=1, is_jpeg=True),
    )

    assert decompressor.flush() == [
        DecompressedFrame(data=b"jpeg", width=1, height=1, is_jpeg=True)
    ]
