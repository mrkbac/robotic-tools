"""Regression tests for backend-independent image layout handling."""

from queue import Empty
from types import SimpleNamespace

import mcap_codec_support.video.pyav as pyav_module
import pytest
from mcap_codec_support.video.common import DecompressedFrame
from mcap_codec_support.video.compression import _FfmpegCliCompressionBackend
from mcap_codec_support.video.ffmpeg import FFmpegVideoDecompressor
from mcap_codec_support.video.gstreamer import GStreamerCompressionBackend


@pytest.mark.parametrize(
    "backend_factory",
    [_FfmpegCliCompressionBackend, GStreamerCompressionBackend],
    ids=["ffmpeg-cli", "gstreamer"],
)
def test_backends_remove_ros_row_padding_before_raw_encode(backend_factory) -> None:
    row_bytes = b"abcdef"
    padded_rows = row_bytes + b"XX" + b"ghijkl" + b"YY"
    message = SimpleNamespace(
        decoded_message=SimpleNamespace(
            data=padded_rows,
            width=2,
            height=2,
            encoding="rgb8",
            step=8,
        ),
        channel=SimpleNamespace(topic="/camera/image"),
    )

    backend = backend_factory()
    data, width, height = backend.decode_image(message, "sensor_msgs/Image")

    assert (width, height) == (2, 2)
    assert data == row_bytes + b"ghijkl"
    assert backend.get_pix_fmt("/camera/image") == "rgb24"


def test_pyav_decompressor_does_not_reuse_decoder_for_a_new_codec(monkeypatch) -> None:
    """A channel can carry a new keyframe with a different codec."""
    created = []

    class FakeDecoder:
        def __init__(self, codec: str) -> None:
            self.codec = codec

        def open(self) -> None:
            pass

        def decode(self, _packet):
            return []

    def create(codec: str, direction: str) -> FakeDecoder:
        assert direction == "r"
        decoder = FakeDecoder(codec)
        created.append(decoder)
        return decoder

    monkeypatch.setattr(pyav_module, "_create_codec_context", create)
    decompressor = pyav_module.PyAVVideoDecompressor()

    assert decompressor._ensure_decoder("h264").codec == "h264"
    assert decompressor._ensure_decoder("vp9").codec == "vp9"
    assert [decoder.codec for decoder in created] == ["h264", "vp9"]


def test_ffmpeg_decompressor_restarts_when_codec_changes(monkeypatch) -> None:
    created: list[str] = []

    class FakeStdin:
        closed = False

        def write(self, data: bytes) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        stdin = FakeStdin()

        def wait(self, timeout: float) -> None:
            pass

    decompressor = FFmpegVideoDecompressor()

    def start(codec: str) -> None:
        created.append(codec)
        decompressor._process = FakeProcess()

    monkeypatch.setattr(decompressor, "_start_process", start)

    def no_frame(*, timeout: float):
        del timeout
        raise Empty

    monkeypatch.setattr(decompressor._output_queue, "get", no_frame)

    decompressor.decompress(b"h264 packet", "h264")
    decompressor.decompress(b"vp9 packet", "vp9")

    assert created == ["h264", "vp9"]


def test_ffmpeg_decompressor_flush_resets_process_for_reuse() -> None:
    class FakeStdin:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        stdin = FakeStdin()

        def wait(self, timeout: float) -> None:
            pass

    decompressor = FFmpegVideoDecompressor()
    decompressor._process = FakeProcess()
    decompressor._codec_family = "h264"
    decompressor._output_queue.put(None)

    assert decompressor.flush() == []
    assert decompressor._process is None


def test_pyav_decompressor_preserves_multiple_frames_from_one_packet(monkeypatch) -> None:
    class FakeDecoder:
        def __init__(self) -> None:
            self.calls = 0

        def decode(self, _packet):
            self.calls += 1
            return ["first", "second"] if self.calls == 1 else []

    decoder = FakeDecoder()
    decompressor = pyav_module.PyAVVideoDecompressor()
    decompressor._decoder = decoder
    decompressor._decoder_codecs = ("h264",)
    monkeypatch.setattr(
        decompressor,
        "_frame_to_jpeg",
        lambda frame: DecompressedFrame(data=frame.encode(), width=1, height=1, is_jpeg=True),
    )

    first = decompressor.decompress(b"packet", "h264")
    second = decompressor.decompress(b"next packet", "h264")

    assert first is not None
    assert first.data == b"first"
    assert second is not None
    assert second.data == b"second"


def test_pyav_decompressor_flushes_delayed_frames_before_codec_switch(monkeypatch) -> None:
    class FakeDecoder:
        def __init__(self, codec: str) -> None:
            self.codec = codec

        def open(self) -> None:
            pass

        def decode(self, packet):
            if packet is None:
                return ["delayed"] if self.codec == "h264" else []
            return ["new"] if self.codec == "vp9" else []

    monkeypatch.setattr(
        pyav_module, "_create_codec_context", lambda codec, _direction: FakeDecoder(codec)
    )
    decompressor = pyav_module.PyAVVideoDecompressor()
    monkeypatch.setattr(
        decompressor,
        "_frame_to_jpeg",
        lambda frame: DecompressedFrame(data=frame.encode(), width=1, height=1, is_jpeg=True),
    )

    assert decompressor.decompress(b"h264 packet", "h264") is None
    delayed = decompressor.decompress(b"vp9 packet", "vp9")
    assert delayed is not None
    assert delayed.data == b"delayed"
    new = decompressor.decompress(b"next vp9 packet", "vp9")
    assert new is not None
    assert new.data == b"new"
