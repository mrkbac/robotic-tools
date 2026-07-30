"""Tests for VideoEncoder resource management."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from mcap_codec_support.video import VideoEncoderError
from mcap_codec_support.video.common import raw_image_to_pil
from mcap_codec_support.video.pyav import (
    VideoEncoder,
    raw_image_to_frame,
    video_frame_to_rgb_bytes,
)

av = pytest.importorskip("av")


def _make_encoder() -> VideoEncoder:
    return VideoEncoder(width=16, height=16, codec_name="libx264")


@pytest.fixture
def encoder():
    enc = _make_encoder()
    yield enc
    # Close in case a test exits before releasing the encoder.
    enc.close()


class TestVideoEncoderClose:
    def test_close_releases_context(self, encoder):
        encoder.close()
        assert encoder._context is None

    def test_close_is_idempotent(self, encoder):
        encoder.close()
        encoder.close()  # should not raise

    def test_context_manager_closes_on_exit(self):
        with _make_encoder() as enc:
            assert enc._context is not None
        assert enc._context is None

    def test_context_manager_returns_encoder(self):
        enc = _make_encoder()
        with enc as ctx:
            assert ctx is enc
        enc.close()

    def test_context_manager_closes_on_exception(self):
        enc = _make_encoder()
        with pytest.raises(RuntimeError), enc:
            raise RuntimeError("simulated error")
        assert enc._context is None


@pytest.mark.parametrize(
    ("encoding", "source_pixel", "expected_pixel"),
    [
        ("rgb8", b"\x0a\x14\x1e", b"\x0a\x14\x1e"),
        ("bgr8", b"\x0a\x14\x1e", b"\x1e\x14\x0a"),
        ("mono8", b"\x28", b"\x28\x28\x28"),
    ],
)
def test_raw_image_to_frame_honors_padded_rows(
    encoding: str, source_pixel: bytes, expected_pixel: bytes
) -> None:
    width = 5
    height = 3
    row = source_pixel * width
    step = len(row) + 2
    message = SimpleNamespace(
        width=width,
        height=height,
        encoding=encoding,
        step=step,
        data=b"".join(row + bytes((0xA0 + index, 0xB0 + index)) for index in range(height)),
    )

    frame = raw_image_to_frame(message)

    assert frame.width == width
    assert frame.height == height
    assert frame.planes[0].line_size > len(row)
    assert video_frame_to_rgb_bytes(frame) == expected_pixel * width * height
    assert raw_image_to_pil(message).tobytes() == expected_pixel * width * height


def test_raw_image_to_frame_rejects_short_step() -> None:
    message = SimpleNamespace(
        width=5,
        height=3,
        encoding="rgb8",
        step=14,
        data=bytes(14 * 3),
    )

    with pytest.raises(VideoEncoderError, match="step 14 is smaller than row size 15"):
        raw_image_to_frame(message)


def test_raw_image_to_frame_rejects_short_padded_data() -> None:
    message = SimpleNamespace(
        width=5,
        height=3,
        encoding="rgb8",
        step=17,
        data=bytes(50),
    )

    with pytest.raises(VideoEncoderError, match="50 bytes, expected at least 51"):
        raw_image_to_frame(message)


def test_video_runtime_imports_without_numpy() -> None:
    script = """
import importlib.abc
import sys
from types import SimpleNamespace

class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError(f"blocked {fullname}", name=fullname)
        return None

sys.meta_path.insert(0, BlockNumpy())
from mcap_codec_support.video import EncoderMode
from mcap_codec_support.video.pyav import raw_image_to_frame, video_frame_to_rgb_bytes

message = SimpleNamespace(
    width=5,
    height=3,
    encoding="rgb8",
    step=17,
    data=(bytes(range(15)) + b"xx") * 3,
)
frame = raw_image_to_frame(message)
assert len(video_frame_to_rgb_bytes(frame)) == 5 * 3 * 3
assert EncoderMode.PYAV.value == "pyav"
assert "numpy" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)
