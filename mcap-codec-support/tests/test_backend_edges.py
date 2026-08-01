"""Regression tests for backend-independent image layout handling."""

from types import SimpleNamespace

import pytest
from mcap_codec_support.video.compression import _FfmpegCliCompressionBackend
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
