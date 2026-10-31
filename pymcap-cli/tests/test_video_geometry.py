"""Raw encoder input dimensions must agree with each decoded pixel buffer."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from mcap_codec_support.video.compression import _FfmpegCliCompressionBackend
from mcap_codec_support.video.ffmpeg import check_encoder_cli, find_ffmpeg
from pymcap_cli.core.processors.video_compress import (
    VideoEncoderSession,
    VideoEncoderSettings,
    resolve_video_encoder_settings,
)


class _GeometryBackend:
    def __init__(self, pix_fmt: str | None) -> None:
        self.pix_fmt = pix_fmt

    def get_pix_fmt(self, _topic: str) -> str | None:
        return self.pix_fmt


@pytest.mark.parametrize(
    ("width", "height", "scale", "expected_scale"),
    [(63, 47, None, (62, 46)), (160, 120, 80, (80, 60)), (160, 120, None, None)],
)
def test_resolve_video_encoder_settings_keeps_raw_input_dimensions(
    width: int,
    height: int,
    scale: int | None,
    expected_scale: tuple[int, int] | None,
) -> None:
    assert resolve_video_encoder_settings(
        backend=_GeometryBackend("rgb24"),
        topic="/CAM_FRONT/image",
        width=width,
        height=height,
        scale=scale,
    ) == VideoEncoderSettings(width, height, "rgb24", expected_scale)


@pytest.mark.parametrize(
    ("width", "height", "scale", "output_size"),
    [(63, 47, None, (62, 46)), (160, 120, 80, (80, 60))],
)
def test_video_session_ffmpeg_jpeg_preserves_scaled_image_boundaries(
    width: int, height: int, scale: int | None, output_size: tuple[int, int]
) -> None:
    av = pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    if find_ffmpeg() is None or not check_encoder_cli("libx264"):
        pytest.skip("ffmpeg/libx264 not available")

    backend = _FfmpegCliCompressionBackend()
    session = VideoEncoderSession(
        backend=backend,
        encoder_name="libx264",
        codec="h264",
        quality=18,
        topic="/CAM_FRONT/image",
        scale=scale,
    )
    packets = []
    expected = []
    try:
        for index in range(3):
            pixels = bytes(
                component
                for y in range(height)
                for x in range(width)
                for component in (x * 255 // width, y * 255 // height, index * 60)
            )
            buffer = BytesIO()
            image_module.frombytes("RGB", (width, height), pixels).save(
                buffer, format="JPEG", quality=95
            )
            jpeg = buffer.getvalue()
            message = SimpleNamespace(
                decoded_message=SimpleNamespace(data=jpeg, format="jpeg"),
                channel=SimpleNamespace(topic="/CAM_FRONT/image"),
            )
            frame, frame_width, frame_height = backend.decode_image(
                message, "sensor_msgs/CompressedImage"
            )
            outcome = session.encode_with_fallback(frame, frame_width, frame_height)
            assert not outcome.failed
            if outcome.data is not None:
                packets.append(outcome.data)
            expected.append(
                image_module.open(BytesIO(jpeg))
                .convert("RGB")
                .resize(output_size, image_module.Resampling.BILINEAR)
                .tobytes()
            )
        assert session.encoder is not None
        packets.extend(session.encoder.flush_packets())
    finally:
        session.close()

    decoder = av.CodecContext.create("h264", "r")
    frames = [frame for packet in packets for frame in decoder.decode(av.Packet(packet))]
    frames.extend(decoder.decode(None))
    assert len(frames) == len(expected)
    for frame, expected_pixels in zip(frames, expected, strict=True):
        assert (frame.width, frame.height) == output_size
        actual_pixels = frame.to_image().tobytes()
        mean_error = sum(
            abs(actual - wanted)
            for actual, wanted in zip(actual_pixels, expected_pixels, strict=True)
        ) / len(expected_pixels)
        assert mean_error < 10
