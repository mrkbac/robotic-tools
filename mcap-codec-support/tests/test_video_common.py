"""Portable tests for shared video configuration and image helpers."""

from types import SimpleNamespace

import mcap_codec_support.video.common as common
import mcap_codec_support.video.compression as compression_module
import pytest
from mcap_codec_support.video.common import (
    VideoCodec,
    VideoEncoderError,
    build_encoder_options,
    calculate_downscale_dimensions,
    get_encoder_options,
    get_software_encoder,
    raw_image_to_pil,
    resolve_encoder,
    resolve_encoder_for_backend,
)


@pytest.mark.parametrize(
    ("codec_name", "quality", "preset", "expected", "bit_rate"),
    [
        ("libx264", 24, None, {"crf": "24", "preset": "superfast", "tune": "zerolatency"}, None),
        ("libx264", 24, "fast", {"crf": "24", "preset": "fast", "tune": "zerolatency"}, None),
        ("libx265", 26, None, {"crf": "26", "preset": "superfast"}, None),
        ("h264_videotoolbox", 28, None, {}, 5_000_000),
        ("hevc_videotoolbox", 22, None, {}, 5_000_000 * 2),
        ("h264_vaapi", 28, None, {"qp": "28"}, None),
        ("libvpx-vp9", 30, "good", {"cpu-used": "4", "crf": "30", "deadline": "good"}, None),
        ("libaom-av1", 30, "realtime", {"cpu-used": "6", "crf": "30", "usage": "realtime"}, None),
        ("libsvtav1", 30, None, {"preset": "8", "crf": "30"}, None),
        ("unknown", 30, None, {}, None),
    ],
)
def test_build_encoder_options_matrix(codec_name, quality, preset, expected, bit_rate) -> None:
    options, actual_bit_rate = build_encoder_options(codec_name, quality, 1920, 1080, preset=preset)

    assert options == expected
    assert actual_bit_rate == bit_rate


def test_build_encoder_options_videotoolbox_scales_bitrate() -> None:
    options, bit_rate = build_encoder_options("h264_videotoolbox", 28, 960, 540)

    assert options == {}
    assert bit_rate == 1_250_000


@pytest.mark.parametrize(
    ("codec", "encoder", "expected"),
    [
        (VideoCodec.H264, "h264_nvenc", {"preset": "p4"}),
        (VideoCodec.H264, "libx264", {"preset": "medium"}),
        (VideoCodec.H265, "libx265", {"preset": "medium"}),
        (VideoCodec.VP9, "libvpx-vp9", {}),
    ],
)
def test_get_encoder_options(codec: VideoCodec, encoder: str, expected: dict[str, str]) -> None:
    assert get_encoder_options(codec, encoder) == expected


def test_get_software_encoder_rejects_unknown_codec() -> None:
    with pytest.raises(ValueError, match="Unsupported codec 'unknown'"):
        get_software_encoder("unknown")


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            SimpleNamespace(width=0, height=1, encoding="mono8", step=1, data=b"x"),
            "dimensions must be positive",
        ),
        (
            SimpleNamespace(width=1, height=1, encoding="mono8", step=1, data=b""),
            "Image has no data",
        ),
    ],
)
def test_raw_image_to_buffer_rejects_invalid_messages(message, error: str) -> None:
    with pytest.raises(VideoEncoderError, match=error):
        common.raw_image_to_buffer(message)


def test_encode_raw_image_rejects_unknown_format() -> None:
    message = SimpleNamespace(width=2, height=2, encoding="rgb8", step=6, data=bytes(12))

    with pytest.raises(VideoEncoderError, match="Unsupported image format 'webp'"):
        compression_module.encode_raw_image_to_compressed(
            message, image_format="webp", jpeg_quality=90, scale=None
        )


@pytest.mark.parametrize(
    ("width", "height", "maximum", "expected"),
    [
        (640, 480, 1920, (640, 480)),
        (641, 481, 1920, (640, 480)),
        (4000, 2000, 1000, (1000, 500)),
        (2000, 4000, 1000, (500, 1000)),
    ],
)
def test_calculate_downscale_dimensions(width, height, maximum, expected) -> None:
    assert calculate_downscale_dimensions(width, height, maximum) == expected


def test_raw_image_to_pil_converts_mono_to_rgb() -> None:
    message = SimpleNamespace(
        width=2,
        height=1,
        encoding="mono8",
        step=2,
        data=b"\x01\xfe",
    )

    image = raw_image_to_pil(message)

    assert image.mode == "RGB"
    assert image.tobytes() == b"\x01\x01\x01\xfe\xfe\xfe"


def test_raw_image_to_pil_reports_missing_pillow(monkeypatch) -> None:
    original_import = __import__("builtins").__import__

    def block_pillow(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", block_pillow)

    with pytest.raises(VideoEncoderError, match="Pillow is required"):
        raw_image_to_pil(SimpleNamespace(width=1, height=1, encoding="mono8", step=1, data=b"x"))


def test_resolve_encoder_prefers_linux_hardware(monkeypatch) -> None:
    monkeypatch.setattr(common.platform, "system", lambda: "Linux")
    seen: list[str] = []

    def available(name: str) -> bool:
        seen.append(name)
        return name == "h264_nvenc"

    assert resolve_encoder("h264", test_fn=available) == "h264_nvenc"
    assert seen == ["h264_nvenc"]


def test_resolve_encoder_uses_software_when_hardware_disabled() -> None:
    assert resolve_encoder("h264", test_fn=lambda name: name == "libx264", use_hardware=False) == (
        "libx264"
    )


def test_resolve_encoder_rejects_unknown_codec() -> None:
    with pytest.raises(ValueError, match="Unsupported codec 'unknown'"):
        resolve_encoder("unknown", test_fn=lambda _: True)


def test_resolve_encoder_reports_no_software_encoder() -> None:
    with pytest.raises(ValueError, match="No available encoder for codec 'h264'"):
        resolve_encoder("h264", test_fn=lambda _: False, use_hardware=False)


@pytest.mark.parametrize("backend", ["auto", "software"])
def test_resolve_encoder_for_backend_wraps_resolution_errors(backend: str) -> None:
    with pytest.raises(VideoEncoderError, match="No available encoder"):
        resolve_encoder_for_backend("h264", backend, test_fn=lambda _: False)


def test_resolve_encoder_for_backend_rejects_unsupported_hardware() -> None:
    with pytest.raises(VideoEncoderError, match="not available for codec"):
        resolve_encoder_for_backend("vp9", "nvenc", test_fn=lambda _: True)


def test_resolve_encoder_for_backend_reports_unavailable_hardware() -> None:
    with pytest.raises(VideoEncoderError, match="not available on this system"):
        resolve_encoder_for_backend("h264", "nvenc", test_fn=lambda _: False)


def test_resolve_encoder_for_backend_returns_available_hardware() -> None:
    assert resolve_encoder_for_backend("h264", "nvenc", test_fn=lambda _: True) == "h264_nvenc"
