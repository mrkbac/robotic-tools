"""Tests for codec-specific encoder option construction."""

import mcap_codec_support.video.ffmpeg as ffmpeg_module
import pytest
from mcap_codec_support.video.common import build_encoder_options


def test_build_encoder_options_hevc_nvenc_uses_constant_qp() -> None:
    low_quality_options, low_quality_bit_rate = build_encoder_options("hevc_nvenc", 28, 1920, 1080)
    high_quality_options, high_quality_bit_rate = build_encoder_options(
        "hevc_nvenc", 20, 1920, 1080
    )

    assert low_quality_options == {
        "preset": "p5",
        "tune": "hq",
        "rc": "constqp",
        "qp": "28",
    }
    assert high_quality_options == {
        "preset": "p5",
        "tune": "hq",
        "rc": "constqp",
        "qp": "20",
    }
    assert low_quality_bit_rate is None
    assert high_quality_bit_rate is None


def test_build_encoder_options_h264_nvenc_keeps_vbr_quality() -> None:
    options, bit_rate = build_encoder_options("h264_nvenc", 28, 1920, 1080)

    assert options == {"rc": "vbr", "cq": "28"}
    assert bit_rate is None


def test_build_encoder_options_nvenc_honors_preset() -> None:
    options, _ = build_encoder_options("hevc_nvenc", 20, 1920, 1080, preset="p7")

    assert options["preset"] == "p7"


def test_ffmpeg_hevc_nvenc_output_args_include_constant_qp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ffmpeg_module, "_frame_sync_args", lambda _: [])
    options, bit_rate = build_encoder_options("hevc_nvenc", 20, 1920, 1080)

    args = ffmpeg_module._build_output_args("ffmpeg", "hevc", "hevc_nvenc", 30, options, bit_rate)

    assert "-b:v" not in args
    assert args[args.index("-rc") : args.index("-rc") + 2] == ["-rc", "constqp"]
    assert args[args.index("-qp") : args.index("-qp") + 2] == ["-qp", "20"]
    assert "-cq" not in args
