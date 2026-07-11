"""Tests for the subprocess-based ffmpeg video encoder backend."""

from __future__ import annotations

import io
import subprocess

import mcap_codec_support.video.compression as compression_module
import mcap_codec_support.video.ffmpeg as ffmpeg_module
import pytest
from mcap_codec_support.video import EncoderMode, create_video_compression_backend
from mcap_codec_support.video.common import PROBE_JPEG
from mcap_codec_support.video.ffmpeg import (
    AnnexBParser,
    FFmpegVideoEncoder,
    check_decoder_cli,
    check_encoder_cli,
    find_ffmpeg,
    probe_hw_mjpeg_decoder,
    probe_image_dimensions,
)

# ---------------------------------------------------------------------------
# AnnexBParser tests
# ---------------------------------------------------------------------------

# H.264 start code + slice NAL type 5 (IDR)
H264_IDR = b"\x00\x00\x00\x01\x65\xab\xcd\xef"
# H.264 start code + SPS NAL type 7 (non-VCL)
H264_SPS = b"\x00\x00\x00\x01\x67\x42\x00\x0a"
# H.264 start code + slice NAL type 1 (non-IDR)
H264_SLICE = b"\x00\x00\x00\x01\x41\xab\xcd\xef"

# x264 writes a 4-byte start code on the first NAL of an access unit and 3-byte
# codes on the NALs inside it; IDR access units open with SPS + PPS.
# Payload high bit set ⇔ first_mb_in_slice == 0 (slice starts a new picture).
H264_P1 = b"\x00\x00\x00\x01\x41\xe0\x11\x22"
H264_P2 = b"\x00\x00\x00\x01\x41\xe0\x33\x44"
# Continuation slice of the same picture (first_mb_in_slice > 0 → high bit clear).
H264_P_CONT_4BYTE = b"\x00\x00\x00\x01\x41\x40\x55\x66"
H264_PPS_3BYTE = b"\x00\x00\x01\x68\xce\x38\x80"
H264_IDR_3BYTE = b"\x00\x00\x01\x65\x88\x84\x00"


class TestAnnexBParser:
    def test_single_au_flush(self) -> None:
        """A single AU followed by flush returns it."""
        parser = AnnexBParser("h264")
        data = H264_SPS + H264_IDR
        result = parser.feed(data)
        assert result == []
        flushed = parser.flush()
        assert flushed is not None
        assert len(flushed) > 0

    def test_two_aus(self) -> None:
        """Two VCL NALs yield the first AU when the second VCL starts."""
        parser = AnnexBParser("h264")
        data = H264_SPS + H264_IDR + H264_SLICE
        result = parser.feed(data)
        assert len(result) == 1
        flushed = parser.flush()
        assert flushed is not None

    def test_incremental_feed(self) -> None:
        """Feeding byte-by-byte should produce the same results."""
        parser = AnnexBParser("h264")
        data = H264_IDR + H264_SLICE + b"\x00\x00\x00\x00"

        all_aus: list[bytes] = []
        for byte in data:
            all_aus.extend(parser.feed(bytes([byte])))

        flushed = parser.flush()
        assert len(all_aus) == 1
        assert flushed is not None

    def test_empty(self) -> None:
        parser = AnnexBParser("h264")
        assert parser.feed(b"") == []
        assert parser.flush() is None

    def test_h265(self) -> None:
        """H.265 VCL NAL types (0-31) are detected correctly."""
        # H.265 NAL type 1 (coded slice): header byte = (1 << 1) = 0x02
        h265_vcl = b"\x00\x00\x00\x01\x02\x01\xab\xcd"
        h265_vcl2 = b"\x00\x00\x00\x01\x02\x01\xef\x01"
        parser = AnnexBParser("h265")
        result = parser.feed(h265_vcl + h265_vcl2)
        assert len(result) == 1

    def test_sps_led_idr_au_is_own_au(self) -> None:
        """A P-frame directly before an SPS-led IDR AU must not merge into it.

        This is the x264 GOP-boundary layout: the IDR access unit starts with
        SPS/PPS (4-byte start code on the SPS, 3-byte codes inside the AU).
        """
        parser = AnnexBParser("h264")
        idr_au = H264_SPS + H264_PPS_3BYTE + H264_IDR_3BYTE
        aus = parser.feed(H264_P1 + idr_au + H264_P2)
        aus.extend(parser.flush_list())
        assert aus == [H264_P1, idr_au, H264_P2]

    def test_sps_led_idr_au_boundary_in_flush(self) -> None:
        """flush_list splits the SPS-led IDR AU from the preceding P-frame."""
        parser = AnnexBParser("h264")
        idr_au = H264_SPS + H264_PPS_3BYTE + H264_IDR_3BYTE
        aus = parser.feed(H264_P1 + idr_au)
        aus.extend(parser.flush_list())
        assert aus == [H264_P1, idr_au]

    def test_multi_slice_picture_is_one_au(self) -> None:
        """Continuation slices (first_mb_in_slice > 0) stay in the same AU."""
        parser = AnnexBParser("h264")
        aus = parser.feed(H264_P1 + H264_P_CONT_4BYTE + H264_P2)
        aus.extend(parser.flush_list())
        assert aus == [H264_P1 + H264_P_CONT_4BYTE, H264_P2]

    def test_h265_sps_led_idr_au_is_own_au(self) -> None:
        """H.265 equivalent: SPS (type 33) opens the IDR AU."""
        h265_p1 = b"\x00\x00\x00\x01\x02\x01\xab\xcd"
        h265_sps = b"\x00\x00\x00\x01\x42\x01\x01\x02"
        h265_idr_3byte = b"\x00\x00\x01\x26\x01\xaf\x08"  # IDR_W_RADL (type 19)
        h265_p2 = b"\x00\x00\x00\x01\x02\x01\xef\x01"
        parser = AnnexBParser("h265")
        idr_au = h265_sps + h265_idr_3byte
        aus = parser.feed(h265_p1 + idr_au + h265_p2)
        aus.extend(parser.flush_list())
        assert aus == [h265_p1, idr_au, h265_p2]


# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------


class TestFfmpegDiscovery:
    def test_find_ffmpeg(self) -> None:
        result = find_ffmpeg()
        assert result is None or isinstance(result, str)

    @pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
    def test_check_encoder_cli_libx264(self) -> None:
        assert check_encoder_cli("libx264") is True

    @pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
    def test_check_encoder_cli_nonexistent(self) -> None:
        assert check_encoder_cli("totally_fake_encoder_xyz") is False

    def test_resolve_encoder_skips_listed_but_unusable_hardware(self, monkeypatch) -> None:
        monkeypatch.setattr(ffmpeg_module, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
        monkeypatch.setattr(ffmpeg_module.platform, "system", lambda: "Linux")

        def fake_run(args, **_kwargs):
            if "-encoders" in args:
                stdout = " V..... h264_nvenc NVIDIA NVENC H.264 encoder\n"
                return subprocess.CompletedProcess(args, 0, stdout, "")
            if "h264_nvenc" in args:
                return subprocess.CompletedProcess(args, 1, b"", b"no NVIDIA device")
            raise AssertionError(f"unexpected ffmpeg command: {args}")

        monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)

        ffmpeg_module.probe_encoder_cli.cache_clear()
        try:
            assert ffmpeg_module.resolve_encoder("h264") == "libx264"
        finally:
            ffmpeg_module.probe_encoder_cli.cache_clear()


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
class TestImageDimensionProbe:
    def test_probe_falls_back_to_ffprobe_with_binary_input(self) -> None:
        image_module = pytest.importorskip("PIL.Image")
        image_module.init()
        if "WEBP" not in image_module.SAVE:
            pytest.skip("Pillow build cannot write WebP")

        image = image_module.new("RGB", (37, 23), color=(10, 20, 30))
        buf = io.BytesIO()
        image.save(buf, format="WEBP")

        assert probe_image_dimensions(buf.getvalue()) == (37, 23)


class TestHwMjpegDecodeProbe:
    @pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
    def test_check_decoder_cli(self) -> None:
        assert check_decoder_cli("mjpeg") is True  # CPU mjpeg always present
        assert check_decoder_cli("totally_fake_decoder_xyz") is False

    def test_probe_returns_str_or_none(self) -> None:
        result = probe_hw_mjpeg_decoder()
        assert result is None or isinstance(result, str)

    def test_probe_none_when_no_candidates(self, monkeypatch) -> None:
        # No platform candidates → None regardless of ffmpeg/platform. A broken
        # candidate that hangs is covered by the real timed probe (killed on
        # timeout), which cannot be reproduced hermetically here.
        monkeypatch.setattr(ffmpeg_module, "_HW_MJPEG_DECODERS", {})
        ffmpeg_module.probe_hw_mjpeg_decoder.cache_clear()
        try:
            assert ffmpeg_module.probe_hw_mjpeg_decoder() is None
        finally:
            ffmpeg_module.probe_hw_mjpeg_decoder.cache_clear()

    @pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
    def test_encoder_accepts_forced_decode_codec(self) -> None:
        # Forcing the CPU mjpeg decoder exercises the decode_codec code path
        # end-to-end (the -c:v insertion must produce a valid command).
        encoder = FFmpegVideoEncoder(
            width=32, height=32, codec_name="libx264", quality=28, decode_codec="mjpeg"
        )
        outputs: list[bytes] = []
        try:
            for _ in range(10):
                result = encoder.encode(PROBE_JPEG)
                if result is not None:
                    outputs.append(result)
            outputs.extend(encoder.flush_packets())
        finally:
            encoder.close()
        assert len(outputs) == 10


class TestBackendSelection:
    @pytest.fixture(autouse=True)
    def _skip_gstreamer_auto(self, monkeypatch) -> None:
        monkeypatch.setattr(
            compression_module,
            "_create_gstreamer_backend_if_healthy",
            lambda *_: None,
        )

    def test_auto_falls_back_to_ffmpeg_when_pyav_is_missing(self, monkeypatch) -> None:
        def raise_import_error(*_args):
            raise ImportError("No module named av")

        monkeypatch.setattr(
            compression_module._PyAVCompressionBackend,
            "resolve_encoder",
            raise_import_error,
        )

        backend = create_video_compression_backend(EncoderMode.AUTO, "h264", do_video=True)

        assert backend.label == "ffmpeg-cli"

    def test_auto_prefers_ffmpeg_when_pyav_only_software_but_ffmpeg_has_hardware(
        self, monkeypatch
    ) -> None:
        # PyAV present but its build lacks NVENC → resolves to software libx264;
        # system ffmpeg has hardware NVENC. AUTO should pick ffmpeg-cli so the
        # GPU is used (the common pip-installed-PyAV case).
        monkeypatch.setattr(
            compression_module._PyAVCompressionBackend,
            "resolve_encoder",
            lambda *_: "libx264",
        )
        monkeypatch.setattr(
            compression_module._FfmpegCliCompressionBackend,
            "resolve_encoder",
            lambda *_: "h264_nvenc",
        )
        backend = create_video_compression_backend(EncoderMode.AUTO, "h264", do_video=True)
        assert backend.label == "ffmpeg-cli"

    def test_auto_keeps_pyav_when_it_has_hardware(self, monkeypatch) -> None:
        monkeypatch.setattr(compression_module.os, "cpu_count", lambda: 4)
        monkeypatch.setattr(
            compression_module._PyAVCompressionBackend,
            "resolve_encoder",
            lambda *_: "h264_nvenc",
        )
        backend = create_video_compression_backend(EncoderMode.AUTO, "h264", do_video=True)
        assert backend.label != "ffmpeg-cli"

    def test_auto_prefers_ffmpeg_for_hardware_on_high_core_hosts(self, monkeypatch) -> None:
        monkeypatch.setattr(compression_module.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(
            compression_module._PyAVCompressionBackend,
            "resolve_encoder",
            lambda *_: "h264_nvenc",
        )
        monkeypatch.setattr(
            compression_module._FfmpegCliCompressionBackend,
            "resolve_encoder",
            lambda *_: "h264_nvenc",
        )
        backend = create_video_compression_backend(EncoderMode.AUTO, "h264", do_video=True)
        assert backend.label == "ffmpeg-cli"

    def test_auto_keeps_pyav_when_neither_has_hardware(self, monkeypatch) -> None:
        # No hardware anywhere → stay on PyAV (in-process, no subprocess/pipe cost).
        monkeypatch.setattr(compression_module.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(
            compression_module._PyAVCompressionBackend,
            "resolve_encoder",
            lambda *_: "libx264",
        )
        monkeypatch.setattr(
            compression_module._FfmpegCliCompressionBackend,
            "resolve_encoder",
            lambda *_: "libx264",
        )
        backend = create_video_compression_backend(EncoderMode.AUTO, "h264", do_video=True)
        assert backend.label != "ffmpeg-cli"


# ---------------------------------------------------------------------------
# FFmpegVideoEncoder integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
class TestFFmpegVideoEncoder:
    def test_encode_raw_bytes(self) -> None:
        """Encode raw YUV420p bytes without PyAV."""
        width, height = 64, 64
        frame_size = width * height * 3 // 2
        encoder = FFmpegVideoEncoder(
            width=width,
            height=height,
            codec_name="libx264",
            quality=28,
            target_fps=30.0,
            gop_size=10,
            input_pix_fmt="yuv420p",
        )

        outputs: list[bytes] = []
        for i in range(20):
            raw = bytes([i * 10 % 256] * frame_size)
            result = encoder.encode(raw)
            if result is not None:
                outputs.append(result)

        outputs.extend(encoder.flush_packets())

        assert len(outputs) > 0
        total_bytes = sum(len(o) for o in outputs)
        assert total_bytes > 0

    def test_access_unit_boundaries(self) -> None:
        """Each encode() output should start with a start code."""
        width, height = 64, 64
        frame_size = width * height * 3 // 2
        encoder = FFmpegVideoEncoder(
            width=width,
            height=height,
            codec_name="libx264",
            quality=28,
            target_fps=30.0,
            gop_size=10,
            input_pix_fmt="yuv420p",
        )

        for i in range(20):
            raw = bytes([i * 10 % 256] * frame_size)
            result = encoder.encode(raw)
            if result is not None:
                assert result[:4] == b"\x00\x00\x00\x01", (
                    f"Frame {i}: output does not start with start code"
                )

        for packet in encoder.flush_packets():
            assert packet[:4] == b"\x00\x00\x00\x01"

    def test_no_frame_loss_across_gops(self) -> None:
        """Every input frame yields exactly one access unit, incl. GOP boundaries.

        x264 opens each IDR access unit with SPS/PPS; a parser that only splits
        on VCL NALs merges the preceding P-frame into it, losing one frame per
        GOP (and shifting message/AU pairing by one from there on).
        """
        width, height = 64, 64
        frame_size = width * height * 3 // 2
        encoder = FFmpegVideoEncoder(
            width=width,
            height=height,
            codec_name="libx264",
            quality=28,
            target_fps=30.0,
            gop_size=10,
            input_pix_fmt="yuv420p",
        )

        n = 35
        outputs: list[bytes] = []
        for i in range(n):
            raw = bytes([(i * 7) % 256] * frame_size)
            result = encoder.encode(raw)
            if result is not None:
                outputs.append(result)
        outputs.extend(encoder.flush_packets())

        assert len(outputs) == n
        for packet in outputs:
            assert packet.startswith(b"\x00\x00\x00\x01")

    def test_cleanup_on_del(self) -> None:
        encoder = FFmpegVideoEncoder(
            width=64, height=64, codec_name="libx264", input_pix_fmt="yuv420p"
        )
        pid = encoder._process.pid
        del encoder
        assert pid > 0
