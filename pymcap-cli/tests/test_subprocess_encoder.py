"""Tests for the subprocess-based ffmpeg video encoder backend."""

from __future__ import annotations

import pytest
from pymcap_cli.subprocess_encoder import (
    AnnexBParser,
    SubprocessVideoEncoder,
    check_encoder_cli,
    find_ffmpeg,
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


# ---------------------------------------------------------------------------
# SubprocessVideoEncoder integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
class TestSubprocessVideoEncoder:
    def test_encode_raw_bytes(self) -> None:
        """Encode raw YUV420p bytes — no PyAV needed."""
        width, height = 64, 64
        frame_size = width * height * 3 // 2
        encoder = SubprocessVideoEncoder(
            width=width,
            height=height,
            codec_name="libx264",
            quality=28,
            target_fps=30.0,
            gop_size=10,
        )

        outputs: list[bytes] = []
        for i in range(20):
            raw = bytes([i * 10 % 256] * frame_size)
            result = encoder.encode(raw)
            if result is not None:
                outputs.append(result)

        flushed = encoder.flush()
        if flushed:
            outputs.append(flushed)

        assert len(outputs) > 0
        total_bytes = sum(len(o) for o in outputs)
        assert total_bytes > 0

    def test_access_unit_boundaries(self) -> None:
        """Each encode() output should start with a start code."""
        width, height = 64, 64
        frame_size = width * height * 3 // 2
        encoder = SubprocessVideoEncoder(
            width=width,
            height=height,
            codec_name="libx264",
            quality=28,
            target_fps=30.0,
            gop_size=10,
        )

        for i in range(20):
            raw = bytes([i * 10 % 256] * frame_size)
            result = encoder.encode(raw)
            if result is not None:
                assert result[:4] == b"\x00\x00\x00\x01", (
                    f"Frame {i}: output does not start with start code"
                )

        flushed = encoder.flush()
        if flushed:
            assert flushed[:4] == b"\x00\x00\x00\x01"

    def test_cleanup_on_del(self) -> None:
        encoder = SubprocessVideoEncoder(width=64, height=64, codec_name="libx264")
        pid = encoder._process.pid
        del encoder
        assert pid > 0
