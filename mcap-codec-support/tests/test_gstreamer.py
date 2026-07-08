"""Tests for the Jetson GStreamer video backend."""

from __future__ import annotations

import mcap_codec_support.video.compression as compression_module
import mcap_codec_support.video.gstreamer as gstreamer_module
import pytest
from mcap_codec_support.video import EncoderMode, create_video_compression_backend
from mcap_codec_support.video.common import VideoEncoderError
from mcap_codec_support.video.gstreamer import (
    PROBE_JPEG,
    GStreamerCompressionBackend,
    _codec_key,
    check_encoder,
    find_gst_launch,
    gst_element_available,
    probe_hw_jpeg_pipeline,
    resolve_encoder,
)

_HAS_NV = find_gst_launch() is not None and gst_element_available("nvv4l2h264enc")
# The real hardware path only runs when a timed probe confirms the codec stack
# actually produces frames (present-but-broken hardware, like CUVID on Thor,
# probes False rather than hanging the whole suite).
_HW_OK = _HAS_NV and probe_hw_jpeg_pipeline("h264")


# ---------------------------------------------------------------------------
# Discovery / resolution (no hardware needed)
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_find_gst_launch(self) -> None:
        result = find_gst_launch()
        assert result is None or isinstance(result, str)

    def test_codec_key(self) -> None:
        assert _codec_key("h264") == "h264"
        assert _codec_key("nvv4l2h264enc") == "h264"
        assert _codec_key("h265") == "h265"
        assert _codec_key("hevc") == "h265"

    @pytest.mark.skipif(not _HAS_NV, reason="Jetson nv elements not available")
    def test_resolve_encoder_h264(self) -> None:
        assert resolve_encoder("h264") == "nvv4l2h264enc"

    @pytest.mark.skipif(not _HAS_NV, reason="Jetson nv elements not available")
    def test_resolve_encoder_h265(self) -> None:
        assert resolve_encoder("h265") == "nvv4l2h265enc"

    def test_resolve_encoder_unsupported_codec(self) -> None:
        with pytest.raises(VideoEncoderError):
            resolve_encoder("vp9")

    @pytest.mark.skipif(not _HAS_NV, reason="Jetson nv elements not available")
    def test_check_encoder(self) -> None:
        assert check_encoder("nvv4l2h264enc") is True
        assert check_encoder("libx264") is False


# ---------------------------------------------------------------------------
# Backend decode helpers (no hardware needed)
# ---------------------------------------------------------------------------


class _FakeImageMsg:
    def __init__(self, data: bytes, **kw: object) -> None:
        self.data = data
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeChannel:
    def __init__(self, topic: str) -> None:
        self.topic = topic


class _FakeDecoded:
    def __init__(self, decoded: object, topic: str) -> None:
        self.decoded_message = decoded
        self.channel = _FakeChannel(topic)


class TestBackendDecode:
    def test_decode_compressed_probes_dims(self) -> None:
        backend = GStreamerCompressionBackend()
        data, w, h = backend.decode_compressed(PROBE_JPEG)
        assert data == PROBE_JPEG
        assert (w, h) == (32, 32)

    def test_decode_image_jpeg_sets_no_pix_fmt(self) -> None:
        backend = GStreamerCompressionBackend()
        msg = _FakeDecoded(_FakeImageMsg(PROBE_JPEG), "/cam")
        data, w, h = backend.decode_image(msg, "sensor_msgs/CompressedImage")
        assert data == PROBE_JPEG
        assert (w, h) == (32, 32)
        assert backend.get_pix_fmt("/cam") is None

    def test_decode_image_raw_sets_pix_fmt(self) -> None:
        backend = GStreamerCompressionBackend()
        raw = _FakeImageMsg(b"\x00" * (4 * 4 * 3), width=4, height=4, encoding="rgb8")
        msg = _FakeDecoded(raw, "/raw")
        _data, w, h = backend.decode_image(msg, "sensor_msgs/Image")
        assert (w, h) == (4, 4)
        assert backend.get_pix_fmt("/raw") == "rgb24"

    def test_decode_image_unsupported_raw_encoding_raises(self) -> None:
        backend = GStreamerCompressionBackend()
        raw = _FakeImageMsg(b"\x00" * 48, width=4, height=4, encoding="yuv422")
        msg = _FakeDecoded(raw, "/raw")
        with pytest.raises(VideoEncoderError):
            backend.decode_image(msg, "sensor_msgs/Image")


# ---------------------------------------------------------------------------
# Backend selection (no hardware needed — probe monkeypatched)
# ---------------------------------------------------------------------------


class TestBackendSelection:
    @pytest.mark.skipif(not _HW_OK, reason="Jetson hardware pipeline not healthy")
    def test_explicit_gstreamer_mode(self) -> None:
        # Explicit mode runs the liveness probe, then returns the backend.
        backend = create_video_compression_backend(EncoderMode.GSTREAMER, "h264", do_video=True)
        assert backend.label == "gstreamer"

    def test_explicit_gstreamer_raises_when_pipeline_unhealthy(self, monkeypatch) -> None:
        # A codec stack that hangs / produces nothing probes False → explicit
        # mode raises a clear error rather than proceeding into a hang.
        monkeypatch.setattr(gstreamer_module, "probe_hw_jpeg_pipeline", lambda *_a, **_k: False)
        with pytest.raises(VideoEncoderError):
            create_video_compression_backend(EncoderMode.GSTREAMER, "h264", do_video=True)

    def test_auto_never_selects_gstreamer(self, monkeypatch) -> None:
        # Jetson is opt-in only (its nvjpegdec colour handling is not faithful on
        # all inputs), so AUTO must never return it — even where gst is present.
        monkeypatch.setattr(
            compression_module._PyAVCompressionBackend, "resolve_encoder", lambda *_: "libx264"
        )
        monkeypatch.setattr(
            compression_module._FfmpegCliCompressionBackend, "resolve_encoder", lambda *_: "libx264"
        )
        backend = create_video_compression_backend(EncoderMode.AUTO, "h264", do_video=True)
        assert backend.label != "gstreamer"


# ---------------------------------------------------------------------------
# Real hardware encode (skipped unless the probe confirms a working stack)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HW_OK, reason="Jetson hardware JPEG/encode pipeline not healthy")
class TestGStreamerHardwareEncode:
    def test_probe_healthy(self) -> None:
        assert probe_hw_jpeg_pipeline("h264") is True

    def test_one_au_per_frame(self) -> None:
        """Every JPEG fed yields exactly one access unit, incl. GOP boundaries."""
        backend = GStreamerCompressionBackend()
        encoder = backend.create_encoder(256, 256, "nvv4l2h264enc", 28, scale=(256, 256))
        n = 70  # spans multiple 30-frame GOPs
        outputs: list[bytes] = []
        try:
            for _ in range(n):
                result = encoder.encode(PROBE_JPEG)
                if result is not None:
                    outputs.append(result)
            outputs.extend(encoder.flush_packets())
        finally:
            encoder.close()
        assert len(outputs) == n
        for packet in outputs:
            assert packet.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01"))

    def test_h265_encode(self) -> None:
        backend = GStreamerCompressionBackend()
        encoder = backend.create_encoder(256, 256, "nvv4l2h265enc", 28, scale=(256, 256))
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

    def test_close_is_idempotent(self) -> None:
        backend = GStreamerCompressionBackend()
        encoder = backend.create_encoder(256, 256, "nvv4l2h264enc", 28, scale=(256, 256))
        encoder.close()
        encoder.close()
