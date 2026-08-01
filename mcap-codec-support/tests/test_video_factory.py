"""Tests for channel-aware CompressedVideo decoder routing."""

from types import SimpleNamespace

from mcap_codec_support.video import EncoderMode, VideoDecompressFactory
from mcap_codec_support.video.common import DecompressedFrame
from mcap_codec_support.video.schemas import COMPRESSED_VIDEO_SCHEMA


def _schema() -> SimpleNamespace:
    return SimpleNamespace(name=COMPRESSED_VIDEO_SCHEMA)


def _channel(channel_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=channel_id)


def test_decoder_for_returns_none_when_cdr_schema_decoder_is_missing(monkeypatch) -> None:
    factory = VideoDecompressFactory(backend=EncoderMode.PYAV)
    monkeypatch.setattr(factory._cdr_factory, "decoder_for", lambda *_: None)

    assert factory.decoder_for("cdr", _schema(), _channel(1)) is None


def test_decoder_for_returns_none_when_video_packet_is_buffered(monkeypatch) -> None:
    factory = VideoDecompressFactory(backend=EncoderMode.PYAV)
    decoded = SimpleNamespace(
        format="h264",
        data=memoryview(b"packet"),
        timestamp=SimpleNamespace(sec=4, nanosec=5),
        frame_id="camera",
    )
    monkeypatch.setattr(factory._cdr_factory, "decoder_for", lambda *_: lambda _: decoded)

    class FakeDecompressor:
        def decompress(self, data: bytes, codec: str) -> DecompressedFrame | None:
            assert data == b"packet"
            assert codec == "h264"
            return None

        def flush(self) -> list[DecompressedFrame]:
            return []

    monkeypatch.setattr(factory, "_get_decompressor", lambda _: FakeDecompressor())
    decoder = factory.decoder_for("cdr", _schema(), _channel(1))

    assert decoder is not None
    assert decoder(b"cdr payload") is None
