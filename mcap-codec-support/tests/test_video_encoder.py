"""Tests for VideoEncoder resource management."""

import pytest
from mcap_codec_support.video.pyav import VideoEncoder

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
