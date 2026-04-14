"""Tests for VideoEncoder resource management."""

import pytest
from pymcap_cli.encoding.video_pyav import VideoEncoder

av = pytest.importorskip("av")


def _make_encoder():

    return VideoEncoder(width=16, height=16, codec_name="libx264")


@pytest.fixture
def encoder():
    enc = _make_encoder()
    yield enc
    # Belt-and-suspenders: close in case a test forgot to
    enc.close()


class TestVideoEncoderClose:
    def test_close_releases_context(self, encoder):
        encoder.close()
        assert not hasattr(encoder, "_context")

    def test_close_is_idempotent(self, encoder):
        encoder.close()
        encoder.close()  # should not raise

    def test_context_manager_closes_on_exit(self):
        with _make_encoder() as enc:
            assert hasattr(enc, "_context")
        assert not hasattr(enc, "_context")

    def test_context_manager_returns_encoder(self):
        enc = _make_encoder()
        with enc as ctx:
            assert ctx is enc
        enc.close()

    def test_context_manager_closes_on_exception(self):
        enc = _make_encoder()
        with pytest.raises(RuntimeError), enc:
            raise RuntimeError("simulated error")
        assert not hasattr(enc, "_context")
