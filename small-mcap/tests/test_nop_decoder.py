import pytest
from small_mcap.nop_decoder import NOPDecoderFactory, RuntimeDecoderNotFoundError


def test_nop_decoder_rejects_decoding() -> None:
    decoder = NOPDecoderFactory().decoder_for("cdr", None)

    with pytest.raises(RuntimeDecoderNotFoundError, match="does not support decoding"):
        decoder(b"payload")
