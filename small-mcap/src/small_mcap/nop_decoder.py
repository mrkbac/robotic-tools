"""NOP decoder factory for small-mcap."""

from collections.abc import Callable
from typing import Any


class RuntimeDecoderNotFoundError(RuntimeError):
    """Exception raised by the NOP decoder when decoding is attempted."""


def _nop_decoder(_: bytes | memoryview) -> Any:
    """Decoder function that raises a RuntimeDecoderNotFoundError when called."""
    raise RuntimeDecoderNotFoundError("NOP decoder does not support decoding messages.")


class NOPDecoderFactory:
    """Decoder factory that raises RuntimeDecoderNotFoundError for any encoding.

    Useful as a fallback when messages are only copied as raw bytes, never decoded.
    """

    def decoder_for(
        self,
        message_encoding: str,  # noqa: ARG002
        schema: Any | None,  # noqa: ARG002
    ) -> Callable[[bytes | memoryview], Any] | None:
        """Return a decoder that raises a RuntimeDecoderNotFoundError for any message encoding."""
        return _nop_decoder
