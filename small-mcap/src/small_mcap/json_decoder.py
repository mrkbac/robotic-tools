"""JSON decoder factory for small-mcap."""

import json
from collections.abc import Callable
from typing import Any


def _decode_json(data: bytes | memoryview) -> Any:
    """Decode JSON data from bytes or memoryview."""
    # Convert memoryview to bytes for json.loads type compatibility
    return json.loads(data if isinstance(data, bytes) else bytes(data))


class JSONDecoderFactory:
    """Decoder factory for JSON-encoded messages.

    Handles messages with message_encoding="json" by parsing the raw bytes as JSON.
    """

    def decoder_for(
        self,
        message_encoding: str,
        schema: Any | None,  # noqa: ARG002
    ) -> Callable[[bytes | memoryview], Any] | None:
        """Return a JSON decoder if message encoding is 'json', otherwise None."""
        if message_encoding.lower() == "json":
            return _decode_json
        return None


def _encode_json(message: Any) -> bytes:
    """Encode a message to JSON bytes."""
    return json.dumps(message).encode()


class JSONEncoderFactory:
    """Encoder factory for JSON-encoded messages.

    Handles messages with message_encoding="json" by serializing the message to JSON bytes.
    Compatible with McapWriter's EncoderFactoryProtocol.
    """

    profile: str = ""
    encoding: str = "jsonschema"
    message_encoding: str = "json"

    def encoder_for(
        self,
        schema: Any | None,  # noqa: ARG002
    ) -> Callable[[Any], bytes] | None:
        """Return a JSON encoder (schema is ignored for JSON encoding)."""
        return _encode_json
