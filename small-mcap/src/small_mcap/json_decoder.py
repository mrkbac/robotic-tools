"""JSON decoder factory for small-mcap."""

import json
from collections.abc import Callable
from typing import Any


class JSONDecoderFactory:
    """Decoder factory for JSON-encoded messages.

    Handles messages with message_encoding="json" by parsing the raw bytes as JSON.
    """

    def decoder_for(
        self, message_encoding: str, _schema: Any | None
    ) -> Callable[[bytes], Any] | None:
        """Return a JSON decoder if message encoding is 'json', otherwise None."""
        if message_encoding.lower() == "json":
            return self._decode_json
        return None

    @staticmethod
    def _decode_json(data: bytes) -> Any:
        """Decode JSON data from bytes."""
        return json.loads(data)
