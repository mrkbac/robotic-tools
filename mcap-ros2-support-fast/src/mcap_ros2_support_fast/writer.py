from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from ._planner import serialize_dynamic

if TYPE_CHECKING:
    from ._plans import EncoderFunction


class Schema(Protocol):
    id: int
    name: str
    encoding: str
    data: bytes


class McapROS2WriteError(Exception):
    """Raised if a ROS2 message cannot be encoded to CDR with a given schema."""


class ROS2EncoderFactory:
    """
    Encoder factory for ROS2 messages that implements EncoderFactoryProtocol.
    Caches encoders by schema identity for efficient repeated encoding.
    """

    profile = "ros2"
    encoding = "ros2msg"  # Schema encoding format
    message_encoding = "cdr"  # Message data encoding format

    def __init__(self) -> None:
        self._encoders: dict[tuple[str, str, bytes], EncoderFunction] = {}

    @staticmethod
    def _cache_key(schema: Schema) -> tuple[str, str, bytes]:
        return (schema.name, schema.encoding, schema.data)

    def encoder_for(self, schema: Schema | None) -> Callable[[Any], bytes | memoryview] | None:
        """
        Get an encoder function for the given schema.

        :param schema: The schema to get an encoder for
        :return: A function that encodes a message object to bytes, or None if schema is None
        """
        if schema is None:
            return None

        cache_key = self._cache_key(schema)
        encoder = self._encoders.get(cache_key)
        if encoder is not None:
            return encoder

        # Validate schema encoding
        if schema.encoding != "ros2msg":
            raise McapROS2WriteError(f'can\'t parse schema with encoding "{schema.encoding}"')

        # Create and cache encoder
        encoder = serialize_dynamic(schema.name, schema.data.decode())
        self._encoders[cache_key] = encoder
        return encoder
