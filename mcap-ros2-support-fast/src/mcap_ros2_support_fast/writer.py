from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from . import __version__
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


def _library_identifier() -> str:
    # TODO: readd small-mcap version
    return f"mcap-ros2-support-fast {__version__}; small-mcap"


class ROS2EncoderFactory:
    """
    Encoder factory for ROS2 messages that implements EncoderFactoryProtocol.
    Caches encoders by schema ID for efficient repeated encoding.
    """

    profile = "ros2"
    encoding = "ros2msg"  # Schema encoding format
    message_encoding = "cdr"  # Message data encoding format

    def __init__(self) -> None:
        self._encoders: dict[int, EncoderFunction] = {}
        self.library = _library_identifier()

    def encoder_for(self, schema: Schema | None) -> Callable[[Any], bytes | memoryview] | None:
        """
        Get an encoder function for the given schema.

        :param schema: The schema to get an encoder for
        :return: A function that encodes a message object to bytes, or None if schema is None
        """
        if schema is None:
            return None

        # Check cache
        if schema.id in self._encoders:
            return self._encoders[schema.id]

        # Validate schema encoding
        if schema.encoding != "ros2msg":
            raise McapROS2WriteError(f'can\'t parse schema with encoding "{schema.encoding}"')

        # Create and cache encoder
        encoder = serialize_dynamic(schema.name, schema.data.decode())
        self._encoders[schema.id] = encoder

        return encoder
