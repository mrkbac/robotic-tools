"""Decoder class for decoding ROS2 messages from MCAP files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from mcap_ros2_support_fast._dynamic_decoder import create_decoder

from ._planner import generate_dynamic

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._plans import DecoderFunction

_SCHEMA_ENCODING_ROS2 = "ros2msg"
_MESSAGE_ENCODING_CDR = "cdr"


class _SchemaProtocol(Protocol):
    id: int
    name: str
    encoding: str
    data: bytes


class McapROS2DecodeError(Exception):
    """Raised if a MCAP message record cannot be decoded as a ROS2 message."""


class DecoderFactory:
    """Provides functionality to an :py:class:`~mcap.reader.McapReader` to decode CDR-encoded
    messages. Requires valid `ros2msg` schema to decode messages. Schemas written in IDL are not
    currently supported.
    """

    def __init__(self) -> None:
        self._decoders: dict[int, DecoderFunction] = {}

    def decoder_for(
        self, message_encoding: str, schema: _SchemaProtocol | None
    ) -> Callable[[bytes], Any] | None:
        if (
            message_encoding != _MESSAGE_ENCODING_CDR
            or schema is None
            or schema.encoding != _SCHEMA_ENCODING_ROS2
        ):
            return None

        decoder = self._decoders.get(schema.id)
        if decoder is None:
            decoder = generate_dynamic(schema.name, schema.data.decode(), parser=create_decoder)
            self._decoders[schema.id] = decoder
        return decoder
