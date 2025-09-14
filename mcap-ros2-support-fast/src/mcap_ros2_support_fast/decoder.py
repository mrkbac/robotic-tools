"""Decoder class for decoding ROS2 messages from MCAP files."""

from collections.abc import Callable

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.exceptions import McapError
from mcap.records import Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

from mcap_ros2_support_fast._dynamic_codegen import create_decoder

from ._planner import generate_dynamic
from ._plans import DecodedMessage, DecoderFunction


class McapROS2DecodeError(McapError):
    """Raised if a MCAP message record cannot be decoded as a ROS2 message."""


class DecoderFactory(McapDecoderFactory):
    """Provides functionality to an :py:class:`~mcap.reader.McapReader` to decode CDR-encoded
    messages. Requires valid `ros2msg` schema to decode messages. Schemas written in IDL are not
    currently supported.
    """

    def __init__(self) -> None:
        self._decoders: dict[int, DecoderFunction] = {}

    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes], DecodedMessage] | None:
        if (
            message_encoding != MessageEncoding.CDR
            or schema is None
            or schema.encoding != SchemaEncoding.ROS2
        ):
            return None

        decoder = self._decoders.get(schema.id)
        if decoder is None:
            decoder = generate_dynamic(schema.name, schema.data.decode(), parser=create_decoder)
            self._decoders[schema.id] = decoder
        return decoder
