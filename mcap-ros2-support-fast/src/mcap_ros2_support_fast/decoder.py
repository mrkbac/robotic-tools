"""Decoder class for decoding ROS2 messages from MCAP files."""

import warnings
from collections.abc import Callable
from typing import Any

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.exceptions import McapError
from mcap.records import Message, Schema
from mcap.well_known import MessageEncoding, SchemaEncoding

from mcap_ros2_support_fast._dynamic_codegen import create_decoder

from ._planner import generate_dynamic
from ._plans import DecodedMessage, DecoderFunction, PlanList


class McapROS2DecodeError(McapError):
    """Raised if a MCAP message record cannot be decoded as a ROS2 message."""


class DecoderFactory(McapDecoderFactory):
    """Provides functionality to an :py:class:`~mcap.reader.McapReader` to decode CDR-encoded
    messages. Requires valid `ros2msg` schema to decode messages. Schemas written in IDL are not
    currently supported.
    """

    def __init__(self, parser: Callable[[PlanList], DecoderFunction]) -> None:
        self._decoders: dict[int, DecoderFunction] = {}
        self._parser = parser

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
            decoder = generate_dynamic(schema.name, schema.data.decode(), parser=self._parser)
            self._decoders[schema.id] = decoder
        return decoder


class Decoder:
    """Decodes ROS 2 messages.

    .. deprecated:: 0.5.0
      Use :py:class:`~mcap_ros2.decoder.DecoderFactory` with :py:class:`~mcap.reader.McapReader`
      instead.
    """

    def __init__(self) -> None:
        warnings.warn(
            """The `mcap_ros2.decoder.Decoder` class is deprecated.
For similar functionality, instantiate the `mcap.reader.McapReader` with a
`mcap_ros2.decoder.DecoderFactory` instance.""",
            DeprecationWarning,
            stacklevel=2,
        )
        self._decoder_factory = DecoderFactory(create_decoder)

    def decode(self, schema: Schema, message: Message) -> Any:
        decoder = self._decoder_factory.decoder_for(MessageEncoding.CDR, schema)
        assert decoder is not None
        return decoder(message.data)
