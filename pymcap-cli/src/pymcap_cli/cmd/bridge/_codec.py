"""Shared ROS2/JSON payload codec + request-argument parsing for bridge commands.

`call` and `pub` both turn ``field:=value`` CLI arguments into an encoded message and
(for `call`) decode a response, dispatching on the wire encoding advertised by the
bridge. That logic lives here so the command modules stay focused on their own UX.
"""

from __future__ import annotations

import dataclasses
import json

from mcap_ros2_support_fast import ROS2EncoderFactory
from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap import Schema

PayloadValue = str | int | float | bool | None | list["PayloadValue"] | dict[str, "PayloadValue"]

CDR_ENCODINGS = frozenset({"cdr", "ros2"})


class CodecError(RuntimeError):
    """Raised when a payload cannot be encoded/decoded for the advertised encoding."""


class FieldSyntaxError(ValueError):
    """Raised when a ``field:=value`` request argument is malformed."""


def parse_field_args(fields: list[str]) -> dict[str, PayloadValue]:
    """Parse ROS2-style ``field:=value`` tokens into a request dict.

    Each value is parsed as JSON (so ``data:=true`` → ``True``, ``n:=3`` → ``3``,
    ``xs:=[1,2]`` → ``[1, 2]``), falling back to the raw string when it is not valid
    JSON (``name:=hello`` → ``"hello"``).
    """
    request: dict[str, PayloadValue] = {}
    for token in fields:
        key, sep, raw = token.partition(":=")
        if not sep or not key:
            raise FieldSyntaxError(f"Invalid request argument {token!r}; expected 'field:=value'")
        try:
            value: PayloadValue = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        request[key] = value
    return request


def encode_message(
    *,
    encoding: str,
    schema_name: str,
    schema_encoding: str,
    schema_text: str,
    value: dict[str, PayloadValue],
) -> bytes:
    """Encode ``value`` to bytes for the given wire ``encoding``."""
    if encoding in CDR_ENCODINGS:
        schema = Schema(
            id=1, name=schema_name, encoding=schema_encoding or "ros2msg", data=schema_text.encode()
        )
        encoder = ROS2EncoderFactory().encoder_for(schema)
        if encoder is None:
            raise CodecError(f"No encoder for schema {schema_name}")
        return bytes(encoder(value))
    if encoding == "json":
        return json.dumps(value).encode()
    raise CodecError(f"Unsupported encoding: {encoding!r}")


def decode_message(
    *,
    encoding: str,
    schema_name: str,
    schema_encoding: str,
    schema_text: str,
    payload: bytes,
) -> dict[str, PayloadValue]:
    """Decode ``payload`` produced with the given wire ``encoding`` into a dict."""
    if encoding in CDR_ENCODINGS:
        schema = Schema(
            id=2, name=schema_name, encoding=schema_encoding or "ros2msg", data=schema_text.encode()
        )
        decoder = DecoderFactory().decoder_for("cdr", schema)
        if decoder is None:
            raise CodecError(f"No decoder for schema {schema_name}")
        return _decoded_to_dict(decoder(payload))
    if encoding == "json":
        return json.loads(payload) if payload else {}
    raise CodecError(f"Unsupported encoding: {encoding!r}")


def _decoded_to_dict(message: object) -> dict[str, PayloadValue]:
    if dataclasses.is_dataclass(message) and not isinstance(message, type):
        return dataclasses.asdict(message)
    raise CodecError("Decoded payload was not a message object")
