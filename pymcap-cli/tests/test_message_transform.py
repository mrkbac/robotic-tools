"""Tests for the MessageTransformProcessor base (decode → transform → re-encode).

Covers the two shapes the base supports — value edits (same container, reuse
input channel) and transcode (new schema/topic, register a new channel) — plus
dropping and passthrough of unmatched channels. Uses real ``std_msgs/msg/String``
CDR so the actual decoder/encoder round-trip is exercised, not a stub.
"""
# ruff: noqa: ARG002  # example subclasses match the base's override signature

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cmd._run_processor import run_processor
from pymcap_cli.cmd._run_processor_multi import run_processor_multi
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.core.processors.base import (
    InputProcessor,
    MessageContext,
    MessageWithContext,
    OutputRouter,
)
from pymcap_cli.core.processors.message_transform import (
    MessageTransformProcessor,
    TransformOutput,
)
from small_mcap import McapWriter, read_message_decoded

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from small_mcap import Channel, Message, Schema

_STRING_SCHEMA = b"string data"
# A distinct output schema for the transcode shape: single string field.
_SHOUT_SCHEMA_NAME = "test_msgs/msg/Shout"
_SHOUT_SCHEMA = b"string text"


def _write_strings(path: Path, topic_values: list[tuple[str, str]]) -> None:
    """Write std_msgs/String messages: list of (topic, value), 1ms apart."""
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=4096, encoder_factory=ROS2EncoderFactory())
    writer.start()
    topics = sorted({t for t, _ in topic_values})
    writer.add_schema(1, "std_msgs/msg/String", "ros2msg", _STRING_SCHEMA)
    channel_ids: dict[str, int] = {}
    for i, topic in enumerate(topics, start=1):
        writer.add_channel(i, topic, "cdr", 1)
        channel_ids[topic] = i
    for i, (topic, value) in enumerate(topic_values):
        writer.add_message_encode(
            channel_id=channel_ids[topic],
            log_time=i + 1,
            publish_time=i + 1,
            data={"data": value},
        )
    writer.finish()
    path.write_bytes(buf.getvalue())


def _run(input_path: Path, output_path: Path, processors: list[MessageTransformProcessor]) -> None:
    run_processor(
        files=[str(input_path)],
        output=output_path,
        input_options=InputOptions.from_args(extra_processors=list(processors)),
        output_options=OutputOptions(overwrite_policy=OverwriteCollisionPolicy.OVERWRITE),
    )


def _read(path: Path) -> list[tuple[str, str, str]]:
    """Return (topic, schema_name, decoded string field) for each message."""
    out: list[tuple[str, str, str]] = []
    with path.open("rb") as f:
        for msg in read_message_decoded(f, decoder_factories=[DecoderFactory()]):
            schema_name = msg.schema.name if msg.schema else ""
            decoded = msg.decoded_message
            # std_msgs/String -> .data ; Shout -> .text
            value = getattr(decoded, "data", None)
            if value is None:
                value = getattr(decoded, "text", "")
            out.append((msg.channel.topic, schema_name, value))
    return out


# --------------------------------------------------------------------------
# Example subclasses
# --------------------------------------------------------------------------


class _UppercaseStrings(MessageTransformProcessor):
    """Value edit: uppercase the ``data`` field, same schema + topic."""

    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        return schema is not None and schema.name == "std_msgs/msg/String"

    def transform(self, channel: Channel, schema: Schema, decoded: Any):
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=schema.name,
                schema_encoding=schema.encoding,
                schema_data=schema.data,
                data={"data": decoded.data.upper()},
            )
        ]


class _DropShortStrings(MessageTransformProcessor):
    """Drop String messages whose value is shorter than 3 chars."""

    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        return schema is not None and schema.name == "std_msgs/msg/String"

    def transform(self, channel: Channel, schema: Schema, decoded: Any):
        if len(decoded.data) < 3:
            return []  # drop
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=schema.name,
                schema_encoding=schema.encoding,
                schema_data=schema.data,
                data={"data": decoded.data},
            )
        ]


class _AppendBangStrings(MessageTransformProcessor):
    """Value edit: append ``!`` to the ``data`` field, same schema + topic."""

    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        return schema is not None and schema.name == "std_msgs/msg/String"

    def transform(self, channel: Channel, schema: Schema, decoded: Any):
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=schema.name,
                schema_encoding=schema.encoding,
                schema_data=schema.data,
                data={"data": f"{decoded.data}!"},
            )
        ]


class _StringToShout(MessageTransformProcessor):
    """Transcode: std_msgs/String{data} -> test_msgs/Shout{text}, same topic."""

    def matches(self, channel: Channel, schema: Schema | None) -> bool:
        return schema is not None and schema.name == "std_msgs/msg/String"

    def transform(self, channel: Channel, schema: Schema, decoded: Any):
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=_SHOUT_SCHEMA_NAME,
                schema_encoding="ros2msg",
                schema_data=_SHOUT_SCHEMA,
                data={"text": decoded.data + "!"},
            )
        ]


class _BufferUntilFinalize(InputProcessor):
    """Hold messages until finalize, preserving their original routing context."""

    def __init__(self) -> None:
        self._pending: list[MessageWithContext] = []

    def on_message(
        self, context: MessageContext, message: Message
    ) -> Iterable[Message | MessageWithContext]:
        self._pending.append(
            MessageWithContext(
                message=message,
                stream_id=context.input.stream_id,
                input_channel_id=context.input_channel_id,
            )
        )
        return ()

    def finalize(self) -> Iterable[MessageWithContext]:
        return tuple(self._pending)


class _RouteByInputStream(OutputRouter):
    def route_message(self, context: MessageContext, message: Message) -> tuple[int]:
        return (context.input.stream_id,)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_value_edit_same_schema_reuses_channel(tmp_path: Path):
    """Uppercasing (identical schema+topic) edits values and keeps one channel."""
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "hello"), ("/chat", "world")])

    _run(inp, out, [_UppercaseStrings()])

    result = _read(out)
    assert [(t, v) for t, _, v in result] == [("/chat", "HELLO"), ("/chat", "WORLD")]
    # Same schema throughout; no second channel spawned for the topic.
    assert {s for _, s, _ in result} == {"std_msgs/msg/String"}


def test_transform_can_drop_messages(tmp_path: Path):
    """Yielding nothing from transform drops the message."""
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "hi"), ("/chat", "hello"), ("/chat", "yo")])

    _run(inp, out, [_DropShortStrings()])

    result = _read(out)
    assert [v for _, _, v in result] == ["hello"]


def test_transcode_new_schema_registers_channel(tmp_path: Path):
    """String -> Shout registers a new schema/channel; output decodes correctly."""
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "hey"), ("/chat", "bye")])

    _run(inp, out, [_StringToShout()])

    result = _read(out)
    assert [(s, v) for _, s, v in result] == [
        (_SHOUT_SCHEMA_NAME, "hey!"),
        (_SHOUT_SCHEMA_NAME, "bye!"),
    ]
    # The original std_msgs/String schema is fully replaced (no message kept it).
    assert all(s == _SHOUT_SCHEMA_NAME for _, s, _ in result)


def test_unmatched_channel_passes_through(tmp_path: Path):
    """Channels the processor doesn't match are emitted unchanged."""
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "hello"), ("/other", "world")])

    # Only match /chat by overriding matches to a specific topic.
    class _OnlyChat(_UppercaseStrings):
        def matches(self, channel: Channel, schema: Schema | None) -> bool:
            return super().matches(channel, schema) and channel.topic == "/chat"

    _run(inp, out, [_OnlyChat()])

    result = {t: v for t, _, v in _read(out)}
    assert result == {"/chat": "HELLO", "/other": "world"}


def test_transform_none_passes_through_unchanged(tmp_path: Path):
    """Returning None from transform keeps the original message unchanged."""
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "keep"), ("/chat", "me")])

    class _DeclineAll(_UppercaseStrings):
        def transform(self, channel: Channel, schema: Schema, decoded: Any):
            return None  # decline — pass through raw

    _run(inp, out, [_DeclineAll()])

    assert [v for _, _, v in _read(out)] == ["keep", "me"]


def test_timestamps_preserved(tmp_path: Path):
    """Transform preserves log_time (critical for time-ordered read-back)."""
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "aaa"), ("/chat", "bbb"), ("/chat", "ccc")])

    _run(inp, out, [_UppercaseStrings()])

    with out.open("rb") as f:
        msgs = read_message_decoded(f, decoder_factories=[DecoderFactory()])
        times = [m.message.log_time for m in msgs]
    assert times == [1, 2, 3]


def test_adjacent_transforms_decode_each_message_once(tmp_path: Path, monkeypatch):
    """Composed decoded transforms share one decoded payload per message."""
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "hello"), ("/chat", "world")])

    decode_count = 0
    original_decoder_for = DecoderFactory.decoder_for

    def counting_decoder_for(self, message_encoding, schema):
        decoder = original_decoder_for(self, message_encoding, schema)
        if decoder is None or schema is None or schema.name != "std_msgs/msg/String":
            return decoder

        def counting_decoder(data):
            nonlocal decode_count
            decode_count += 1
            return decoder(data)

        return counting_decoder

    monkeypatch.setattr(DecoderFactory, "decoder_for", counting_decoder_for)

    _run(inp, out, [_UppercaseStrings(), _AppendBangStrings()])

    processing_decode_count = decode_count
    assert [value for _topic, _schema, value in _read(out)] == ["HELLO!", "WORLD!"]
    assert processing_decode_count == 2


def test_finalize_output_preserves_input_stream_context_for_routing(tmp_path: Path):
    """Buffered tail messages must route using the stream that produced them."""
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_strings(first, [("/chat", "from first")])
    _write_strings(second, [("/chat", "from second")])

    processor = _BufferUntilFinalize()
    run_processor_multi(
        files=[str(first), str(second)],
        input_options=InputOptions.from_args(extra_processors=[processor]),
        output_options=OutputOptions(
            routers=[_RouteByInputStream()],
            output_template=str(tmp_path / "stream_{key}.mcap"),
            overwrite_policy=OverwriteCollisionPolicy.OVERWRITE,
        ),
    )

    assert [value for _topic, _schema, value in _read(tmp_path / "stream_0.mcap")] == ["from first"]
    assert [value for _topic, _schema, value in _read(tmp_path / "stream_1.mcap")] == [
        "from second"
    ]


def test_processor_abort_runs_when_processing_is_interrupted(tmp_path: Path):
    inp, out = tmp_path / "in.mcap", tmp_path / "out.mcap"
    _write_strings(inp, [("/chat", "stop")])

    class _InterruptingProcessor(InputProcessor):
        def __init__(self) -> None:
            self.was_aborted = False

        def on_message(
            self, context: MessageContext, message: Message
        ) -> Iterable[Message | MessageWithContext]:
            raise KeyboardInterrupt

        def abort(self) -> None:
            self.was_aborted = True

    processor = _InterruptingProcessor()
    with pytest.raises(KeyboardInterrupt):
        run_processor(
            files=[str(inp)],
            output=out,
            input_options=InputOptions.from_args(extra_processors=[processor]),
            output_options=OutputOptions(overwrite_policy=OverwriteCollisionPolicy.OVERWRITE),
        )

    assert processor.was_aborted
