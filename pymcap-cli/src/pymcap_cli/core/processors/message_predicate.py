"""Filter decoded messages with topic-qualified MessagePath predicates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    Filter,
    MessagePath,
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory
from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    ChunkDecision,
    InputProcessor,
    MessageContext,
    PipelineContext,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ros_parser.message_path import MessagePathVariables
    from small_mcap import Channel, Chunk, LazyChunk, Message, Schema


def _result_matches(result: object) -> bool:
    if result is None:
        return False
    return not isinstance(result, (list, tuple)) or len(result) > 0


@dataclass(frozen=True, slots=True)
class MessagePredicate:
    source: str
    path: MessagePath


class MessagePredicateProcessor(InputProcessor):
    """Drop messages that match none of their topic's MessagePath predicates.

    Predicates for one topic are ORed. Topics without predicates pass through
    unchanged. Each path must contain a message-local filter expression; AND
    and OR within one predicate use MessagePath's ``&&`` and ``||`` operators.
    """

    def __init__(
        self,
        paths: list[str],
        *,
        variables: MessagePathVariables | None = None,
    ) -> None:
        grouped: dict[str, list[MessagePredicate]] = {}
        for source in paths:
            parsed = parse_message_path(source)
            if parsed.has_stream:
                raise ValueError(f"--where does not support stream modifiers (@@): {source!r}")
            if not any(isinstance(segment, Filter) for segment in parsed.segments):
                raise ValueError(f"--where path must contain a predicate filter: {source!r}")
            grouped.setdefault(parsed.topic, []).append(MessagePredicate(source, parsed))
        self.paths_by_topic = {topic: tuple(predicates) for topic, predicates in grouped.items()}
        self.variables = dict(variables or {})
        self._factories = (JSONDecoderFactory(), Ros2DecoderFactory())
        self._decoders: dict[tuple[int, int], Callable[[bytes | memoryview], object]] = {}
        self._target_channels: dict[int, set[int]] = {}
        self._known_channels: dict[int, set[int]] = {}
        self._topics_by_channel: dict[tuple[int, int], str] = {}
        self._registered_topics: set[str] = set()
        self._validated: set[tuple[int, int, str]] = set()

    @override
    def initialize(self, context: PipelineContext) -> None:
        for input_context in context.inputs:
            summary = input_context.summary
            if summary is None:
                continue
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id) if channel.schema_id else None
                self._register(input_context.stream_id, channel, schema)

    @override
    def on_channel(
        self,
        context: ChannelContext,
        channel: Channel,
        schema: Schema | None,
    ) -> Action:
        self._register(context.input.stream_id, channel, schema)
        return Action.CONTINUE

    def _register(self, stream_id: int, channel: Channel, schema: Schema | None) -> None:
        self._known_channels.setdefault(stream_id, set()).add(channel.id)
        paths = self.paths_by_topic.get(channel.topic)
        if paths is None:
            return
        self._registered_topics.add(channel.topic)
        if schema is None:
            raise ValueError(
                f"--where cannot decode topic {channel.topic!r}: channel has no schema"
            )
        decoder = None
        for factory in self._factories:
            decoder = factory.decoder_for(channel.message_encoding, schema)
            if decoder is not None:
                break
        if decoder is None:
            raise ValueError(
                f"--where cannot decode topic {channel.topic!r} with "
                f"{channel.message_encoding!r}/{schema.encoding!r}"
            )
        self._decoders[(stream_id, channel.id)] = decoder
        self._target_channels.setdefault(stream_id, set()).add(channel.id)
        self._topics_by_channel[(stream_id, channel.id)] = channel.topic
        for predicate in paths:
            key = (stream_id, schema.id, predicate.source)
            if key not in self._validated:
                self._validate_path(predicate, schema, channel.topic)
                self._validated.add(key)

    def _validate_path(
        self,
        predicate: MessagePredicate,
        schema: Schema,
        topic: str,
    ) -> None:
        try:
            definitions = parse_schema_to_definitions(schema.name, schema.data)
        except Exception:  # noqa: BLE001 - JSON schemas have no ROS definition
            return
        root = definitions.get(schema.name)
        if root is None:
            parts = schema.name.split("/")
            root = definitions.get(f"{parts[0]}/{parts[-1]}")
        if root is None:
            return
        try:
            predicate.path.validate(root, definitions)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid --where path {predicate.source!r} for "
                f"topic {topic!r} ({schema.name}): {exc}"
            ) from exc

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        del chunk
        indexes = context.message_indexes
        if indexes is None:
            return ChunkDecision.DECODE
        stream_id = context.input.stream_id
        known = self._known_channels.get(stream_id, set())
        targets = self._target_channels.get(stream_id, set())
        if any(index.channel_id not in known or index.channel_id in targets for index in indexes):
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        decoder = self._decoders.get((context.input.stream_id, message.channel_id))
        if decoder is None:
            yield message
            return
        decoded = decoder(message.data)
        topic = self._topics_by_channel[(context.input.stream_id, message.channel_id)]
        try:
            matched = any(
                _result_matches(predicate.path.apply(decoded, self.variables))
                for predicate in self.paths_by_topic[topic]
            )
        except MessagePathError as exc:
            raise MessagePathError(f"--where on {topic!r}: {exc}") from exc
        if matched:
            yield message

    @override
    def finalize(self) -> Iterable[Message]:
        missing = set(self.paths_by_topic) - self._registered_topics
        if missing:
            topics = ", ".join(repr(topic) for topic in sorted(missing))
            raise ValueError(f"--where topic not found: {topics}")
        return ()
