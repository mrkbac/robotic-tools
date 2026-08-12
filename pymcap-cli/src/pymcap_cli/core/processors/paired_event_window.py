"""Discovery and replay routing for paired MessagePath event windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from ros_parser.message_path import NO_OUTPUT, Filter
from small_mcap import Channel, JSONDecoderFactory, Message, Schema, stream_reader
from typing_extensions import override

from pymcap_cli.core.named_message_path import CatQueryRuntime, parse_cat_queries
from pymcap_cli.core.processors.base import (
    ChannelContext,
    ChunkContext,
    ChunkDecision,
    MessageContext,
    OutputRouter,
    OutputSegmentInfo,
    PipelineContext,
    _SplitRequiredSentinel,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import IO

    from small_mcap import Chunk, LazyChunk, MessageIndex

    from pymcap_cli.core.processors.base import RouteKey, TemplateValue

MalformedEventPolicy: TypeAlias = Literal["error", "ignore", "drop"]
InvalidWindowPolicy: TypeAlias = Literal["error", "drop"]
BoundaryKind: TypeAlias = Literal["start", "stop"]

_START_LABEL = "window_start"
_STOP_LABEL = "window_stop"


@dataclass(frozen=True, slots=True)
class BoundaryEvent:
    kind: BoundaryKind
    log_time: int
    topic: str


@dataclass(frozen=True, slots=True)
class WindowInfo:
    key: int
    start_time: int
    end_time: int


@dataclass(frozen=True, slots=True)
class BoundaryReplayStep:
    event: BoundaryEvent
    route_key: int | None
    active_after: int | None


@dataclass(frozen=True, slots=True)
class PairedWindowPlan:
    windows: tuple[WindowInfo, ...]
    steps: tuple[BoundaryReplayStep, ...]


@dataclass(slots=True)
class _MutableStep:
    event: BoundaryEvent
    route_key: int | None = None
    active_after: int | None = None


@dataclass(slots=True)
class _OpenWindow:
    start_time: int
    step_indexes: list[int]


class BoundaryMatcher:
    """Evaluate two named absolute MessagePaths with one shared query runtime."""

    def __init__(self, start_expression: str, stop_expression: str) -> None:
        self.queries = parse_cat_queries(
            [
                f"{_START_LABEL}={start_expression}",
                f"{_STOP_LABEL}={stop_expression}",
            ]
        )
        for topic_queries in self.queries.values():
            for query in topic_queries:
                if query.path.has_stream:
                    raise ValueError("paired window expressions do not support stream modifiers")
        self.runtime = CatQueryRuntime(self.queries)
        self.topics = frozenset(self.queries)
        self._is_predicate = {
            query.output_name: any(isinstance(segment, Filter) for segment in query.path.segments)
            for topic_queries in self.queries.values()
            for query in topic_queries
        }

    def match(
        self,
        topic: str,
        decoded_message: object,
        log_time: int,
        publish_time: int,
    ) -> tuple[BoundaryKind, ...]:
        if topic not in self.queries:
            return ()
        variables = {"log_time_ns": log_time, "publish_time_ns": publish_time}
        result: object = self.runtime.evaluate(topic, decoded_message, log_time, variables)
        topic_queries = self.queries[topic]
        values: dict[str, object]
        if len(topic_queries) == 1:
            values = {topic_queries[0].output_name: result}
        elif result is NO_OUTPUT:
            values = {}
        elif isinstance(result, dict):
            values = cast("dict[str, object]", result)
        else:
            raise TypeError("paired MessagePath projection did not return a mapping")
        matches: list[BoundaryKind] = []
        if self._matches(_START_LABEL, values.get(_START_LABEL, NO_OUTPUT)):
            matches.append("start")
        if self._matches(_STOP_LABEL, values.get(_STOP_LABEL, NO_OUTPUT)):
            matches.append("stop")
        return tuple(matches)

    def _matches(self, label: str, value: object) -> bool:
        if value is NO_OUTPUT:
            return False
        if self._is_predicate[label]:
            return True
        if type(value) is not bool:
            raise ValueError(
                f"paired window primitive expression {label!r} must evaluate to true or false; "
                f"got {type(value).__name__}"
            )
        return value is True


def discover_paired_windows(
    path: Path,
    start_expression: str,
    stop_expression: str,
    *,
    minimum_duration_ns: int | None = None,
    maximum_duration_ns: int | None = None,
    orphan_stop: MalformedEventPolicy = "error",
    nested_start: MalformedEventPolicy = "error",
    unclosed_window: MalformedEventPolicy = "error",
    invalid_window: InvalidWindowPolicy = "error",
) -> PairedWindowPlan:
    """Discover and validate every boundary before any output is opened."""
    matcher = BoundaryMatcher(start_expression, stop_expression)
    events: list[BoundaryEvent] = []
    with path.open("rb") as stream:
        for topic, message, decoded in _iter_event_messages(stream, matcher.topics):
            events.extend(
                BoundaryEvent(kind, message.log_time, topic)
                for kind in matcher.match(
                    topic,
                    decoded,
                    message.log_time,
                    message.publish_time,
                )
            )
    return _pair_events(
        events,
        minimum_duration_ns=minimum_duration_ns,
        maximum_duration_ns=maximum_duration_ns,
        orphan_stop=orphan_stop,
        nested_start=nested_start,
        unclosed_window=unclosed_window,
        invalid_window=invalid_window,
    )


def _iter_event_messages(
    stream: IO[bytes],
    topics: frozenset[str],
) -> Iterator[tuple[str, Message, object]]:
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    decoders: dict[int, Callable[[bytes | memoryview], object]] = {}
    factories = (JSONDecoderFactory(), Ros2DecoderFactory())
    for record in stream_reader(stream):
        if isinstance(record, Schema):
            schemas[record.id] = record
            continue
        if isinstance(record, Channel):
            channels[record.id] = record
            continue
        if not isinstance(record, Message):
            continue
        channel = channels.get(record.channel_id)
        if channel is None:
            raise ValueError(f"event message references unknown channel {record.channel_id}")
        if channel.topic not in topics:
            continue
        decoder = decoders.get(channel.id)
        if decoder is None:
            schema = schemas.get(channel.schema_id)
            for factory in factories:
                decoder = factory.decoder_for(channel.message_encoding, schema)
                if decoder is not None:
                    break
            if decoder is None:
                raise ValueError(
                    f"no decoder for paired event topic {channel.topic!r} "
                    f"with encoding {channel.message_encoding!r}"
                )
            decoders[channel.id] = decoder
        yield channel.topic, record, decoder(record.data)


def _pair_events(
    events: list[BoundaryEvent],
    *,
    minimum_duration_ns: int | None,
    maximum_duration_ns: int | None,
    orphan_stop: MalformedEventPolicy,
    nested_start: MalformedEventPolicy,
    unclosed_window: MalformedEventPolicy,
    invalid_window: InvalidWindowPolicy,
) -> PairedWindowPlan:
    steps: list[_MutableStep] = []
    windows: list[WindowInfo] = []
    opened: _OpenWindow | None = None
    for event in events:
        step_index = len(steps)
        steps.append(_MutableStep(event))
        if event.kind == "start":
            if opened is None:
                opened = _OpenWindow(event.log_time, [step_index])
            elif nested_start == "error":
                raise ValueError(f"nested window start at {event.log_time}")
            elif nested_start == "ignore":
                opened.step_indexes.append(step_index)
            else:
                opened = _OpenWindow(event.log_time, [step_index])
            continue

        if opened is None:
            if orphan_stop == "error":
                raise ValueError(f"orphan window stop at {event.log_time}")
            continue
        opened.step_indexes.append(step_index)
        duration_ns = event.log_time - opened.start_time
        is_invalid = (
            duration_ns < 0
            or (minimum_duration_ns is not None and duration_ns < minimum_duration_ns)
            or (maximum_duration_ns is not None and duration_ns > maximum_duration_ns)
        )
        if is_invalid:
            if invalid_window == "error":
                raise ValueError(
                    f"window {opened.start_time}..{event.log_time} has invalid "
                    f"duration {duration_ns}ns"
                )
        else:
            key = len(windows)
            windows.append(WindowInfo(key, opened.start_time, event.log_time))
            for index in opened.step_indexes:
                steps[index].route_key = key
                steps[index].active_after = key
            steps[step_index].active_after = None
        opened = None

    if opened is not None and unclosed_window == "error":
        raise ValueError(f"unclosed window start at {opened.start_time}")
    return PairedWindowPlan(
        tuple(windows),
        tuple(BoundaryReplayStep(step.event, step.route_key, step.active_after) for step in steps),
    )


class PairedEventWindowProcessor(OutputRouter):
    """Replay a discovered boundary plan while routing the unchanged source."""

    def __init__(self, start_expression: str, stop_expression: str, plan: PairedWindowPlan) -> None:
        self.matcher = BoundaryMatcher(start_expression, stop_expression)
        self.plan = plan
        self.channels: dict[int, Channel] = {}
        self._decoders: dict[int, Callable[[bytes | memoryview], object]] = {}
        self._factories = (JSONDecoderFactory(), Ros2DecoderFactory())
        self._event_index = 0
        self._active_key: int | None = None

    @override
    def initialize(self, context: PipelineContext) -> None:
        for input_context in context.inputs:
            summary = input_context.summary
            if summary is None:
                continue
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id) if channel.schema_id else None
                self._register(channel, schema)

    @override
    def on_channel(self, context: ChannelContext, channel: Channel, schema: Schema | None) -> None:
        self._register(channel, schema)

    def _register(self, channel: Channel, schema: Schema | None) -> None:
        self.channels[channel.id] = channel
        if channel.topic not in self.matcher.topics or channel.id in self._decoders:
            return
        for factory in self._factories:
            decoder = factory.decoder_for(channel.message_encoding, schema)
            if decoder is not None:
                self._decoders[channel.id] = decoder
                return

    def _chunk_has_event_channel(self, indexes: tuple[MessageIndex, ...]) -> bool:
        return any(
            (channel := self.channels.get(index.channel_id)) is None
            or channel.topic in self.matcher.topics
            for index in indexes
        )

    @override
    def on_chunk(self, context: ChunkContext, chunk: Chunk | LazyChunk) -> ChunkDecision:
        if context.message_indexes is None or self._chunk_has_event_channel(
            context.message_indexes
        ):
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    @override
    def route_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> tuple[int, ...] | _SplitRequiredSentinel:
        return () if self._active_key is None else (self._active_key,)

    @override
    def route_message(self, context: MessageContext, message: Message) -> tuple[int, ...]:
        channel = self.channels.get(message.channel_id)
        if channel is None or channel.topic not in self.matcher.topics:
            return () if self._active_key is None else (self._active_key,)
        decoder = self._decoders.get(channel.id)
        if decoder is None:
            raise ValueError(
                f"no decoder for paired event topic {channel.topic!r} "
                f"with encoding {channel.message_encoding!r}"
            )
        kinds = self.matcher.match(
            channel.topic,
            decoder(message.data),
            message.log_time,
            message.publish_time,
        )
        if not kinds:
            return () if self._active_key is None else (self._active_key,)
        routes: list[int] = []
        for kind in kinds:
            event = BoundaryEvent(kind, message.log_time, channel.topic)
            if self._event_index >= len(self.plan.steps):
                raise RuntimeError(f"unexpected paired boundary during output pass: {event}")
            step = self.plan.steps[self._event_index]
            if step.event != event:
                raise RuntimeError(
                    f"paired boundary mismatch during output pass: "
                    f"expected {step.event}, got {event}"
                )
            self._event_index += 1
            if step.route_key is not None and step.route_key not in routes:
                routes.append(step.route_key)
            self._active_key = step.active_after
        return tuple(routes)

    def validate_complete(self) -> None:
        if self._event_index != len(self.plan.steps):
            missing = self.plan.steps[self._event_index].event
            raise RuntimeError(f"paired boundary missing during output pass: {missing}")

    @override
    def output_segments(self) -> tuple[OutputSegmentInfo, ...] | None:
        return tuple(
            OutputSegmentInfo(window.key, window.start_time, window.end_time)
            for window in self.plan.windows
        )

    @override
    def template_fields(self, key: RouteKey) -> dict[str, TemplateValue]:
        if type(key) is not int or key < 0 or key >= len(self.plan.windows):
            return {}
        window = self.plan.windows[key]
        return {"window_start": window.start_time, "window_end": window.end_time}
