from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    Filter,
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory
from typing_extensions import override

from pymcap_cli.core.processors.base import (
    ChannelContext,
    ChunkContext,
    ChunkDecision,
    MessageContext,
    OutputRouter,
    PipelineContext,
    _SplitRequiredSentinel,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ros_parser.message_path import MessagePathVariable, MessagePathVariables
    from small_mcap import Channel, Chunk, LazyChunk, Message, MessageIndex, Schema

logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel singleton for ``ExpressionSplitProcessor._prev_value``."""


_UNSET = _Unset()


@dataclass(slots=True)
class _Candidate:
    """A pending value waiting for hysteresis thresholds before it commits."""

    value: MessagePathVariable
    first_seen_ns: int
    count: int


@dataclass(slots=True)
class _TailWindow:
    target: int
    until_ns: int | None
    count_left: int


class ExpressionSplitProcessor(OutputRouter):
    """Split output each time a ros-parser message path changes its value.

    The path (e.g. ``/gps/fix.status.status`` or ``/detections{confidence>0.8}``)
    is evaluated on every message whose topic matches ``path.topic``. Whenever
    the result differs from the previous result the segment index increments —
    so predicate filters partition the stream into alternating match / no-match
    runs, and value extractors partition it into runs of equal value.

    Messages on other topics follow the most recent segment (sticky routing),
    and chunks that contain no target-topic messages fast-copy without decoding.

    Output keys are integers (0-based), suitable for ``output_{index:03d}`` or
    ``output_{key}`` templates.
    """

    def __init__(
        self,
        path: str,
        *,
        hysteresis_ns: int | None = None,
        hysteresis_count: int | None = None,
        trailing_context_ns: int | None = None,
        trailing_context_count: int | None = None,
        variables: MessagePathVariables | None = None,
        skip_values: tuple[MessagePathVariable, ...] = (),
        require_value: bool = False,
    ) -> None:
        if hysteresis_ns is not None and hysteresis_ns <= 0:
            raise ValueError("hysteresis_ns must be positive")
        if hysteresis_count is not None and hysteresis_count <= 0:
            raise ValueError("hysteresis_count must be positive")
        if trailing_context_ns is not None and trailing_context_ns <= 0:
            raise ValueError("trailing_context_ns must be positive")
        if trailing_context_count is not None and trailing_context_count <= 0:
            raise ValueError("trailing_context_count must be positive")

        self.path_str = path
        self.parsed = parse_message_path(path)
        self._is_predicate = any(isinstance(segment, Filter) for segment in self.parsed.segments)
        self._factories = (JSONDecoderFactory(), Ros2DecoderFactory())
        self.channels: dict[int, Channel] = {}
        self._decoders: dict[int, Callable[[bytes | memoryview], Any]] = {}
        self._validated_schema_ids: set[int] = set()
        self._segment_index: int = 0
        self._prev_value: MessagePathVariable | _Unset = _UNSET
        self._segment_values: dict[int, MessagePathVariable] = {}
        self._hysteresis_ns = hysteresis_ns
        self._hysteresis_count = hysteresis_count
        self._candidate: _Candidate | None = None
        self._trailing_ns = trailing_context_ns
        self._trailing_count = trailing_context_count
        self._tail_windows: list[_TailWindow] = []
        self._variables = dict(variables or {})
        self.skip_values = skip_values
        self.require_value = require_value

    @override
    def initialize(self, context: PipelineContext) -> None:
        for input_context in context.inputs:
            if input_context.summary is None:
                continue
            for channel in input_context.summary.channels.values():
                schema = (
                    input_context.summary.schemas.get(channel.schema_id)
                    if channel.schema_id
                    else None
                )
                self._register(channel, schema)

    @override
    def on_channel(self, context: ChannelContext, channel: Channel, schema: Schema | None) -> None:
        self._register(channel, schema)

    def _register(self, channel: Channel, schema: Schema | None) -> None:
        self.channels[channel.id] = channel
        if channel.topic != self.parsed.topic or schema is None:
            return
        if channel.id not in self._decoders:
            for factory in self._factories:
                dec = factory.decoder_for(channel.message_encoding, schema)
                if dec is not None:
                    self._decoders[channel.id] = dec
                    break
        if schema.id not in self._validated_schema_ids:
            self._validated_schema_ids.add(schema.id)
            self._validate_path(schema)

    def _validate_path(self, schema: Schema) -> None:
        try:
            all_defs = parse_schema_to_definitions(schema.name, schema.data)
        except Exception:  # noqa: BLE001
            # Non-ROS schema (e.g. empty JSON schema) — skip validation silently.
            return
        root = all_defs.get(schema.name)
        if root is None:
            parts = schema.name.split("/")
            root = all_defs.get(f"{parts[0]}/{parts[-1]}")
        if root is None:
            return
        try:
            result_type, _ = self.parsed.resolve_type(root, all_defs)
        except ValidationError as e:
            logger.warning("path %r invalid for %s: %s", self.path_str, schema.name, e)
            return
        if not self._is_predicate and (not result_type.is_primitive or result_type.is_array):
            raise ValueError(
                f"Expression {self.path_str!r} must resolve to a primitive; "
                f"schema {schema.name!r} resolves it as {result_type}"
            )

    def _chunk_has_target(self, indexes: Iterable[MessageIndex]) -> bool:
        if not self.channels:
            return True  # channels not yet known → decode to stay safe
        target = self.parsed.topic
        return any(
            (ch := self.channels.get(idx.channel_id)) is not None and ch.topic == target
            for idx in indexes
        )

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        if context.message_indexes is None or self._chunk_has_target(context.message_indexes):
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    @override
    def route_chunk(
        self, context: ChunkContext, chunk: Chunk | LazyChunk
    ) -> tuple[int, ...] | _SplitRequiredSentinel:
        # Reached only for fast-copy chunks (on_chunk returned CONTINUE). Those
        # contain no target-topic messages, so route the whole chunk to the
        # current sticky segment.
        return self._current_routes()

    def _commit_transition(self, new_value: MessagePathVariable, log_time_ns: int) -> None:
        """Commit a value transition: bump the segment index and arm the
        trailing-context window if configured."""
        # Skip the very first transition (from _UNSET → first value) so the
        # initial run stays in segment 0 instead of jumping to 1.
        prev_was_unset = self._prev_value is _UNSET
        if not prev_was_unset:
            self._segment_index += 1
        self._prev_value = new_value
        self._segment_values[self._segment_index] = new_value
        self._candidate = None

        # Arm trailing context for the segment we just left (the previous
        # one). No tail for the very first commit since there is no previous
        # segment to back-fill.
        if prev_was_unset or (self._trailing_ns is None and self._trailing_count is None):
            return
        self._tail_windows.append(
            _TailWindow(
                target=self._segment_index - 1,
                until_ns=log_time_ns + self._trailing_ns if self._trailing_ns is not None else None,
                count_left=self._trailing_count if self._trailing_count is not None else -1,
            )
        )

    @override
    def route_message(self, context: MessageContext, message: Message) -> tuple[int, ...]:
        dec = self._decoders.get(message.channel_id)
        ch = self.channels.get(message.channel_id)
        if dec is None or ch is None or ch.topic != self.parsed.topic:
            return self._current_routes()
        try:
            raw_value: object = self.parsed.apply(dec(message.data), self._variables)
        except MessagePathError:
            return self._current_routes()
        value = self._normalize_value(raw_value)

        prev_value = self._prev_value
        if not isinstance(prev_value, _Unset) and self._values_equal(value, prev_value):
            # Same value as current segment — clear any pending candidate
            # (the transition flapped back).
            self._candidate = None
            return self._routes_for_target_message(message)

        # The very first value seen is the initial state, not a transition —
        # commit it without consulting hysteresis so segment 0 holds the
        # right ``_prev_value`` for subsequent diffs.
        if self._prev_value is _UNSET:
            self._commit_transition(value, message.log_time)
            return self._routes_for_target_message(message)

        if self._hysteresis_ns is None and self._hysteresis_count is None:
            self._commit_transition(value, message.log_time)
            return self._routes_for_target_message(message)

        if self._candidate is None or not self._values_equal(self._candidate.value, value):
            self._candidate = _Candidate(value=value, first_seen_ns=message.log_time, count=1)
        else:
            self._candidate.count += 1

        time_ok = self._hysteresis_ns is None or (
            message.log_time - self._candidate.first_seen_ns >= self._hysteresis_ns
        )
        count_ok = self._hysteresis_count is None or self._candidate.count >= self._hysteresis_count
        if time_ok and count_ok:
            self._commit_transition(value, message.log_time)

        return self._routes_for_target_message(message)

    def _routes_for_target_message(self, message: Message) -> tuple[int, ...]:
        routes = [*self._current_routes(), *self._extra_routes_for_target_message(message)]
        return tuple(dict.fromkeys(routes))

    def _normalize_value(self, value: object) -> MessagePathVariable:
        if self._is_predicate:
            return not (value is None or (isinstance(value, (list, tuple)) and not value))
        if type(value) not in (bool, int, float, str):
            raise ValueError(
                f"Expression {self.path_str!r} must resolve to a primitive "
                f"bool, int, float, or str; got {type(value).__name__}"
            )
        return cast("MessagePathVariable", value)

    @staticmethod
    def _values_equal(left: MessagePathVariable, right: MessagePathVariable) -> bool:
        if type(left) is bool or type(right) is bool:
            return type(left) is type(right) and left == right
        return left == right

    def _is_skipped(self, value: MessagePathVariable) -> bool:
        return any(self._values_equal(value, skipped) for skipped in self.skip_values)

    def _current_routes(self) -> tuple[int, ...]:
        prev_value = self._prev_value
        if isinstance(prev_value, _Unset):
            return () if self.require_value else (self._segment_index,)
        if self._is_skipped(prev_value):
            return ()
        return (self._segment_index,)

    @override
    def template_fields(self, key: int | str) -> dict[str, MessagePathVariable]:
        if type(key) is not int:
            return {}
        value = self._segment_values.get(key)
        return {} if value is None else {"value": value}

    def _extra_routes_for_target_message(self, message: Message) -> Iterable[int]:
        """Duplicate target-topic messages into the previous segment during
        the trailing-context window. Other-topic messages stay sticky."""
        if not self._tail_windows:
            return ()
        ch = self.channels.get(message.channel_id)
        if ch is None or ch.topic != self.parsed.topic:
            return ()

        targets: list[int] = []
        active_windows: list[_TailWindow] = []
        for window in self._tail_windows:
            if window.until_ns is not None and message.log_time > window.until_ns:
                continue
            if window.count_left == 0:
                continue

            value = self._segment_values.get(window.target)
            if value is not None and not self._is_skipped(value) and window.target not in targets:
                targets.append(window.target)

            if window.count_left > 0:
                window.count_left -= 1
            if window.count_left != 0:
                active_windows.append(window)

        self._tail_windows = active_windows
        return tuple(targets)
