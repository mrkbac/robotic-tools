# ruff: noqa: ARG002
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory

from pymcap_cli.core.processors.base import (
    Action,
    ChunkDecision,
    Processor,
    _SplitRequiredSentinel,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from small_mcap import (
        Channel,
        Chunk,
        LazyChunk,
        Message,
        MessageIndex,
        Schema,
        Summary,
    )

logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel singleton for ``ExpressionSplitProcessor._prev_value``."""


_UNSET = _Unset()


@dataclass(slots=True)
class _Candidate:
    """A pending value waiting for hysteresis thresholds before it commits."""

    value: object
    first_seen_ns: int
    count: int


@dataclass(slots=True)
class _TailWindow:
    target: int
    until_ns: int | None
    count_left: int


class ExpressionSplitProcessor(Processor):
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
        self._factories = (JSONDecoderFactory(), Ros2DecoderFactory())
        self.channels: dict[int, Channel] = {}
        self._decoders: dict[int, Callable[[bytes | memoryview], Any]] = {}
        self._validated_schema_ids: set[int] = set()
        self._segment_index: int = 0
        self._prev_value: object = _UNSET
        # Hysteresis: a new value must persist for ``hysteresis_ns`` AND/OR
        # appear ``hysteresis_count`` times before a segment transition fires.
        # Each None means "not required". Both None means commit immediately
        # (original behaviour).
        self._hysteresis_ns = hysteresis_ns
        self._hysteresis_count = hysteresis_count
        self._candidate: _Candidate | None = None
        # Trailing context: after a transition, target-topic messages are
        # also written into the previous segment for up to
        # ``trailing_context_ns`` time and/or ``trailing_context_count``
        # messages, whichever stops it first.
        self._trailing_ns = trailing_context_ns
        self._trailing_count = trailing_context_count
        self._tail_windows: list[_TailWindow] = []

    def initialize(self, summaries: list[Summary | None]) -> None:
        for summary in summaries:
            if summary is None:
                continue
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id) if channel.schema_id else None
                self._register(channel, schema)

    def on_channel(self, channel: Channel, schema: Schema | None) -> Action:
        self._register(channel, schema)
        return Action.CONTINUE

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
            self.parsed.validate(root, all_defs)
        except ValidationError as e:
            logger.warning("path %r invalid for %s: %s", self.path_str, schema.name, e)

    def _chunk_has_target(self, indexes: list[MessageIndex]) -> bool:
        if not self.channels:
            return True  # channels not yet known → decode to stay safe
        target = self.parsed.topic
        return any(
            (ch := self.channels.get(idx.channel_id)) is not None and ch.topic == target
            for idx in indexes
        )

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        if self._chunk_has_target(indexes):
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | str | _SplitRequiredSentinel | None:
        # Reached only for fast-copy chunks (on_chunk returned CONTINUE). Those
        # contain no target-topic messages, so route the whole chunk to the
        # current sticky segment.
        return self._segment_index

    def _commit_transition(self, new_value: object, log_time_ns: int) -> None:
        """Commit a value transition: bump the segment index and arm the
        trailing-context window if configured."""
        # Skip the very first transition (from _UNSET → first value) so the
        # initial run stays in segment 0 instead of jumping to 1.
        prev_was_unset = self._prev_value is _UNSET
        if not prev_was_unset:
            self._segment_index += 1
        self._prev_value = new_value
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

    def _hysteresis_active(self) -> bool:
        return self._hysteresis_ns is not None or self._hysteresis_count is not None

    def route_message(self, message: Message) -> int:
        dec = self._decoders.get(message.channel_id)
        ch = self.channels.get(message.channel_id)
        if dec is None or ch is None or ch.topic != self.parsed.topic:
            return self._segment_index
        try:
            value = self.parsed.apply(dec(message.data))
        except MessagePathError:
            return self._segment_index

        if value == self._prev_value:
            # Same value as current segment — clear any pending candidate
            # (the transition flapped back).
            self._candidate = None
            return self._segment_index

        # The very first value seen is the initial state, not a transition —
        # commit it without consulting hysteresis so segment 0 holds the
        # right ``_prev_value`` for subsequent diffs.
        if self._prev_value is _UNSET:
            self._commit_transition(value, message.log_time)
            return self._segment_index

        if not self._hysteresis_active():
            self._commit_transition(value, message.log_time)
            return self._segment_index

        # Track candidate: same value as candidate? bump count. Otherwise
        # restart with this new value.
        if self._candidate is None or self._candidate.value != value:
            self._candidate = _Candidate(value=value, first_seen_ns=message.log_time, count=1)
        else:
            self._candidate.count += 1

        # Check both thresholds — each None means "not required".
        time_ok = self._hysteresis_ns is None or (
            message.log_time - self._candidate.first_seen_ns >= self._hysteresis_ns
        )
        count_ok = self._hysteresis_count is None or self._candidate.count >= self._hysteresis_count
        if time_ok and count_ok:
            self._commit_transition(value, message.log_time)

        return self._segment_index

    def also_route_to(self, message: Message) -> Iterable[int]:
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

            if window.target not in targets:
                targets.append(window.target)

            if window.count_left > 0:
                window.count_left -= 1
            if window.count_left != 0:
                active_windows.append(window)

        self._tail_windows = active_windows
        return tuple(targets)

    def output_keys(self) -> list[int | str] | None:
        return None
