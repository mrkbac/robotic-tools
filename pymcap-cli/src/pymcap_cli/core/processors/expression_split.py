# ruff: noqa: ARG002
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory

from pymcap_cli.core.processors.base import Action, ChunkDecision, Processor

if TYPE_CHECKING:
    from collections.abc import Callable

    from small_mcap import (
        Channel,
        Chunk,
        LazyChunk,
        Message,
        MessageIndex,
        Schema,
        Summary,
    )

_UNSET: Any = object()  # sentinel — no previous value seen yet


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

    def __init__(self, path: str) -> None:
        self.path_str = path
        self.parsed = parse_message_path(path)
        self._factories = (JSONDecoderFactory(), Ros2DecoderFactory())
        self.channels: dict[int, Channel] = {}
        self._decoders: dict[int, Callable[[bytes | memoryview], Any]] = {}
        self._validated_schema_ids: set[int] = set()
        self._segment_index: int = 0
        self._prev_value: Any = _UNSET

    def initialize(self, summaries: list[Summary | None]) -> None:
        for summary in summaries:
            if summary is None:
                continue
            for channel in summary.channels.values():
                schema = (
                    summary.schemas.get(channel.schema_id) if channel.schema_id else None
                )
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
            print(
                f"Warning: path '{self.path_str}' invalid for {schema.name}: {e}",
                file=sys.stderr,
            )

    def _chunk_has_target(self, indexes: list[MessageIndex]) -> bool:
        if not self.channels:
            return True  # channels not yet known → decode to stay safe
        target = self.parsed.topic
        return any(
            (ch := self.channels.get(idx.channel_id)) is not None and ch.topic == target
            for idx in indexes
        )

    def on_chunk(
        self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]
    ) -> ChunkDecision:
        if self._chunk_has_target(indexes):
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def route_chunk(self, chunk: Chunk | LazyChunk) -> int | str | object | None:
        # Reached only for fast-copy chunks (on_chunk returned CONTINUE). Those
        # contain no target-topic messages, so route the whole chunk to the
        # current sticky segment.
        return self._segment_index

    def route_message(self, message: Message) -> int:
        dec = self._decoders.get(message.channel_id)
        ch = self.channels.get(message.channel_id)
        if dec is None or ch is None or ch.topic != self.parsed.topic:
            return self._segment_index
        try:
            value = self.parsed.apply(dec(message.data))
        except MessagePathError:
            return self._segment_index
        if value != self._prev_value:
            # Skip the very first transition (from _UNSET → first value) so the
            # initial run stays in segment 0 instead of jumping to 1.
            if self._prev_value is not _UNSET:
                self._segment_index += 1
            self._prev_value = value
        return self._segment_index

    def output_keys(self) -> list[int | str] | None:
        return None
