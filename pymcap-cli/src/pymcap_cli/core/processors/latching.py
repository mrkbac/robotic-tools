# ruff: noqa: ARG002
"""Latching processor — keep messages on transient-local topics across cuts.

When ``filter`` or ``split`` would drop messages on topics like ``/tf_static``,
the resulting MCAPs become unusable for any consumer that depends on the
last-published value (TF tree, robot description, etc.). This processor:

1. Identifies "latched" channels (explicit topic patterns and/or, opt-in,
   channels whose MCAP metadata advertises ``durability: transient_local``).
2. Vetoes SKIP decisions from other input processors so latched channels
   always reach the output.
3. Caches the most recent message seen on each latched channel.
4. When a new output segment first opens for writing, replays the cached
   latched messages into it via the ``on_segment_open`` hook. The original
   ``log_time`` / ``publish_time`` are preserved.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import yaml
from small_mcap import McapError, read_message

from pymcap_cli.core.processors.base import (
    Action,
    ChunkDecision,
    OutputSegmentInfo,
    Processor,
    ProcessorInputContext,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from re import Pattern

    from small_mcap import Channel, Chunk, LazyChunk, Message, MessageIndex, Schema, Summary

    from pymcap_cli.core.processors.base import OutputKey

_QOS_METADATA_KEY = "offered_qos_profiles"
_TRANSIENT_LOCAL = "transient_local"


def _channel_is_transient_local(channel: Channel) -> bool:
    """Return True if the channel metadata advertises ``durability: transient_local``.

    ROS 2 MCAPs encode QoS as a YAML list of per-publisher dicts under the
    ``offered_qos_profiles`` metadata key. Any malformed/unknown shape returns
    False — the explicit ``--latch`` flag is the user's escape hatch.
    """
    blob = channel.metadata.get(_QOS_METADATA_KEY)
    if not blob:
        return False
    try:
        parsed = yaml.safe_load(blob)
    except yaml.YAMLError:
        return False
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return False
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        durability = entry.get("durability")
        if isinstance(durability, str) and durability.lower() == _TRANSIENT_LOCAL:
            return True
    return False


class LatchingProcessor(Processor):
    """Keep latched topics across filter/split cuts and replay into new segments.

    Args:
        patterns: Compiled regex patterns. Any channel whose topic matches
            (``re.search`` semantics) is considered latched.
        from_metadata: When True, also flag channels whose MCAP metadata
            advertises ``durability: transient_local`` (default False —
            opt-in).
    """

    def __init__(
        self,
        patterns: list[Pattern[str]] | None = None,
        *,
        from_metadata: bool = False,
    ) -> None:
        self._patterns = patterns or []
        self._from_metadata = from_metadata
        self._latched_channel_ids: set[int] = set()
        self._latched_topics: set[str] = set()
        self._last_message: dict[int, Message] = {}
        self._replayed_segments: set[OutputKey] = set()
        self._context: ProcessorInputContext | None = None
        self._segments_by_key: dict[OutputKey, OutputSegmentInfo] = {}
        self._scan_channel_ids: set[int] = set()
        self._segment_replays: dict[OutputKey, tuple[tuple[int, Message], ...]] = {}

    @property
    def latched_channel_ids(self) -> set[int]:
        return self._latched_channel_ids

    @property
    def latched_topics(self) -> set[str]:
        return self._latched_topics

    def _is_latched(self, channel: Channel) -> bool:
        if any(p.search(channel.topic) for p in self._patterns):
            return True
        return self._from_metadata and _channel_is_transient_local(channel)

    def initialize(self, summaries: list[Summary | None]) -> None:
        for summary in summaries:
            if summary is None:
                continue
            for channel in summary.channels.values():
                if self._is_latched(channel):
                    self._latched_channel_ids.add(channel.id)
                    self._latched_topics.add(channel.topic)

    def prepare_input(self, context: ProcessorInputContext) -> None:
        self._context = context
        self._segments_by_key = {segment.key: segment for segment in context.output_segments}
        self._scan_channel_ids.clear()
        summary = context.summary
        if summary is None:
            return
        for channel in summary.channels.values():
            if not self._is_latched(channel):
                continue
            remapped_channel = context.remap_channel(channel)
            self._scan_channel_ids.add(remapped_channel.id)

    def on_channel(self, channel: Channel, schema: Schema | None) -> Action:
        # Discover channels that arrive without a summary (recovery / streaming).
        if channel.id not in self._latched_channel_ids and self._is_latched(channel):
            self._latched_channel_ids.add(channel.id)
            self._latched_topics.add(channel.topic)
            self._scan_channel_ids.add(channel.id)
        if channel.id in self._latched_channel_ids:
            return Action.KEEP
        return Action.CONTINUE

    def on_chunk(self, chunk: Chunk | LazyChunk, indexes: list[MessageIndex]) -> ChunkDecision:
        # Fast-copied chunks bypass on_message — if the chunk references a
        # latched channel we must DECODE so the latch cache stays current.
        if any(idx.channel_id in self._latched_channel_ids for idx in indexes):
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    def on_message(self, message: Message) -> Action:
        if message.channel_id in self._latched_channel_ids:
            self._last_message[message.channel_id] = message
            return Action.KEEP
        return Action.CONTINUE

    def _find_segment_replay(self, key: OutputKey) -> tuple[tuple[int, Message], ...] | None:
        segment = self._segments_by_key.get(key)
        if segment is None:
            return None
        if segment.start_time <= 0:
            return ()
        if key in self._segment_replays:
            return self._segment_replays[key]
        if self._context is None or not self._scan_channel_ids:
            return None

        stream = self._context.stream
        try:
            position = stream.tell()
            stream.seek(0)
        except (OSError, ValueError):
            return None

        found: dict[int, Message] = {}
        try:
            try:
                messages = read_message(
                    stream,
                    end_time_ns=segment.start_time,
                    reverse=True,
                )
                for _schema, _channel, message in messages:
                    remapped = self._context.remap_message(message)
                    if remapped.channel_id not in self._scan_channel_ids:
                        continue
                    if remapped.channel_id in found:
                        continue
                    found[remapped.channel_id] = remapped
                    if self._scan_channel_ids <= found.keys():
                        break
            except (McapError, OSError, ValueError):
                return None
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.seek(position)

        replay = tuple(found.items())
        self._segment_replays[key] = replay
        return replay

    def on_segment_open(self, key: OutputKey) -> Iterable[tuple[int, Message]]:
        if key in self._replayed_segments:
            return ()
        self._replayed_segments.add(key)
        segment_replay = self._find_segment_replay(key)
        if segment_replay is not None:
            return segment_replay
        return tuple(self._last_message.items())
