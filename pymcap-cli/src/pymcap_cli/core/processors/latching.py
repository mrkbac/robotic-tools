"""Latching processor — keep messages on transient-local topics across cuts.

When ``filter`` or ``split`` would drop messages on topics like ``/tf_static``,
the resulting MCAPs become unusable for any consumer that depends on the
last-published value (TF tree, robot description, etc.). This processor:

1. Identifies "latched" channels (explicit topic patterns and/or, opt-in,
   channels whose MCAP metadata advertises ``durability: transient_local``).
2. Caches the most recent message seen on each latched channel.
3. When a new output segment first opens for writing, replays the cached
   latched messages into it via the ``on_segment_open`` hook. The original
   ``log_time`` / ``publish_time`` are preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    InputContext,
    InputProcessor,
    MessageContext,
    MessageScope,
    PipelineContext,
    SegmentContext,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from re import Pattern

    from small_mcap import Channel, Message, Schema

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


class LatchingProcessor(InputProcessor):
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
        self._previous_message: dict[int, Message] = {}
        self._replayed_segments: set[OutputKey] = set()

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

    @override
    def initialize(self, context: PipelineContext) -> None:
        for input_context in context.inputs:
            if input_context.summary is None:
                continue
            for channel in input_context.summary.channels.values():
                if self._is_latched(channel):
                    self._latched_topics.add(channel.topic)

    @override
    def prepare_input(self, context: InputContext) -> None:
        _ = context

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        # Discover channels that arrive without a summary (recovery / streaming).
        if channel.id not in self._latched_channel_ids and self._is_latched(channel):
            self._latched_channel_ids.add(channel.id)
            self._latched_topics.add(channel.topic)
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        # Fast-copied chunks bypass on_message — if the chunk references a
        # latched channel we must DECODE so the latch cache stays current.
        return MessageScope.channels(set(self._latched_channel_ids))

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        if message.channel_id in self._latched_channel_ids:
            previous = self._last_message.get(message.channel_id)
            if previous is not None:
                self._previous_message[message.channel_id] = previous
            self._last_message[message.channel_id] = message
        yield message

    def _replay_messages_before(
        self,
        start_time: int,
        observed_message: Message | None,
    ) -> tuple[tuple[int, Message], ...]:
        replay: list[tuple[int, Message]] = []
        for channel_id, message in self._last_message.items():
            if message is observed_message:
                previous = self._previous_message.get(channel_id)
                if previous is not None and previous.log_time < start_time:
                    replay.append((channel_id, previous))
                continue
            if message.log_time < start_time:
                replay.append((channel_id, message))
                continue
            previous = self._previous_message.get(channel_id)
            if previous is not None and previous.log_time < start_time:
                replay.append((channel_id, previous))
        return tuple(replay)

    @override
    def on_segment_open(self, context: SegmentContext) -> Iterable[tuple[int, Message]]:
        if context.key in self._replayed_segments:
            return ()
        self._replayed_segments.add(context.key)
        return self._replay_messages_before(context.start_time, context.observed_message)
