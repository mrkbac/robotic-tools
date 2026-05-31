"""Add an ns offset to message log_time and publish_time per matching channel.

Rules map a topic-regex to an ns offset (positive or negative). Channels
whose topic matches a pattern have every covered message's ``log_time`` and
``publish_time`` shifted by the offset. Useful for clock-sync correction
between recorders on different time bases.

Pure container-level: only the Message header fields are rewritten; the
payload bytes are never decoded. Chunks containing covered channels are
forced to DECODE so the framework re-emits messages with the shifted
timestamps (the chunk's own ``message_start_time`` / ``message_end_time``
also get rebuilt by the writer); everything else fast-copies.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    InputProcessor,
    MessageContext,
    MessageScope,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from re import Pattern

    from small_mcap import Channel, Message, Schema


class TimeOffsetProcessor(InputProcessor):
    """Shift ``log_time`` and ``publish_time`` for channels matching a pattern."""

    def __init__(self, rules: Mapping[str, int]) -> None:
        self._patterns: list[tuple[Pattern[str], int]] = [
            (re.compile(p), off) for p, off in rules.items() if off != 0
        ]
        self._channel_offset: dict[int, int] = {}

    def _resolve_offset(self, topic: str) -> int | None:
        for pat, offset in self._patterns:
            if pat.search(topic):
                return offset
        return None

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        offset = self._resolve_offset(channel.topic)
        if offset is not None and offset != 0:
            self._channel_offset[channel.id] = offset
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        return MessageScope.channels(set(self._channel_offset))

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        offset = self._channel_offset.get(message.channel_id)
        if offset is not None:
            message.log_time += offset
            message.publish_time += offset
        yield message
