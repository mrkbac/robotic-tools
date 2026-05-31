"""Keep every Nth message per channel.

Rules map a topic-regex to an integer N. Channels whose topic matches a
pattern get downsampled to 1-of-N (the first message on each channel is
always kept). Channels not matching any pattern pass through unchanged.

Pure container-level: only the Message header (``channel_id``) is consulted;
the payload bytes are never decoded. Chunks containing covered channels are
forced to DECODE so ``on_message`` can run per-message; everything else
fast-copies.
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


class NthMessageProcessor(InputProcessor):
    """Keep every Nth message on channels whose topic matches a pattern."""

    def __init__(self, rules: Mapping[str, int]) -> None:
        for pattern, n in rules.items():
            if n < 1:
                msg = f"N for pattern {pattern!r} must be >= 1, got {n}"
                raise ValueError(msg)
        self._patterns: list[tuple[Pattern[str], int]] = [
            (re.compile(p), n) for p, n in rules.items()
        ]
        self._covered_n: dict[int, int] = {}
        self._counters: dict[int, int] = {}

    def _resolve_n(self, topic: str) -> int | None:
        for pat, n in self._patterns:
            if pat.search(topic):
                return n
        return None

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        n = self._resolve_n(channel.topic)
        if n is not None and n > 1:
            self._covered_n[channel.id] = n
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        return MessageScope.channels(set(self._covered_n))

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        n = self._covered_n.get(message.channel_id)
        if n is None:
            yield message
            return
        counter = self._counters.get(message.channel_id, 0)
        self._counters[message.channel_id] = counter + 1
        if counter % n == 0:
            yield message
