"""Collapse duplicate channels with identical content onto a single channel id.

Two channels are "duplicates" when they share ``(topic, schema_id,
message_encoding, metadata)``. Cross-input duplicates are already merged by
the framework's Remapper; this processor catches the within-input case —
typically caused by a recorder restart that re-advertised a channel under a
fresh id.

Pure container-level: only ``Message.channel_id`` is rewritten; the payload
bytes are never decoded. Chunks referencing redirected channel ids are
forced to DECODE so ``on_message`` can rewrite the channel_id. The
absorbed Channel record itself is never written to the output because no
message ever requests it after redirect.
"""

from __future__ import annotations

from dataclasses import replace
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
    from collections.abc import Iterable

    from small_mcap import Channel, Message, Schema

_ContentKey = tuple[str, int, str, frozenset[tuple[str, str]]]


class ChannelMergeProcessor(InputProcessor):
    """Redirect duplicate-content channels onto the first-seen id."""

    def __init__(self) -> None:
        self._first_seen: dict[_ContentKey, int] = {}
        self._redirect: dict[int, int] = {}

    @staticmethod
    def _content_key(channel: Channel) -> _ContentKey:
        return (
            channel.topic,
            channel.schema_id,
            channel.message_encoding,
            frozenset(channel.metadata.items()),
        )

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        key = self._content_key(channel)
        existing_id = self._first_seen.get(key)
        if existing_id is None or existing_id == channel.id:
            self._first_seen[key] = channel.id
            return Action.CONTINUE
        # Duplicate content under a different id — redirect future messages
        # to the first-seen id. Keep the channel marked "included" so the
        # framework lets its messages reach on_message; the absorbed record
        # itself never gets written because ensure_channel_written is only
        # called for the target id.
        self._redirect[channel.id] = existing_id
        return Action.CONTINUE

    @override
    def message_scope(self, context: ChunkContext) -> MessageScope:
        return MessageScope.channels(set(self._redirect))

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        target = self._redirect.get(message.channel_id)
        if target is not None:
            yield replace(message, channel_id=target)
            return
        yield message
