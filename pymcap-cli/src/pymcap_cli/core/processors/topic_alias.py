# ruff: noqa: ARG002
"""Emit every matched message under both its original topic and one or more alias topics.

Used for gradual consumer migration: while subscribers are being switched
from ``/old/foo`` to ``/new/foo``, write each message on both topics so
either set of consumers keeps working. Once migration is complete, drop
the rule.

Pure container-level: the payload bytes are reused verbatim — only the
``Channel.id`` of each emitted copy differs. Chunks referencing aliased
channels are forced to DECODE so ``on_message`` runs and fans out.

Alias Channel records are registered through
``InputContext.register_channel`` — eagerly during ``prepare_input``
when the input has a summary, lazily from ``on_channel`` when it doesn't
(streamed / unindexed / broken files). When no summary is available the
processor pessimistically reports every chunk as touched so inline Channel
records inside chunks get a chance to surface — the summary is an
optimization, not a requirement.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    ChunkDecision,
    InputContext,
    InputProcessor,
    MessageContext,
    MessageScope,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from re import Pattern

    from small_mcap import Channel, Chunk, LazyChunk, Message, Schema


class TopicAliasProcessor(InputProcessor):
    """Duplicate matched messages onto one or more alias channels.

    ``rules`` maps a topic regex to either a single replacement topic or a
    list of replacements (``re.sub`` semantics — backreferences like
    ``\\1`` work in the replacement). For every channel whose topic matches
    a rule, the processor registers one alias channel per replacement and
    fans each incoming message out to all aliases in addition to the
    original.
    """

    def __init__(self, rules: Mapping[str, str | list[str]]) -> None:
        self._rules: list[tuple[Pattern[str], list[str]]] = []
        for pattern, replacements in rules.items():
            replacement_list = (
                [replacements] if isinstance(replacements, str) else list(replacements)
            )
            self._rules.append((re.compile(pattern), replacement_list))
        # original_channel_id -> list of alias channel ids
        self._aliases: dict[int, list[int]] = {}
        self._register_channel: Callable[[Channel], Channel] | None = None
        # Streams with usable summaries can trust ``self._aliases`` to be
        # complete for their known channels. Streams without summaries stay
        # pessimistic at chunk classification time so inline Channel records get
        # decoded and surfaced to on_channel().
        self._streams_with_summary: set[int] = set()

    def _resolve_aliases(self, topic: str) -> list[str]:
        for pattern, replacements in self._rules:
            if pattern.search(topic):
                return [pattern.sub(repl, topic) for repl in replacements]
        return []

    def _register_aliases_for(self, channel: Channel) -> None:
        # Pre-load fires ``on_channel`` *before* ``prepare_input`` stores
        # ``register_channel``. Skip silently until the context is wired up;
        # the same channel re-enters via ``prepare_input`` -> remap path.
        if channel.id in self._aliases or self._register_channel is None:
            return
        alias_topics = self._resolve_aliases(channel.topic)
        if not alias_topics:
            return
        alias_ids: list[int] = []
        for alias_topic in alias_topics:
            if alias_topic == channel.topic:
                continue
            alias_channel = self._register_channel(replace(channel, topic=alias_topic))
            alias_ids.append(alias_channel.id)
        if alias_ids:
            self._aliases[channel.id] = alias_ids

    def prepare_input(self, context: InputContext) -> None:
        # Stash register_channel so on_channel can also register aliases
        # for channels that arrive later (streamed / unindexed files).
        self._register_channel = context.register_channel
        if context.summary is None:
            return
        self._streams_with_summary.add(context.stream_id)
        # The summary's channels carry pre-remap ids; remap each before
        # keying the alias map so on_message lookups by remapped channel_id
        # succeed (cross-input id collisions are otherwise silently broken).
        for channel in context.summary.channels.values():
            remapped = context.remap_channel(channel)
            self._register_aliases_for(remapped)

    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        self._register_aliases_for(channel)
        return Action.CONTINUE

    def message_scope(self, context: ChunkContext) -> MessageScope:
        if context.input.stream_id not in self._streams_with_summary:
            return MessageScope.all()
        return MessageScope.channels(set(self._aliases))

    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        return super().on_chunk(context, chunk)

    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        yield message
        for alias_id in self._aliases.get(message.channel_id, ()):
            yield replace(message, channel_id=alias_id)
