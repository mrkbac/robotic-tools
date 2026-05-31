"""Rewrite ``Channel.topic`` strings via regex substitution.

Rules map a topic-regex to a replacement (``re.sub`` semantics — back-
references like ``\\1`` work). First matching pattern wins. Useful for
consolidating recordings whose channels were named differently across
robot versions.

Pure container-level: only ``Channel.topic`` is rewritten; the payload
bytes are never decoded. Once any channel has been rewritten, every chunk
asks the dispatcher for ``DECODE_VERIFY`` — the chunk is decompressed
once, its embedded Schema/Channel records are compared against the
writer's view, and the chunk fast-copies when clean. Only chunks that
actually carry a stale Channel record fall through to a full re-emit.
Some MCAP writers eagerly embed the full channel list in every chunk
regardless of which channels have messages there, so the chunk's
MessageIndex doesn't reveal whether an inline copy is hiding inside.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    ChunkDecision,
    InputProcessor,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from re import Pattern

    from small_mcap import Channel, Chunk, LazyChunk, Schema


class TopicRewriteProcessor(InputProcessor):
    """Rewrite ``Channel.topic`` strings via regex substitution."""

    def __init__(self, rules: Mapping[str, str]) -> None:
        self._rules: list[tuple[Pattern[str], str]] = [
            (re.compile(p), repl) for p, repl in rules.items()
        ]
        self._rewritten_channel_ids: set[int] = set()

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        for pat, repl in self._rules:
            new_topic, n = pat.subn(repl, channel.topic)
            if n > 0:
                channel.topic = new_topic
                self._rewritten_channel_ids.add(channel.id)
                break
        return Action.CONTINUE

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        if self._rewritten_channel_ids:
            return ChunkDecision.DECODE_VERIFY
        return ChunkDecision.CONTINUE
