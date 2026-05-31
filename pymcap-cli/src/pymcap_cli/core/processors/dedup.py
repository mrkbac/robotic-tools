"""Drop messages whose ``(channel_id, log_time, payload)`` has already been written.

**Scope:** dedup is for *cross-input* duplicate elimination during merges.
The dedup key is built on the post-remap message so two inputs that publish
the same logical topic share a channel id and dedup correctly.

This processor optimises for that contract: when summary chunk indexes are
available for every input, we precompute per-stream intervals from *other*
inputs and let any chunk whose time range doesn't overlap any other input
fast-copy. *Intra-input* duplicates within those fast-copied chunks are
**not** caught — by the time-range argument, an identical message can only
land on chunks whose ranges overlap another input's. If your contract
requires removing intra-input duplicates, build the processor in a chain
behind ``AlwaysDecodeProcessor`` to force per-message inspection on every
chunk.

When chunk indexes are missing for any input, the processor falls back to
forcing ``ChunkDecision.DECODE`` on every chunk — both cross-input and
intra-input duplicates are then caught at the cost of full decompression.

Hashing prefers ``xxhash.xxh3_64_intdigest`` when the optional ``xxhash``
package is installed (~2x faster wall time on multi-MB payloads). Falls back
to ``zlib.crc32`` when it isn't.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    ChunkContext,
    ChunkDecision,
    InputProcessor,
    MessageContext,
    PipelineContext,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from small_mcap import Chunk, LazyChunk, Message

try:
    from xxhash import xxh3_64_intdigest as _hash_bytes
except ImportError:
    from zlib import crc32 as _hash_bytes_fallback

    _hash_bytes: Callable[[bytes | memoryview], int] = _hash_bytes_fallback


class DedupIdenticalProcessor(InputProcessor):
    """Skip messages with a duplicate ``(channel_id, log_time, payload-hash)``.

    Per-key state is a ``set[int]`` of payload hashes — one digest per
    distinct payload seen at that ``(channel_id, log_time)``. 64-bit hash
    collisions are not a practical concern for typical robot logs.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[int, int], set[int]] = {}
        self.dropped_count = 0
        # Per-stream "other-input" chunk intervals (sorted, disjoint, merged).
        # Populated from the full summary set during ``initialize``; an empty
        # list for a stream means "no other input has overlapping chunks."
        self._other_starts: list[list[int]] = []
        self._other_ends: list[list[int]] = []
        self._has_chunk_indexes = False

    @override
    def initialize(self, context: PipelineContext) -> None:
        self._other_starts.clear()
        self._other_ends.clear()
        # Cross-stream overlap checks need at least two streams, all with usable
        # chunk indexes; any missing summary forces the safe DECODE fallback.
        per_stream: list[list[tuple[int, int]]] = []
        for input_context in context.inputs:
            if not input_context.chunk_indexes:
                self._has_chunk_indexes = False
                return
            per_stream.append(
                [
                    (chunk.message_start_time, chunk.message_end_time)
                    for chunk in input_context.chunk_indexes
                ]
            )
        if len(per_stream) < 2:
            self._has_chunk_indexes = False
            return
        self._has_chunk_indexes = True

        for i, _ranges in enumerate(per_stream):
            others: list[tuple[int, int]] = []
            for j, ranges in enumerate(per_stream):
                if j != i:
                    others.extend(ranges)
            others.sort()
            merged: list[tuple[int, int]] = []
            for s, e in others:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            self._other_starts.append([s for s, _ in merged])
            self._other_ends.append([e for _, e in merged])

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        if not self._has_chunk_indexes:
            return ChunkDecision.DECODE
        stream_id = context.input.stream_id
        if stream_id >= len(self._other_starts):
            return ChunkDecision.DECODE

        starts = self._other_starts[stream_id]
        if not starts:
            return ChunkDecision.CONTINUE

        ends = self._other_ends[stream_id]
        cs = chunk.message_start_time
        ce = chunk.message_end_time
        # First interval with start > ce. Everything before could overlap; among
        # those, the immediately preceding interval has the largest end (since
        # intervals are sorted, disjoint, and merged).
        idx = bisect.bisect_right(starts, ce)
        if idx > 0 and ends[idx - 1] >= cs:
            return ChunkDecision.DECODE
        return ChunkDecision.CONTINUE

    @override
    def on_message(self, context: MessageContext, message: Message) -> Iterable[Message]:
        time_key = (message.channel_id, message.log_time)
        digest = _hash_bytes(message.data)
        existing = self._seen.get(time_key)
        if existing is None:
            self._seen[time_key] = {digest}
            yield message
            return
        if digest in existing:
            self.dropped_count += 1
            return
        existing.add(digest)
        yield message
