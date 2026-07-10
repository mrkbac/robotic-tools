"""Concrete OutputProcessor implementations for chunk grouping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from small_mcap import CompressionType
from typing_extensions import override

from pymcap_cli.core.processors.base import OutputProcessor

if TYPE_CHECKING:
    from re import Pattern

    from small_mcap import Channel, Schema

    from pymcap_cli.core.processors.base import OutputKey


class PatternGrouper(OutputProcessor):
    """Group channels into chunks by topic/schema regex.

    Topic patterns are evaluated first (indices ``0..len(topic_patterns)-1``);
    schema patterns are evaluated next (indices offset by ``len(topic_patterns)``).
    First match wins. Channels matching nothing fall into the shared default
    bucket (key ``-1``).
    """

    def __init__(
        self,
        topic_patterns: list[Pattern[str]],
        schema_patterns: list[Pattern[str]] | None = None,
    ) -> None:
        self.topic_patterns = topic_patterns
        self.schema_patterns = schema_patterns or []

    @override
    def chunk_group_key(
        self,
        segment_key: OutputKey,
        channel: Channel,
        schema: Schema | None,
    ) -> int:
        for i, pattern in enumerate(self.topic_patterns):
            if pattern.search(channel.topic):
                return i
        if schema is not None:
            offset = len(self.topic_patterns)
            for i, pattern in enumerate(self.schema_patterns):
                if pattern.search(schema.name):
                    return offset + i
        return -1


class PerChannelGrouper(OutputProcessor):
    """Give every channel its own chunk group, keyed by ``channel.id``."""

    @override
    def chunk_group_key(
        self,
        segment_key: OutputKey,
        channel: Channel,
        schema: Schema | None,
    ) -> int:
        return channel.id


class SchemaCompressionGrouper(OutputProcessor):
    """Route channels whose schema matches a pattern to a chosen compression.

    Defaults to no compression — the motivating case is payloads already
    compressed elsewhere (H.264/H.265 video, Cloudini/Draco point clouds),
    where an extra zstd pass over the chunk gains under 1% (confirmed
    empirically) for real CPU cost on both write and every future read. Pass
    ``compression`` to route matching schemas elsewhere instead.

    By default matching channels join one shared group. With
    ``per_channel=True`` each matching channel gets its own group instead —
    used when the producers (e.g. per-topic video transcode) emit each topic
    monotonically but interleave topics out of log-time order: a shared group
    then buffers wildly out-of-order messages into wide, mutually-overlapping
    chunks, whereas a per-topic group stays time-ordered and non-overlapping.
    Every non-matching channel defers to whatever grouping/compression is
    otherwise configured (composable with ``PatternGrouper``/``PerChannelGrouper``).
    """

    _MARKER = "schema-compression-override"

    def __init__(
        self,
        schema_patterns: list[Pattern[str]],
        compression: CompressionType = CompressionType.NONE,
        *,
        per_channel: bool = False,
    ) -> None:
        self.schema_patterns = schema_patterns
        self.compression = compression
        self.per_channel = per_channel

    def _matches(self, schema: Schema | None) -> bool:
        return schema is not None and any(p.search(schema.name) for p in self.schema_patterns)

    @override
    def chunk_group_key(
        self,
        segment_key: OutputKey,
        channel: Channel,
        schema: Schema | None,
    ) -> str | tuple[str, int] | None:
        if not self._matches(schema):
            return None
        return (self._MARKER, channel.id) if self.per_channel else self._MARKER

    @override
    def chunk_compression(
        self,
        segment_key: OutputKey,
        channel: Channel,
        schema: Schema | None,
    ) -> CompressionType | None:
        return self.compression if self._matches(schema) else None
