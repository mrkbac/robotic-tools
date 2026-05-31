"""Concrete OutputProcessor implementations for chunk grouping."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
