"""Override and normalize ROS 2 channel QoS metadata."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from typing_extensions import override

from pymcap_cli.core.processors.base import (
    Action,
    ChannelContext,
    ChunkContext,
    ChunkDecision,
    InputProcessor,
)
from pymcap_cli.core.qos import YamlValue, qos_profiles_to_numeric, qos_profiles_with_overrides

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from re import Pattern

    from small_mcap import Channel, Chunk, LazyChunk, Schema


class QosMetadataProcessor(InputProcessor):
    """Apply per-topic overrides and optionally convert policy names to codes."""

    def __init__(
        self,
        *,
        qos_format: Literal["preserve", "numeric"],
        topic_overrides: Mapping[str, Mapping[str, YamlValue]] | None = None,
        set_rules: Sequence[tuple[str, str, YamlValue]] = (),
    ) -> None:
        self._qos_format = qos_format
        self._topic_overrides = {
            topic: dict(profile) for topic, profile in (topic_overrides or {}).items()
        }
        self._set_rules: list[tuple[Pattern[str], str, YamlValue]] = []
        for pattern, field, value in set_rules:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                msg = f"Invalid QoS topic regex {pattern!r}: {exc}"
                raise ValueError(msg) from exc
            self._set_rules.append((compiled, field, value))

    def _overrides_for_topic(self, topic: str) -> dict[str, YamlValue]:
        overrides = dict(self._topic_overrides.get(topic, {}))
        overrides.update(
            {field: value for pattern, field, value in self._set_rules if pattern.fullmatch(topic)}
        )
        return overrides

    @override
    def on_channel(
        self, context: ChannelContext, channel: Channel, schema: Schema | None
    ) -> Action:
        overrides = self._overrides_for_topic(channel.topic)
        raw = channel.metadata.get("offered_qos_profiles")
        if overrides and raw is None:
            msg = (
                f"Cannot rewrite QoS metadata for topic {channel.topic!r}: "
                "matched a QoS override but has no offered_qos_profiles"
            )
            raise ValueError(msg)
        try:
            if overrides and raw is not None:
                raw = qos_profiles_with_overrides(raw, overrides)
                channel.metadata["offered_qos_profiles"] = raw
            if self._qos_format == "numeric" and raw is not None:
                raw = qos_profiles_to_numeric(raw)
                channel.metadata["offered_qos_profiles"] = raw
        except (TypeError, ValueError) as exc:
            msg = f"Cannot rewrite QoS metadata for topic {channel.topic!r}: {exc}"
            raise ValueError(msg) from exc
        return Action.CONTINUE

    @override
    def on_chunk(
        self,
        context: ChunkContext,
        chunk: Chunk | LazyChunk,
    ) -> ChunkDecision:
        return ChunkDecision.DECODE_VERIFY
