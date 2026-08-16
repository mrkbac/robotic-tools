"""Shared lightweight validation for transformed MCAP outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from small_mcap import InvalidMagicError, McapError

from pymcap_cli.core.message_filter import TopicSelection
from pymcap_cli.utils import read_info

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class _McapCounts:
    is_readable: bool
    message_counts_by_topic: dict[str, int] | None = None


def _read_mcap_counts(path: Path) -> _McapCounts:
    try:
        with path.open("rb") as stream:
            info = read_info(stream)
    except (McapError, InvalidMagicError, OSError, AssertionError):
        return _McapCounts(is_readable=False)

    statistics = info.summary.statistics
    if statistics is None:
        return _McapCounts(is_readable=True)

    channel_counts = statistics.channel_message_counts
    if not channel_counts:
        counts_by_topic = {} if statistics.message_count == 0 else None
        return _McapCounts(
            is_readable=True,
            message_counts_by_topic=counts_by_topic,
        )
    if sum(channel_counts.values()) != statistics.message_count:
        return _McapCounts(is_readable=True)

    counts_by_topic: dict[str, int] = {}
    for channel_id, count in channel_counts.items():
        channel = info.summary.channels.get(channel_id)
        if channel is None:
            return _McapCounts(is_readable=True)
        counts_by_topic[channel.topic] = counts_by_topic.get(channel.topic, 0) + count
    return _McapCounts(
        is_readable=True,
        message_counts_by_topic=counts_by_topic,
    )


def validate_mcap_outputs(
    sources: Sequence[str | Path],
    outputs: Sequence[Path],
    *,
    preserved_topic_patterns: Sequence[str] = (),
    lossy_topic_patterns: Sequence[str] = (),
) -> str | None:
    """Validate output structure and message preservation.

    Every source topic selected by ``preserved_topic_patterns`` and not matched
    by a lossy pattern must retain its message count. Channel IDs are intentionally
    aggregated by topic because transforms may replace channels.
    """
    if not outputs:
        return "no output files were produced"

    output_counts = [(path, _read_mcap_counts(path)) for path in outputs]
    invalid_outputs = [path for path, counts in output_counts if not counts.is_readable]
    if invalid_outputs:
        paths = ", ".join(str(path) for path in invalid_outputs)
        return f"output failed MCAP validation: {paths}"

    unavailable_outputs = [
        path for path, counts in output_counts if counts.message_counts_by_topic is None
    ]
    if unavailable_outputs:
        paths = ", ".join(str(path) for path in unavailable_outputs)
        return f"per-topic message counts unavailable: {paths}"

    if ".*" in lossy_topic_patterns:
        return None

    local_sources = [
        Path(source) for source in sources if urlparse(str(source)).scheme not in ("http", "https")
    ]
    source_counts = [(path, _read_mcap_counts(path)) for path in local_sources]

    if any(counts.message_counts_by_topic is None for _, counts in source_counts):
        return None

    source_by_topic = _sum_topic_counts(source_counts)
    output_by_topic = _sum_topic_counts(output_counts)
    preserved_topics = TopicSelection.from_patterns(
        include=preserved_topic_patterns,
        exclude=lossy_topic_patterns,
    )
    losses = [
        f"{topic} ({source_count} -> {output_by_topic.get(topic, 0)})"
        for topic, source_count in sorted(source_by_topic.items())
        if preserved_topics.selects(topic) and output_by_topic.get(topic, 0) < source_count
    ]
    if not losses:
        return None
    return f"output lost messages on preserved topics: {', '.join(losses)}"


def _sum_topic_counts(
    counts_by_path: Sequence[tuple[Path, _McapCounts]],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for _, counts in counts_by_path:
        assert counts.message_counts_by_topic is not None
        for topic, count in counts.message_counts_by_topic.items():
            totals[topic] = totals.get(topic, 0) + count
    return totals
