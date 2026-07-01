"""Pre-scan coverage hints derived from the MCAP summary section.

Every exporter runs by scanning the file end to end. The driver already reads
the file's summary once (to size the progress bar); these helpers turn that
same summary — no extra read — into warnings when the run will be slow or
produce empty / sparse output:

- no summary — the scan can't be size-estimated and tends to be slow; the file
  should be rebuilt (``pymcap-cli recover``) to add one;
- a requested topic is absent from the summary — it is misnamed or empty, so
  the output for it comes up blank;
- a requested topic has far fewer messages than the busiest requested topic —
  its output is sparse.

All hints are advisory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from small_mcap import Channel, Statistics, Summary

logger = logging.getLogger(__name__)

# A requested topic present in the summary but this far below the busiest
# requested topic is flagged as "sparse". The floor of 2 also catches
# single-message topics even when every requested topic is small.
_SPARSE_RATIO_DIVISOR = 100
_SPARSE_MIN = 2


def _topic_counts(stats: Statistics, channels: Mapping[int, Channel]) -> dict[str, int]:
    """Per-topic message counts from a summary's statistics record.

    May be empty when the statistics record carries no per-channel counts
    (some writers record only the file-level ``message_count``).
    """
    counts: dict[str, int] = {}
    for channel_id, count in stats.channel_message_counts.items():
        channel = channels.get(channel_id)
        if channel is None:
            continue
        counts[channel.topic] = counts.get(channel.topic, 0) + int(count)
    return counts


def warn_topic_coverage(summary: Summary | None, file: str, topics: list[str] | None) -> None:
    """Warn about a missing summary and, when given, missing / sparse topics.

    ``summary`` is the summary the driver already read for ``file`` (``None``
    when the file has none). ``topics`` is the explicit topic list the caller
    will export (e.g. plot's message-path topics); when it is ``None`` / empty
    the exporter reads every topic, so the topic checks are skipped.

    A missing statistics record and a present-but-empty one are distinct: the
    former means the scan can't be sized (slow); the latter means the file is
    genuinely empty (the export produces nothing).
    """
    if summary is None or summary.statistics is None:
        logger.warning(
            f"No usable summary in {file} — the export must scan the whole file, which is "
            "slow and gives no accurate progress total. Rebuild it with `pymcap-cli recover` "
            "to add one."
        )
        return

    stats = summary.statistics
    if stats.message_count == 0:
        logger.warning(f"{file} has a summary but reports 0 messages — the export will be empty.")
        return

    counts = _topic_counts(stats, summary.channels)
    if not topics or not counts:
        # No explicit topics to check, or no per-channel breakdown to check them
        # against (some writers record only the file-level message_count).
        return

    max_count = max(counts.values())
    threshold = max(_SPARSE_MIN, max_count // _SPARSE_RATIO_DIVISOR)
    for topic in topics:
        count = counts.get(topic, 0)  # absent topic and 0-message channel are both "blank"
        if count == 0:
            logger.warning(
                f"Topic {topic!r} has no messages in {file} (per its summary) — it may be "
                "empty or misnamed; its output will be blank."
            )
        elif count < threshold:
            logger.warning(
                f"Topic {topic!r} has only {count:,} message(s) in {file} (vs up to "
                f"{max_count:,} on other requested topics) — its output will be sparse."
            )
