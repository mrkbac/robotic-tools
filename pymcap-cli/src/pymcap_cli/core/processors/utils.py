from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from small_mcap import Summary


def global_time_range(summaries: list[Summary | None]) -> tuple[int, int] | None:
    """Extract the global time range across all summaries."""
    start: int | None = None
    end: int | None = None
    for summary in summaries:
        if summary and summary.statistics:
            stats = summary.statistics
            if start is None or stats.message_start_time < start:
                start = stats.message_start_time
            if end is None or stats.message_end_time > end:
                end = stats.message_end_time
    if start is not None and end is not None:
        return start, end
    return None
