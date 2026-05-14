from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymcap_cli.core.processors.base import PipelineContext


def global_time_range(context: PipelineContext) -> tuple[int, int] | None:
    """Extract the global time range across all summaries."""
    start: int | None = None
    end: int | None = None
    for input_context in context.inputs:
        if input_context.statistics is not None:
            stats = input_context.statistics
            if start is None or stats.message_start_time < start:
                start = stats.message_start_time
            if end is None or stats.message_end_time > end:
                end = stats.message_end_time
    if start is not None and end is not None:
        return start, end
    return None
