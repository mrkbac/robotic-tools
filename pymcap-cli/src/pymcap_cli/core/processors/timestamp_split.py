from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.boundary_split import BoundarySplitProcessor
from pymcap_cli.core.processors.utils import global_time_range

if TYPE_CHECKING:
    from small_mcap import Summary


class TimestampSplitProcessor(BoundarySplitProcessor):
    """Split output at specific nanosecond timestamps.

    Given split points [t1, t2], creates 3 segments:
    [global_start, t1), [t1, t2), [t2, global_end].

    Preserves the COPY fast-path for non-boundary chunks.
    """

    def __init__(self, split_points: list[int]) -> None:
        super().__init__()
        if not split_points:
            raise ValueError("split_points must not be empty")
        self.split_points = sorted(split_points)

    def initialize(self, summaries: list[Summary | None]) -> None:
        time_range = global_time_range(summaries)
        if time_range is None:
            return
        global_start_ns, global_end_ns = time_range
        valid = [point for point in self.split_points if global_start_ns < point <= global_end_ns]
        self.boundaries = [global_start_ns, *valid, global_end_ns + 1]
