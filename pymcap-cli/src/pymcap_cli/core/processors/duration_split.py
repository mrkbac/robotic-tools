from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.boundary_split import BoundarySplitProcessor
from pymcap_cli.core.processors.utils import global_time_range

if TYPE_CHECKING:
    from small_mcap import Summary


class DurationSplitProcessor(BoundarySplitProcessor):
    """Split output every N nanoseconds.

    Preserves the COPY fast-path for chunks that fall entirely within one segment.
    Only chunks spanning a segment boundary are decoded.
    """

    def __init__(self, duration_ns: int) -> None:
        super().__init__()
        if duration_ns <= 0:
            raise ValueError(f"duration_ns must be positive, got {duration_ns}")
        self.duration_ns = duration_ns

    def initialize(self, summaries: list[Summary | None]) -> None:
        time_range = global_time_range(summaries)
        if time_range is None:
            return
        global_start_ns, global_end_ns = time_range
        self.boundaries = []
        timestamp = global_start_ns
        while timestamp <= global_end_ns:
            self.boundaries.append(timestamp)
            timestamp += self.duration_ns
        if not self.boundaries or self.boundaries[-1] <= global_end_ns:
            self.boundaries.append(global_end_ns + 1)
