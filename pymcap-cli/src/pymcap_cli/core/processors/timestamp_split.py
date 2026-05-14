from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.boundary_split import BoundarySplitProcessor
from pymcap_cli.core.processors.utils import global_time_range

if TYPE_CHECKING:
    from pymcap_cli.core.processors.base import PipelineContext


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

    def initialize(self, context: PipelineContext) -> None:
        time_range = global_time_range(context)
        if time_range is None:
            # No summary: use sentinel bounds so routing still works. The
            # user-supplied split_points are absolute, so we don't need the
            # actual file range — segments outside the real data range will
            # just stay empty.
            global_start_ns = 0
            global_end_ns = (1 << 63) - 1
            valid = list(self.split_points)
        else:
            global_start_ns, global_end_ns = time_range
            valid = [p for p in self.split_points if global_start_ns < p <= global_end_ns]
        self.boundaries = [global_start_ns, *valid, global_end_ns + 1]
