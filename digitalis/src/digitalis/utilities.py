import logging
import math
import shlex
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment

logger = logging.getLogger(__name__)


@contextmanager
def function_time(name: str) -> Generator[None, None, None]:
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        duration = (end - start) / 1_000_000  # Convert to milliseconds
        logger.info(f"{name} took {duration:.2f} ms")


NANOSECONDS_PER_SECOND = 1_000_000_000
STRFTIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def nanoseconds_to_iso(timestamp_ns: int) -> str:
    """Convert a timestamp in nanoseconds to ISO 8601 format."""
    return datetime.fromtimestamp(timestamp_ns / NANOSECONDS_PER_SECOND, tz=timezone.utc).strftime(
        STRFTIME_FORMAT
    )


def nanoseconds_duration(ns_total: int) -> str:
    """Format a positive duration in nanoseconds as D:HH:MM:SS.mmm."""
    whole_seconds, rem_ns = divmod(ns_total, 1_000_000_000)
    milliseconds = rem_ns // 1_000_000  # truncate to milliseconds

    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    return f"{days}:{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = min(t2, +1.0)
    t2 = max(t2, -1.0)
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


@dataclass(slots=True, frozen=True)
class RichRender:
    segments: list[Segment]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield from self.segments


def get_file_paths(text: str) -> list[Path]:
    """Extract file paths from a paste event."""
    split_paths = shlex.split(text)
    filepaths = []
    for path_str in split_paths:
        path = Path(path_str)
        if path.exists() and path.is_file():
            filepaths.append(path.resolve())

    return filepaths
