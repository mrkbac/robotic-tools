import math

from digitalis.utilities import (
    RichRender,
    get_file_paths,
    nanoseconds_duration,
    nanoseconds_to_iso,
    quaternion_to_euler,
)
from rich.console import Console
from rich.segment import Segment


def test_nanoseconds_to_iso_uses_utc() -> None:
    assert nanoseconds_to_iso(1_500_000_000) == "1970-01-01T00:00:01.500000Z"


def test_nanoseconds_duration_formats_days_and_truncates_submilliseconds() -> None:
    duration = ((2 * 24 + 3) * 60 * 60 + 4 * 60 + 5) * 1_000_000_000 + 6_999_999

    assert nanoseconds_duration(duration) == "2:03:04:05.006"


def test_quaternion_to_euler_identity() -> None:
    assert quaternion_to_euler(0.0, 0.0, 0.0, 1.0) == (0.0, 0.0, 0.0)


def test_quaternion_to_euler_clamps_pitch() -> None:
    roll, pitch, yaw = quaternion_to_euler(0.0, 1.0, 0.0, 1.0)

    assert math.isfinite(roll)
    assert pitch == math.pi / 2
    assert math.isfinite(yaw)


def test_rich_render_yields_segments() -> None:
    segments = [Segment("hello")]

    assert list(RichRender(segments).__rich_console__(Console(), Console().options)) == segments


def test_get_file_paths_accepts_quoted_files_and_ignores_other_paths(tmp_path) -> None:
    first = tmp_path / "first file.mcap"
    second = tmp_path / "second.mcap"
    directory = tmp_path / "directory"
    first.write_bytes(b"")
    second.write_bytes(b"")
    directory.mkdir()

    assert get_file_paths(f'"{first}" {directory} missing.mcap "{second}"') == [
        first.resolve(),
        second.resolve(),
    ]
