"""Thin wrapper around small-mcap rebuild functionality with progress UI."""

from typing import BinaryIO

from rich.console import Console
from small_mcap import RebuildInfo as Info
from small_mcap import get_header, get_summary, rebuild_summary

from pymcap_cli.utils import ProgressTrackingIO, file_progress

__all__ = ["Info", "read_info", "rebuild_info"]

console = Console()


def rebuild_info(f: BinaryIO, file_size: int, *, exact_sizes: bool = False) -> Info:
    """Rebuild MCAP summary with progress bar.

    Thin wrapper around small-mcap's rebuild_summary that adds progress visualization.

    Args:
        f: Input file stream
        file_size: Total file size for progress tracking
        exact_sizes: Whether to calculate exact sizes (True) or estimate from indexes (False)

    Returns:
        Info (RebuildInfo) with header, summary, and channel_sizes
    """
    with file_progress("[bold blue]Rebuilding MCAP info...", console) as progress:
        task = progress.add_task("Processing", total=file_size)

        # Wrap stream to track progress
        wrapped_stream = ProgressTrackingIO(f, task, progress)

        result = rebuild_summary(
            wrapped_stream,  # type: ignore[arg-type]
            validate_crc=False,
            calculate_channel_sizes=True,
            exact_sizes=exact_sizes,
        )

        # Complete progress
        progress.update(task, completed=file_size, visible=False)

    return result


def read_info(f: BinaryIO) -> Info:
    """Read existing MCAP summary from file.

    Args:
        f: Input file stream

    Returns:
        Info (RebuildInfo) with header, summary, and no channel_sizes
    """
    header = get_header(f)
    summary = get_summary(f)
    assert summary is not None, "Summary should not be None"
    return Info(header=header, summary=summary)
