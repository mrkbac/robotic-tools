import io
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import IO, Any

from rich import filesize
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text
from small_mcap import RebuildInfo, get_header, get_summary, rebuild_summary

from pymcap_cli.osc_utils import OSCProgressColumn


def bytes_to_human(size_bytes: float | None) -> str:
    """Convert bytes to a human-readable format."""
    if size_bytes is None:
        return "N/A"

    return filesize.decimal(int(abs(size_bytes)), separator="")


def file_progress(title: str, console: Console | None = None) -> Progress:
    """Create a file processing progress bar with auto-detected OSC 9;4 support.

    Args:
        title: Progress bar title text (also used for terminal title)
        console: Rich console instance (None for default)

    Returns:
        Configured Rich Progress instance with OSC support if available
    """
    # Strip Rich markup from title for terminal title
    clean_title = Text.from_markup(title).plain

    return Progress(
        TextColumn(title),
        TimeElapsedColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        OSCProgressColumn(title=clean_title),
        console=console,
    )


class ProgressTrackingIO(io.RawIOBase, IO[bytes]):
    """Read-only IO[bytes] wrapper that updates progress bar as data is read.

    Wraps a binary stream and automatically updates a Rich progress bar
    using incremental progress tracking. Works correctly for both single-stream
    and multi-stream scenarios (e.g., merging multiple files).

    This class implements the IO[bytes] protocol methods for reading and seeking.
    Write operations are not supported and will raise NotImplementedError.
    This wrapper is not thread-safe.

    Example:
        Basic usage with file reading:

        >>> from pathlib import Path
        >>> with file_progress("Reading MCAP...", console) as progress:
        ...     task = progress.add_task("Processing", total=file_size)
        ...     with Path("data.mcap").open("rb") as f:
        ...         wrapped = ProgressTrackingIO(f, task, progress, f.tell())
        ...         while chunk := wrapped.read(8192):
        ...             process_chunk(chunk)  # Progress bar updates automatically

        Multi-stream usage (e.g., merging files):

        >>> total_size = sum(file_sizes)
        >>> with file_progress("Merging files...", console) as progress:
        ...     task = progress.add_task("Processing", total=total_size)
        ...     wrapped_streams = [
        ...         ProgressTrackingIO(stream, task, progress, stream.tell())
        ...         for stream in input_streams
        ...     ]
        ...     # Process streams in any order - progress advances smoothly
        ...     for stream in wrapped_streams:
        ...         data = stream.read(8192)
    """

    # Minimum bytes between progress updates to reduce overhead
    _UPDATE_THRESHOLD = 1_000_000  # 1MB

    def __init__(
        self,
        stream: IO[bytes],
        progress_task: TaskID,
        progress_obj: Progress,
        initial_offset: int = 0,
    ) -> None:
        """Initialize the progress-tracking wrapper.

        Args:
            stream: The underlying binary stream to wrap
            progress_task: The Rich progress task ID to update
            progress_obj: The Rich Progress instance
            initial_offset: Current position in stream (use stream.tell() for multi-stream)
        """
        self._stream = stream
        self._progress_task = progress_task
        self._progress = progress_obj
        self._last_position = initial_offset
        self._max_position = initial_offset  # Track high water mark
        self._pending_delta = 0  # Accumulated delta waiting to be reported

    def _update_progress(self) -> None:
        """Update progress bar based on current position.

        Uses batched updates to reduce overhead from frequent small reads.
        Progress is only reported when accumulated delta exceeds threshold.
        """
        if self._last_position > self._max_position:
            delta = self._last_position - self._max_position
            self._pending_delta += delta
            self._max_position = self._last_position

            # Only update progress bar when we've accumulated enough
            if self._pending_delta >= self._UPDATE_THRESHOLD:
                self._progress.advance(self._progress_task, self._pending_delta)
                self._pending_delta = 0

    def flush_progress(self) -> None:
        """Flush any pending progress updates.

        Call this when processing is complete to ensure the progress bar
        shows the final position accurately.
        """
        if self._pending_delta > 0:
            self._progress.advance(self._progress_task, self._pending_delta)
            self._pending_delta = 0

    def read(self, size: int = -1) -> bytes:
        """Read bytes and advance progress by delta."""
        data = self._stream.read(size)
        delta = len(data)
        if delta > 0:
            self._last_position += delta
            self._update_progress()
        return data

    def readinto(self, b: Any) -> int | None:
        """Read bytes into buffer and advance progress by delta."""
        result: int | None = self._stream.readinto(b)  # type: ignore[attr-defined]
        if result is not None and result > 0:
            self._last_position += result
            self._update_progress()
        return result

    def readable(self) -> bool:
        """Return whether the stream is readable."""
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek and advance progress by delta."""
        result = self._stream.seek(offset, whence)
        self._last_position = result
        self._update_progress()
        return result

    def tell(self) -> int:
        """Return current position."""
        return self._last_position

    def seekable(self) -> bool:
        """Return whether the stream is seekable."""
        return self._stream.seekable()

    def close(self) -> None:
        """Close the underlying stream."""
        self._stream.close()

    @property
    def closed(self) -> bool:
        """Return whether the stream is closed."""
        return self._stream.closed

    def flush(self) -> None:
        """Flush the stream (no-op for read-only)."""
        self._stream.flush()

    def writable(self) -> bool:
        """Return whether the stream is writable (always False)."""
        return False

    def fileno(self) -> int:
        """Return underlying file descriptor."""
        return self._stream.fileno()

    def isatty(self) -> bool:
        """Return whether this is a tty device."""
        return self._stream.isatty()


def rebuild_info(
    f: IO[bytes], file_size: int, *, exact_sizes: bool = False, console: Console | None = None
) -> RebuildInfo:
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
        wrapped_stream = ProgressTrackingIO(f, task, progress, f.tell())

        result = rebuild_summary(
            wrapped_stream,
            validate_crc=False,
            calculate_channel_sizes=True,
            exact_sizes=exact_sizes,
        )

        # Flush any pending progress updates before completing
        wrapped_stream.flush_progress()

        # Complete progress
        progress.update(task, completed=file_size, visible=False)

    return result


def read_info(f: IO[bytes]) -> RebuildInfo:
    """Read existing MCAP summary from file.

    Args:
        f: Input file stream

    Returns:
        Info (RebuildInfo) with header, summary, and no channel_sizes
    """
    header = get_header(f)
    summary = get_summary(f)
    assert summary is not None, "Summary should not be None"
    return RebuildInfo(header=header, summary=summary)


# Maximum value for a signed 64-bit integer (used for unbounded time range)
MAX_INT64 = 2**63 - 1


class MetadataMode(str, Enum):
    """Metadata inclusion mode."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


class AttachmentsMode(str, Enum):
    """Attachments inclusion mode."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


def parse_time_arg(time_str: str) -> int:
    """Parse time argument that can be nanoseconds or RFC3339 date."""
    if not time_str:
        return 0

    # Try parsing as integer nanoseconds first
    try:
        return int(time_str)
    except ValueError:
        pass

    # Try parsing as RFC3339 date
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except ValueError:
        raise ValueError(
            f"Invalid time format: {time_str}. Use nanoseconds or RFC3339 format"
        ) from None


def compile_topic_patterns(patterns: list[str]) -> list[Pattern[str]]:
    """Compile topic regex patterns."""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    return compiled


def parse_timestamp_args(date_or_nanos: str, nanoseconds: int, seconds: int) -> int:
    """Parse timestamp with precedence: date_or_nanos > nanoseconds > seconds."""
    if date_or_nanos:
        return parse_time_arg(date_or_nanos)
    if nanoseconds != 0:
        return nanoseconds
    return seconds * 1_000_000_000


def confirm_output_overwrite(output: Path, force: bool) -> None:
    """Confirm overwrite if output exists and force=False.

    Args:
        output: Output file path
        force: If True, skip confirmation

    Raises:
        SystemExit: If user declines to overwrite
    """
    if output.exists() and not force:
        response = input(f"Output file '{output}' already exists. Overwrite? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")  # noqa: T201
            raise SystemExit(1)
