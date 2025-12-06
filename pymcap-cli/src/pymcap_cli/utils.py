import io
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

    def _update_progress(self) -> None:
        """Update progress bar based on current position."""
        if self._last_position > self._max_position:
            delta = self._last_position - self._max_position
            self._progress.advance(self._progress_task, delta)
            self._max_position = self._last_position

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
        if hasattr(self._stream, "flush"):
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
