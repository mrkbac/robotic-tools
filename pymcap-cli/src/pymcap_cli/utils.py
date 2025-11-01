from typing import IO

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


def bytes_to_human(size_bytes: float) -> str:
    """Convert bytes to a human-readable format."""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PiB"


def file_progress(title: str, console: Console | None = None) -> Progress:
    return Progress(
        TextColumn(title),
        TimeElapsedColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    )


class ProgressTrackingIO:
    """Read-only IO[bytes] wrapper that updates progress bar as data is read.

    Wraps a binary stream and automatically updates a Rich progress bar
    based on the current file position. Useful for long-running read operations.

    This class implements the IO[bytes] protocol methods for reading and seeking.
    Write operations are not supported and will raise NotImplementedError.
    This wrapper is not thread-safe.

    Example:
        Basic usage with file reading:

        >>> from pathlib import Path
        >>> with file_progress("Reading MCAP...", console) as progress:
        ...     task = progress.add_task("Processing", total=file_size)
        ...     with Path("data.mcap").open("rb") as f:
        ...         wrapped = ProgressTrackingIO(f, task, progress)
        ...         while chunk := wrapped.read(8192):
        ...             process_chunk(chunk)  # Progress bar updates automatically

        Integration with stream processing functions:

        >>> def process_stream(stream: IO[bytes]) -> None:
        ...     # This function works with any IO[bytes], including ProgressTrackingIO
        ...     data = stream.read()
        ...     return parse_data(data)
        >>>
        >>> with file_progress("Processing...", console) as progress:
        ...     task = progress.add_task("Reading", total=file_size)
        ...     wrapped = ProgressTrackingIO(file_handle, task, progress)
        ...     result = process_stream(wrapped)
    """

    def __init__(self, stream: IO[bytes], progress_task: TaskID, progress_obj: Progress) -> None:
        """Initialize the progress-tracking wrapper.

        Args:
            stream: The underlying binary stream to wrap
            progress_task: The Rich progress task ID to update
            progress_obj: The Rich Progress instance
        """
        self._stream = stream
        self._progress_task = progress_task
        self._progress = progress_obj

    def read(self, size: int = -1) -> bytes:
        """Read bytes and update progress."""
        data = self._stream.read(size)
        self._progress.update(self._progress_task, completed=self._stream.tell())
        return data

    def readinto(self, b: memoryview) -> int | None:
        """Read bytes into buffer and update progress."""
        result: int | None = self._stream.readinto(b)  # type: ignore[attr-defined]
        self._progress.update(self._progress_task, completed=self._stream.tell())
        return result

    def readable(self) -> bool:
        """Return whether the stream is readable."""
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position and update progress."""
        result = self._stream.seek(offset, whence)
        self._progress.update(self._progress_task, completed=result)
        return result

    def tell(self) -> int:
        """Return current position."""
        return self._stream.tell()

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

    def write(self, b: bytes) -> int:
        """Write operation not supported."""
        raise NotImplementedError("ProgressTrackingIO is read-only")

    def writelines(self, lines: list[bytes]) -> None:
        """Write operation not supported."""
        raise NotImplementedError("ProgressTrackingIO is read-only")

    def truncate(self, size: int | None = None) -> int:
        """Truncate operation not supported."""
        raise NotImplementedError("ProgressTrackingIO is read-only")

    def fileno(self) -> int:
        """Return underlying file descriptor."""
        return self._stream.fileno()

    def isatty(self) -> bool:
        """Return whether this is a tty device."""
        return self._stream.isatty()
