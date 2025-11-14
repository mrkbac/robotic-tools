import io


class DebugStreamWrapper(io.RawIOBase):
    """Wrapper for file streams that tracks I/O statistics."""

    def __init__(self, stream: io.RawIOBase) -> None:
        self.stream = stream
        self.read_calls = 0
        self.read_into_calls = 0
        self.read_bytes = 0
        self.seek_calls = 0
        self.tell_calls = 0

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        data = self.stream.read(size)
        self.read_bytes += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        self.seek_calls += 1
        return self.stream.seek(offset, whence)

    def tell(self) -> int:
        self.tell_calls += 1
        return self.stream.tell()

    def seekable(self) -> bool:
        return self.stream.seekable()

    def readable(self) -> bool:
        return self.stream.readable()

    def writable(self) -> bool:
        return self.stream.writable()

    def readinto(self, b) -> int | None:  # type: ignore[no-untyped-def]  # noqa: ANN001
        self.read_into_calls += 1
        n = self.stream.readinto(b)
        if n is not None:
            self.read_bytes += n
        return n

    def truncate(self, size: int | None = None) -> int:
        return self.stream.truncate(size)

    def flush(self) -> None:
        return self.stream.flush()

    @property
    def closed(self) -> bool:
        return self.stream.closed

    def close(self) -> None:
        return self.stream.close()

    def print_stats(self, total_file_size: int) -> None:
        # """Print debug statistics."""
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415

        console = Console()
        table = Table(title="Debug I/O Statistics")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Read Calls", f"{self.read_calls:,}")
        table.add_row("Readinto Calls", f"{self.read_into_calls:,}")
        table.add_row("Read Bytes", f"{self.read_bytes:,}")
        percent_read = (self.read_bytes / total_file_size * 100) if total_file_size > 0 else 0
        table.add_row("Percent of File Read", f"{percent_read:.2f}%")
        table.add_row("Seek Calls", f"{self.seek_calls:,}")
        table.add_row("Tell Calls", f"{self.tell_calls:,}")

        console.print(table)
