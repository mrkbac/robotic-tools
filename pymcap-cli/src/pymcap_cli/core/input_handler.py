"""Unified input handler for local files and HTTP URLs."""

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, cast
from urllib.parse import ParseResult, urlparse

from pymcap_cli.debug_wrapper import DebugStreamWrapper
from pymcap_cli.http_utils import open_http_stream


def resolve_mcap_path(path: str) -> str:
    """Resolve a ROS 2-style bag directory to its inner ``.mcap`` file.

    ROS 2 lays a recording out as ``<bagname>/<bagname>.mcap``. When ``path``
    points at such a directory, return the inner file so callers can pass just
    the folder. Anything else (a regular file, a URL, a directory without the
    matching inner file) is returned unchanged.
    """
    candidate_dir = Path(path)
    if candidate_dir.is_dir():
        inner = candidate_dir / f"{candidate_dir.name}.mcap"
        if inner.is_file():
            return str(inner)
    return path


def _open_path_file(url: ParseResult) -> tuple[io.RawIOBase, int]:
    file_path = Path(resolve_mcap_path(url.path))
    raw_stream = file_path.open("rb", buffering=0)
    size = file_path.stat().st_size
    return raw_stream, size


REGISTRY: dict[str, Callable[[ParseResult], tuple[io.RawIOBase, int]]] = {
    "http": open_http_stream,
    "https": open_http_stream,
    "file": _open_path_file,
    "": _open_path_file,
}


@contextmanager
def open_input(
    path: str, buffering: int = 8192, *, debug: bool = False
) -> Iterator[tuple[IO[bytes], int]]:
    result = urlparse(path)
    opener = REGISTRY.get(result.scheme, _open_path_file)
    if opener is None:
        raise ValueError(f"Unsupported URL scheme: {result.scheme}")

    base_stream: io.RawIOBase | io.BufferedIOBase | None = None
    debug_wrapper: DebugStreamWrapper | None = None
    size = 0
    try:
        original_stream, size = opener(result)

        # Optionally wrap in debug wrapper (cast since DebugStreamWrapper implements interface)
        if debug:
            debug_wrapper = DebugStreamWrapper(original_stream)
            base_stream = debug_wrapper
        else:
            base_stream = original_stream

        # Apply buffering
        final_stream: io.RawIOBase | io.BufferedIOBase | io.BufferedReader
        if buffering == 0 or isinstance(base_stream, io.BufferedIOBase):
            final_stream = base_stream
        else:
            final_stream = io.BufferedReader(base_stream, buffer_size=buffering)

        yield cast("IO[bytes]", final_stream), size
    finally:
        if base_stream:
            base_stream.close()

        if debug_wrapper:
            debug_wrapper.print_stats(size)

    return
