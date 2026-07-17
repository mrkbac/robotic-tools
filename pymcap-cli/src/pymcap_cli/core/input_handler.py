"""Unified input handler for local files and HTTP URLs."""

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, cast
from urllib.parse import ParseResult, urlparse

from pymcap_cli.debug_wrapper import DebugStreamWrapper
from pymcap_cli.http_utils import open_http_stream

_debug_io_default = False


def set_debug_io_default(enabled: bool) -> None:
    """Wrap every subsequent ``open_input`` stream in the I/O statistics wrapper.

    Set by the CLI's global ``--debug-io`` flag; individual call sites can still
    opt in explicitly via ``open_input(..., debug=True)``.
    """
    global _debug_io_default
    _debug_io_default = enabled


def resolve_mcap_path(path: str) -> str:
    """Resolve a rosbag2 bag directory to its single ``.mcap`` file.

    rosbag2 lays a recording out as ``<bagname>/<bagname>_<N>.mcap`` splits.
    For single-file commands this resolves a directory holding exactly one split
    to that file. A multi-split directory (which has no single-file meaning here)
    or a directory with no resolvable MCAP raises ``ValueError`` rather than
    surfacing an opaque ``IsADirectoryError`` downstream. Files and URLs pass
    through unchanged.
    """
    candidate_dir = Path(path)
    if not candidate_dir.is_dir():
        return path

    from pymcap_cli.core.rosbag2_layout import find_bag_splits  # noqa: PLC0415

    splits = find_bag_splits(candidate_dir)
    if len(splits) == 1:
        return str(splits[0])
    if not splits:
        raise ValueError(f"{path!r} is not an MCAP file or a rosbag2 bag directory")
    raise ValueError(
        f"{path!r} is a multi-split rosbag2 directory ({len(splits)} files); "
        f"this command reads a single file — use 'info' or 'merge'"
    )


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
        if debug or _debug_io_default:
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
            debug_wrapper.print_stats(size, name=path)

    return
