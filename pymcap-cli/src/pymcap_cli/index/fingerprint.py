"""Cheap byte probe for MCAP files.

Reads up to ``HEAD_BYTES`` from the start and ``TAIL_BYTES`` from the end of a
file plus the file size, and hashes them with ``xxh3_128``. The result is
hex-encoded for fast moved/copy detection.

This is neither a cryptographic hash nor exact byte identity. It deliberately
avoids full-file reads so scanning multi-terabyte datasets stays bounded.

Switching algorithms would re-fingerprint every file on the next scan, so
``xxhash`` is a hard requirement — no runtime fallback.
"""

from __future__ import annotations

import struct
from typing import IO, TYPE_CHECKING

import xxhash

if TYPE_CHECKING:
    from pathlib import Path

HEAD_BYTES = 64 * 1024
TAIL_BYTES = 64 * 1024
_SIZE_STRUCT = struct.Struct("<Q")


def fingerprint_stream(stream: IO[bytes], size_bytes: int) -> str:
    """Compute the fingerprint of an open seekable stream.

    The stream position is not guaranteed to be preserved.
    """
    hasher = xxhash.xxh3_128()
    head_len = min(HEAD_BYTES, size_bytes)
    stream.seek(0)
    hasher.update(stream.read(head_len))
    if size_bytes > HEAD_BYTES:
        tail_offset = max(size_bytes - TAIL_BYTES, head_len)
        stream.seek(tail_offset)
        hasher.update(stream.read(size_bytes - tail_offset))
    hasher.update(_SIZE_STRUCT.pack(size_bytes))
    return hasher.hexdigest()


def fingerprint_path(path: Path) -> tuple[str, int]:
    """Fingerprint the file at ``path``. Returns (fingerprint, size_bytes)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        return fingerprint_stream(f, size), size
