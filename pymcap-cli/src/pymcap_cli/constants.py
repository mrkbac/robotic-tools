"""Shared dependency-free constants."""

from __future__ import annotations

import sys
from typing import Final

DEFAULT_CHUNK_SIZE: Final = 4 * 1024 * 1024
DEFAULT_COMPRESSION: Final = "zstd"

# Max log-time span of a single chunk for roscompress output. Already-compressed
# payloads (video/point clouds) have a low byte-rate per second of log time, so a
# size-only chunk boundary would let one chunk cover minutes; capping the span
# keeps chunks time-local and seek-friendly.
DEFAULT_ROSCOMPRESS_CHUNK_SPAN_NS: Final = 10 * 1_000_000_000

# small_mcap uses sys.maxsize as the unbounded time-range sentinel.
MAX_INT64: Final = sys.maxsize
NS_TO_MS: Final = 1_000_000
NS_TO_SEC: Final = 1_000_000_000
