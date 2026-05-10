"""Shared dependency-free constants."""

from __future__ import annotations

import sys
from typing import Final

DEFAULT_CHUNK_SIZE: Final = 4 * 1024 * 1024
DEFAULT_COMPRESSION: Final = "zstd"

# small_mcap uses sys.maxsize as the unbounded time-range sentinel.
MAX_INT64: Final = sys.maxsize
NS_TO_MS: Final = 1_000_000
NS_TO_SEC: Final = 1_000_000_000
