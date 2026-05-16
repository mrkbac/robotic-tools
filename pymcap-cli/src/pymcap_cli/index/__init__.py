"""Sidecar index package.

The :data:`SANE_EPOCH_NS` constant is shared across the scanner, the read-side
CLI commands, and SQL expressions that gate "did this clock look initialised
before we trust its timestamps." Frozen migrations capture their own copy at
the moment they were authored — they intentionally do not import this value.
"""

from __future__ import annotations

# 2000-01-01T00:00:00Z in nanoseconds. Anything earlier almost certainly comes
# from an uninitialised clock (e.g. ROS time before NTP sync).
SANE_EPOCH_NS = 946_684_800 * 1_000_000_000
