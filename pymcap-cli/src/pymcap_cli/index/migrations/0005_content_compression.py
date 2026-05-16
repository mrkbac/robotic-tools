"""V5 schema — store per-content compression aggregates on ``content``.

We dropped the per-chunk byte / compression info with ``content_chunk`` in
0002 because nothing was reading it, but the *file-level* aggregate is
genuinely useful — it answers "is this MCAP zstd / lz4 / uncompressed?"
and "what's the compression ratio?" without re-reading the file.

Adds three nullable columns to ``content``:

- ``compression``: comma-joined sorted set of distinct codec names seen
  across the file's chunks (``""`` if chunks exist but none compress,
  ``NULL`` if no chunk index info was available e.g. ``scan_kind = 'rebuilt'``).
- ``compressed_size_bytes``: ``SUM(chunk.compressed_size)`` — total
  on-disk bytes after compression.
- ``uncompressed_size_bytes``: ``SUM(chunk.uncompressed_size)`` — total
  message-payload bytes before compression.

Existing rows get NULL — they'll be backfilled on the next ``index scan``
(or stay NULL forever, which the readers handle gracefully). No table
rewrite, just ``ALTER TABLE ADD COLUMN`` which SQLite does in O(1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "store compression / compressed-size / uncompressed-size on content"

_ADD_COLUMNS = """
ALTER TABLE content ADD COLUMN compression              TEXT;
ALTER TABLE content ADD COLUMN compressed_size_bytes    INTEGER;
ALTER TABLE content ADD COLUMN uncompressed_size_bytes  INTEGER;
"""


def apply(conn: sqlite3.Connection) -> None:
    for stmt in _ADD_COLUMNS.split(";"):
        sql = stmt.strip()
        if sql:
            conn.execute(sql)
