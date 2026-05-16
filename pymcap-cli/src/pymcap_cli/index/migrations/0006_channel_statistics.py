"""V6 schema — per-channel statistics on ``content_channel``.

Brings ``pymcap-cli index info`` to parity with the standalone ``info`` for
per-channel breakdowns (bytes, duration, Hz, distribution). All four columns
are nullable; existing rows keep ``NULL`` until they're refreshed via
``index scan --force-rebuild``. Readers treat ``NULL`` as "unknown".

The distribution blob is a zlib-compressed ``uint16`` bucket count followed by
that many ``uint32`` bin counts. The bucket count is selected by the same
logic used by the standalone ``info`` command, so cached and live channel
tables render the same distribution shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "per-channel size, time range, and distribution on content_channel"

_ADD_COLUMNS = """
ALTER TABLE content_channel ADD COLUMN uncompressed_size_bytes INTEGER;
ALTER TABLE content_channel ADD COLUMN message_start_time      INTEGER;
ALTER TABLE content_channel ADD COLUMN message_end_time        INTEGER;
ALTER TABLE content_channel ADD COLUMN distribution_blob       BLOB;
"""


def apply(conn: sqlite3.Connection) -> None:
    for stmt in _ADD_COLUMNS.split(";"):
        sql = stmt.strip()
        if sql:
            conn.execute(sql)
