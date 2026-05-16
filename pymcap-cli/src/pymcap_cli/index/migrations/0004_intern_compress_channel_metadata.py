"""V4 schema — intern ``channel_sig.metadata`` and zlib-compress the payload.

The ``channel_sig.metadata`` column carries per-channel JSON dumped from the
MCAP Channel record. ROS2 recordings stuff a ``offered_qos_profiles`` YAML
blob in there which is enormous (single blobs of 100s of KB up to multiple
MB) and compresses ~90x with zlib. Even after the ``channel_sig`` dedupe
introduced in 0003, the same metadata blob still lives many times across
recordings, and the wide UNIQUE INDEX over the raw bytes doubles the cost.

This migration:
  - Adds ``channel_metadata(metadata_id INTEGER PK, content_hash TEXT UNIQUE,
    blob_zlib BLOB)`` — one row per *distinct* metadata payload, content-hashed
    so the UNIQUE INDEX is tiny.
  - Rewrites ``channel_sig`` so its old ``metadata TEXT`` column becomes
    ``channel_metadata_id INTEGER REFERENCES channel_metadata(metadata_id)``.
  - Recreates the channel_sig UNIQUE INDEX over the surrogate id instead of
    the raw blob.

The catalog still records the same per-channel metadata; readers that want
the original JSON decompress ``blob_zlib`` and look up by ``metadata_id``.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import xxhash

if TYPE_CHECKING:
    import sqlite3

description = "intern + zlib-compress channel_sig.metadata into channel_metadata dim"


_NEW_TABLES = """
CREATE TABLE channel_metadata (
  metadata_id  INTEGER PRIMARY KEY AUTOINCREMENT,
  content_hash TEXT NOT NULL UNIQUE,
  blob_zlib    BLOB NOT NULL
);

CREATE TABLE channel_sig_v4 (
  channel_sig_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id            INTEGER NOT NULL REFERENCES topic(topic_id),
  schema_pk_id        INTEGER REFERENCES schema(schema_pk_id),
  message_encoding    TEXT,
  channel_metadata_id INTEGER REFERENCES channel_metadata(metadata_id)
);
"""

_INDEXES = """
CREATE UNIQUE INDEX channel_sig_unique ON channel_sig (
  topic_id,
  COALESCE(schema_pk_id, 0),
  COALESCE(message_encoding, ''),
  COALESCE(channel_metadata_id, 0)
);
CREATE INDEX channel_sig_topic_id    ON channel_sig(topic_id);
CREATE INDEX channel_sig_metadata_id ON channel_sig(channel_metadata_id);
"""


def _exec_script(conn: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement SQL string while staying in the outer txn."""
    for raw in script.split(";"):
        stmt = raw.strip()
        if stmt:
            conn.execute(stmt)


def apply(conn: sqlite3.Connection) -> None:
    """Intern + compress ``channel_sig.metadata`` into ``channel_metadata``."""
    conn.execute("PRAGMA defer_foreign_keys = ON")
    _exec_script(conn, _NEW_TABLES)

    # Pre-compute hash + compressed blob for every DISTINCT metadata value.
    # Doing this in Python is fine: total work is bounded by the small number
    # of distinct metadata blobs (typically <1k even on large catalogs), and
    # zlib at level 6 compresses each in microseconds.
    raw_to_id: dict[str, int] = {}
    insert_rows: list[tuple[str, bytes]] = []
    for (raw,) in conn.execute(
        "SELECT DISTINCT metadata FROM channel_sig WHERE metadata IS NOT NULL"
    ):
        encoded = raw.encode("utf-8")
        content_hash = xxhash.xxh3_128_hexdigest(encoded)
        compressed = zlib.compress(encoded, 6)
        insert_rows.append((content_hash, compressed))
        raw_to_id[raw] = -1  # placeholder; filled in below after we have rowids

    conn.executemany(
        "INSERT OR IGNORE INTO channel_metadata(content_hash, blob_zlib) VALUES (?, ?)",
        insert_rows,
    )

    # Re-read so we have the actual metadata_id for each hash. ``INSERT OR
    # IGNORE`` doesn't return a usable ``lastrowid`` for ignored rows.
    hash_to_id: dict[str, int] = {
        h: mid for mid, h in conn.execute("SELECT metadata_id, content_hash FROM channel_metadata")
    }
    for raw in list(raw_to_id):
        h = xxhash.xxh3_128_hexdigest(raw.encode("utf-8"))
        raw_to_id[raw] = hash_to_id[h]

    # Stream old rows into the new shape, resolving metadata_id per row.
    src = conn.execute(
        "SELECT channel_sig_id, topic_id, schema_pk_id, message_encoding, metadata "
        "FROM channel_sig"
    )
    rebuilt: list[tuple[int, int, int | None, str | None, int | None]] = []
    for sig_id, topic_id, schema_pk_id, encoding, meta in src:
        meta_id = raw_to_id[meta] if meta is not None else None
        rebuilt.append((sig_id, topic_id, schema_pk_id, encoding, meta_id))
    conn.executemany(
        "INSERT INTO channel_sig_v4 "
        "(channel_sig_id, topic_id, schema_pk_id, message_encoding, channel_metadata_id) "
        "VALUES (?, ?, ?, ?, ?)",
        rebuilt,
    )

    # Swap. The indexes on the old ``channel_sig`` (channel_sig_unique,
    # channel_sig_topic_id) drop with the table.
    conn.execute("DROP TABLE channel_sig")
    conn.execute("ALTER TABLE channel_sig_v4 RENAME TO channel_sig")
    _exec_script(conn, _INDEXES)
