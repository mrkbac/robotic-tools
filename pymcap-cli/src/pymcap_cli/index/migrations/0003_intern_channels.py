"""V3 schema — intern repeating channel attributes into a ``channel_sig`` dimension.

On the corpus this was built against, 937k ``content_channel`` rows collapse to
~13k distinct ``(topic_id, schema_pk_id, message_encoding, metadata)`` tuples —
a ~70x repetition factor, dominated by the per-channel ``metadata`` JSON
(>200 byte blobs duplicated across files). This migration:

- Adds a ``channel_sig(channel_sig_id, topic_id, schema_pk_id,
  message_encoding, metadata)`` dimension table.
- Rebuilds ``content_channel`` as ``(content_id, channel_id, channel_sig_id,
  message_count)`` — the file-local ``schema_id`` and the wide
  ``message_encoding`` / ``metadata`` columns are gone from the fact table.
- Uses a UNIQUE INDEX with ``COALESCE(...)`` so ``NULL`` values dedupe (a
  plain UNIQUE constraint would treat each NULL as distinct).

User-facing output is unchanged: the catalog still records the same
per-channel topic, schema, encoding, metadata, and message_count for every
indexed file; they're just stored via a join instead of inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "intern channel attributes into channel_sig dimension"

_NEW_TABLES = """
CREATE TABLE channel_sig (
  channel_sig_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id         INTEGER NOT NULL REFERENCES topic(topic_id),
  schema_pk_id     INTEGER REFERENCES schema(schema_pk_id),
  message_encoding TEXT,
  metadata         TEXT
);

-- COALESCE-based UNIQUE so two NULLs collide as a duplicate (SQLite would
-- otherwise treat NULL != NULL and let identical signatures pile up).
CREATE UNIQUE INDEX channel_sig_unique ON channel_sig (
  topic_id,
  COALESCE(schema_pk_id, 0),
  COALESCE(message_encoding, ''),
  COALESCE(metadata, '')
);

CREATE TABLE content_channel_v3 (
  content_id     INTEGER NOT NULL REFERENCES content(content_id),
  channel_id     INTEGER NOT NULL,
  channel_sig_id INTEGER NOT NULL REFERENCES channel_sig(channel_sig_id),
  message_count  INTEGER,
  PRIMARY KEY (content_id, channel_id)
);
"""

_POPULATE = """
INSERT INTO channel_sig (topic_id, schema_pk_id, message_encoding, metadata)
SELECT DISTINCT
       cc.topic_id,
       cs.schema_pk_id,
       cc.message_encoding,
       cc.metadata
FROM content_channel cc
LEFT JOIN content_schema cs
  ON cs.content_id = cc.content_id
 AND cs.schema_id  = cc.schema_id;

INSERT INTO content_channel_v3 (content_id, channel_id, channel_sig_id, message_count)
SELECT cc.content_id, cc.channel_id, sig.channel_sig_id, cc.message_count
FROM content_channel cc
LEFT JOIN content_schema cs
  ON cs.content_id = cc.content_id
 AND cs.schema_id  = cc.schema_id
JOIN channel_sig sig
  ON sig.topic_id = cc.topic_id
 AND COALESCE(sig.schema_pk_id, 0)     = COALESCE(cs.schema_pk_id, 0)
 AND COALESCE(sig.message_encoding,'') = COALESCE(cc.message_encoding,'')
 AND COALESCE(sig.metadata, '')        = COALESCE(cc.metadata, '');
"""

_SWAP = """
DROP TABLE content_channel;
ALTER TABLE content_channel_v3 RENAME TO content_channel;
CREATE INDEX content_channel_sig_id ON content_channel(channel_sig_id);
CREATE INDEX channel_sig_topic_id   ON channel_sig(topic_id);
"""


def _exec_script(conn: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement SQL string while staying in the outer txn.

    ``sqlite3.Connection.executescript`` implicitly COMMITs first, which would
    break the migration framework's surrounding transaction.
    """
    for raw in script.split(";"):
        stmt = raw.strip()
        if stmt:
            conn.execute(stmt)


def apply(conn: sqlite3.Connection) -> None:
    """Rewrite ``content_channel`` via the new ``channel_sig`` dimension."""
    # ``content_channel_topic_id`` (built in 0002) goes away with its table.
    conn.execute("PRAGMA defer_foreign_keys = ON")
    _exec_script(conn, _NEW_TABLES)
    _exec_script(conn, _POPULATE)
    _exec_script(conn, _SWAP)
