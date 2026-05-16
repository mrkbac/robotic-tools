# ruff: noqa: S608
"""V2 schema — normalise wide TEXT FKs to INTEGER surrogates and drop ``content_chunk``.

What this migration does:

- ``content`` gains an ``INTEGER PRIMARY KEY AUTOINCREMENT content_id`` and two
  precomputed columns ``sane_message_start_time`` / ``sane_message_end_time``
  (MIN/MAX over chunks whose own start clears 2000-01-01T00:00:00Z).
- ``schema`` gains an ``INTEGER PRIMARY KEY AUTOINCREMENT schema_pk_id``;
  ``schema_hash`` becomes a UNIQUE TEXT column.
- A new ``topic(topic_id INTEGER PRIMARY KEY, name TEXT UNIQUE)`` table
  collects the distinct channel topic names.
- ``content_channel``, ``content_schema``, ``content_chunk``,
  ``file_observation`` swap their wide ``summary_fingerprint`` /
  ``schema_hash`` FKs for INTEGER surrogates.
- ``content_chunk`` and its ``content_chunk_times`` index are **dropped** —
  no consumer reads the per-chunk byte offsets / compression columns, and
  the MIN/MAX we actually used now lives on ``content``.

The user-facing identifiers (``summary_fingerprint`` text, ``schema_hash``
text, ``topic`` text) are preserved on the dimension tables, so existing
queries that filter or display them still work via a join.

The migration runner (``migrations/__init__.py``) wraps ``apply`` in a single
transaction. We toggle ``PRAGMA defer_foreign_keys = ON`` so the multi-step
table swap doesn't trip FK enforcement; SQLite checks all FKs at COMMIT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "normalise keys; drop content_chunk; precompute sane times"

# 2000-01-01T00:00:00Z in nanoseconds. Anything earlier almost certainly comes
# from an uninitialised clock (e.g. ROS time before NTP sync). Frozen here at
# the value used when this migration was authored; the live constant lives at
# ``pymcap_cli.index.SANE_EPOCH_NS`` but migrations don't import it.
_SANE_EPOCH_NS = 946_684_800 * 1_000_000_000

_NEW_TABLES = """
CREATE TABLE content_v2 (
  content_id              INTEGER PRIMARY KEY AUTOINCREMENT,
  summary_fingerprint     TEXT NOT NULL UNIQUE,
  size_bytes              INTEGER NOT NULL,
  library                 TEXT,
  profile                 TEXT,
  message_count           INTEGER,
  schema_count            INTEGER,
  channel_count           INTEGER,
  attachment_count        INTEGER,
  metadata_count          INTEGER,
  chunk_count             INTEGER,
  message_start_time      INTEGER,
  message_end_time        INTEGER,
  sane_message_start_time INTEGER,
  sane_message_end_time   INTEGER,
  scan_kind               TEXT NOT NULL,
  first_seen_at           INTEGER NOT NULL,
  first_seen_session      INTEGER REFERENCES scan_session(id),
  extra                   TEXT
);

CREATE TABLE topic (
  topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name     TEXT NOT NULL UNIQUE
);

CREATE TABLE schema_v2 (
  schema_pk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  schema_hash  TEXT NOT NULL UNIQUE,
  name         TEXT,
  encoding     TEXT,
  schema_size  INTEGER,
  extra        TEXT
);

CREATE TABLE content_channel_v2 (
  content_id       INTEGER NOT NULL REFERENCES content_v2(content_id),
  channel_id       INTEGER NOT NULL,
  topic_id         INTEGER NOT NULL REFERENCES topic(topic_id),
  schema_id        INTEGER NOT NULL,
  message_encoding TEXT,
  metadata         TEXT,
  message_count    INTEGER,
  extra            TEXT,
  PRIMARY KEY (content_id, channel_id)
);

CREATE TABLE content_schema_v2 (
  content_id   INTEGER NOT NULL REFERENCES content_v2(content_id),
  schema_id    INTEGER NOT NULL,
  schema_pk_id INTEGER NOT NULL REFERENCES schema_v2(schema_pk_id),
  extra        TEXT,
  PRIMARY KEY (content_id, schema_id)
);

CREATE TABLE file_observation_v2 (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  abs_path         TEXT NOT NULL,
  size_bytes       INTEGER NOT NULL,
  mtime_ns         INTEGER NOT NULL,
  inode            INTEGER,
  file_fingerprint TEXT NOT NULL,
  content_id       INTEGER REFERENCES content_v2(content_id),
  is_deleted       INTEGER NOT NULL DEFAULT 0,
  session_id       INTEGER NOT NULL REFERENCES scan_session(id),
  observed_at      INTEGER NOT NULL,
  extra            TEXT
);
"""

_COPY_DATA = f"""
INSERT INTO content_v2 (
  summary_fingerprint, size_bytes, library, profile,
  message_count, schema_count, channel_count,
  attachment_count, metadata_count, chunk_count,
  message_start_time, message_end_time,
  sane_message_start_time, sane_message_end_time,
  scan_kind, first_seen_at, first_seen_session, extra
)
SELECT c.summary_fingerprint, c.size_bytes, c.library, c.profile,
       c.message_count, c.schema_count, c.channel_count,
       c.attachment_count, c.metadata_count, c.chunk_count,
       c.message_start_time, c.message_end_time,
       cs.sane_start, cs.sane_end,
       c.scan_kind, c.first_seen_at, c.first_seen_session, c.extra
FROM content c
LEFT JOIN (
  SELECT summary_fingerprint,
         MIN(message_start_time) AS sane_start,
         MAX(message_end_time)   AS sane_end
  FROM content_chunk
  WHERE message_start_time >= {_SANE_EPOCH_NS}
  GROUP BY summary_fingerprint
) cs ON cs.summary_fingerprint = c.summary_fingerprint;

INSERT INTO topic (name)
SELECT DISTINCT topic FROM content_channel ORDER BY topic;

INSERT INTO schema_v2 (schema_hash, name, encoding, schema_size, extra)
SELECT schema_hash, name, encoding, schema_size, extra FROM schema;

INSERT INTO content_channel_v2 (
  content_id, channel_id, topic_id, schema_id,
  message_encoding, metadata, message_count, extra
)
SELECT c.content_id, cc.channel_id, t.topic_id, cc.schema_id,
       cc.message_encoding, cc.metadata, cc.message_count, cc.extra
FROM content_channel cc
JOIN content_v2 c ON c.summary_fingerprint = cc.summary_fingerprint
JOIN topic     t ON t.name                = cc.topic;

INSERT INTO content_schema_v2 (content_id, schema_id, schema_pk_id, extra)
SELECT c.content_id, cs.schema_id, s.schema_pk_id, cs.extra
FROM content_schema cs
JOIN content_v2 c ON c.summary_fingerprint = cs.summary_fingerprint
JOIN schema_v2  s ON s.schema_hash         = cs.schema_hash;

INSERT INTO file_observation_v2 (
  id, abs_path, size_bytes, mtime_ns, inode,
  file_fingerprint, content_id, is_deleted,
  session_id, observed_at, extra
)
SELECT obs.id, obs.abs_path, obs.size_bytes, obs.mtime_ns, obs.inode,
       obs.file_fingerprint, c.content_id, obs.is_deleted,
       obs.session_id, obs.observed_at, obs.extra
FROM file_observation obs
LEFT JOIN content_v2 c ON c.summary_fingerprint = obs.summary_fingerprint;
"""

_DROP_OLD = """
DROP VIEW  IF EXISTS current_file;
DROP TABLE content_chunk;
DROP TABLE content_schema;
DROP TABLE content_channel;
DROP TABLE file_observation;
DROP TABLE content;
DROP TABLE schema;
"""

# RENAME after the old tables / view are gone. SQLite (>=3.26, 2018) auto-
# rewrites FK references in other tables to point at the renamed name.
_RENAME = """
ALTER TABLE content_v2          RENAME TO content;
ALTER TABLE schema_v2           RENAME TO schema;
ALTER TABLE content_channel_v2  RENAME TO content_channel;
ALTER TABLE content_schema_v2   RENAME TO content_schema;
ALTER TABLE file_observation_v2 RENAME TO file_observation;
"""

_INDEXES_AND_VIEW = """
-- Re-create indexes for every table we rebuilt. ``scan_error`` and
-- ``scan_session`` are untouched, so their indexes from 0001 still apply.
CREATE INDEX file_observation_path        ON file_observation(abs_path);
CREATE INDEX file_observation_content_id  ON file_observation(content_id);
CREATE INDEX file_observation_file_fp     ON file_observation(file_fingerprint);
CREATE INDEX file_observation_session     ON file_observation(session_id);
CREATE INDEX file_observation_stat        ON file_observation(abs_path, size_bytes, mtime_ns);
CREATE INDEX content_channel_topic_id     ON content_channel(topic_id);
CREATE INDEX content_schema_pk_id         ON content_schema(schema_pk_id);
CREATE INDEX schema_name                  ON schema(name);
CREATE INDEX content_times                ON content(message_start_time, message_end_time);

-- Keep ``current_file`` exposing ``summary_fingerprint`` so existing callers
-- that select that column don't have to change. They can also use
-- ``content_id`` directly when joining further child tables.
CREATE VIEW current_file AS
  SELECT obs.id, obs.abs_path, obs.size_bytes, obs.mtime_ns, obs.inode,
         obs.file_fingerprint, obs.content_id,
         c.summary_fingerprint,
         obs.is_deleted, obs.session_id, obs.observed_at, obs.extra
  FROM file_observation AS obs
  LEFT JOIN content c ON c.content_id = obs.content_id
  JOIN (
    SELECT abs_path, MAX(id) AS max_id
    FROM file_observation GROUP BY abs_path
  ) latest ON latest.max_id = obs.id
  WHERE obs.is_deleted = 0;
"""


def _exec_script(conn: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement SQL string while staying in the outer txn.

    ``sqlite3.Connection.executescript`` issues an implicit COMMIT before it
    runs, which would break the migration framework's transaction. So we split
    on ``;`` and execute each non-empty statement ourselves.
    """
    for raw in script.split(";"):
        stmt = raw.strip()
        if stmt:
            conn.execute(stmt)


def apply(conn: sqlite3.Connection) -> None:
    """Rewrite the catalog in place. Caller (migration runner) owns the txn."""
    # Defer FK enforcement until COMMIT so the multi-step table swap works.
    conn.execute("PRAGMA defer_foreign_keys = ON")
    _exec_script(conn, _NEW_TABLES)
    _exec_script(conn, _COPY_DATA)
    _exec_script(conn, _DROP_OLD)
    _exec_script(conn, _RENAME)
    _exec_script(conn, _INDEXES_AND_VIEW)
