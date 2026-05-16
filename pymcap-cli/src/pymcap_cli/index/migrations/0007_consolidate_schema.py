"""V7 schema — dedup repeated strings, drop dead columns, normalise naming, STRICT.

One-shot consolidation pass. Bundled cleanups:

1. **Dedup** ``content.library`` / ``content.profile`` / every ``abs_path`` and
   ``root_path`` into ``library`` / ``profile`` / ``file_path`` dim tables.
2. **Drop** the unused ``extra TEXT`` column on every table that had it
   (``content``, ``file_observation``, ``scan_error``, ``content_schema``,
   ``schema``, ``scan_session``) — 0 rows populated across the entire corpus.
3. **Rename** for consistency:
   - PKs become ``id``; FKs become ``<referenced_table>_id``.
   - Every nanosecond timestamp ends in ``_ns``.
   - Disambiguate MCAP-local ids (``content_channel.channel_id`` →
     ``mcap_channel_id``; ``content_schema.schema_id`` → ``mcap_schema_id``).
   - Expand abbreviations (``channel_sig`` → ``channel_signature``).
   - Drop type-in-name (``blob_zlib`` → ``metadata_json_zlib``).
   - Normalise ``schema.schema_size`` → ``schema.size_bytes``.
4. **STRICT** on every user table. SQLite enforces declared column types at
   write time, catching coercion bugs at the SQL layer.

SQLite cannot retrofit STRICT onto an existing table, and the rename / FK
changes need a full rebuild anyway, so we use the same rebuild-copy-swap
pattern as migration 0002. Steps:

1. Create the dim tables (``library``, ``profile``, ``file_path``).
2. Create every fact table in its final shape as ``<table>_v7``.
3. Populate dims from distinct values (NULL-filtered for library / profile).
4. ``INSERT INTO <table>_v7 SELECT … FROM <table>`` through the joins,
   resolving FKs as rows flow.
5. Drop the old view and tables (in FK order).
6. ``ALTER TABLE … RENAME TO`` each ``_v7`` to its final name.
7. Recreate indexes.
8. Recreate the ``current_file`` view; alias ``file_path.value AS abs_path``
   so existing read-side queries that select ``current_file.abs_path`` keep
   working without source changes.
9. ``PRAGMA foreign_key_check`` to verify referential integrity before
   handing control back to the migration framework (which then VACUUMs).

The earlier short-lived v7 (``channel_distribution``) is folded into this
migration: the new ``content_channel`` carries ``distribution_blob BLOB``
from the rebuild, NULL on existing rows until ``index scan --force-rebuild``
populates them.

``schema_migrations.applied_at`` is *not* renamed to ``applied_at_ns`` — it's
framework metadata, not catalog data, and renaming it would couple the
migration runner to this specific migration's naming pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "intern strings into dims; rename to <table>.id / <ref>_id / *_ns; STRICT"


_CREATE_DIMS = """
CREATE TABLE library   (id INTEGER PRIMARY KEY AUTOINCREMENT, name  TEXT NOT NULL UNIQUE) STRICT;
CREATE TABLE profile   (id INTEGER PRIMARY KEY AUTOINCREMENT, name  TEXT NOT NULL UNIQUE) STRICT;
CREATE TABLE file_path (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT NOT NULL UNIQUE) STRICT;
"""


_CREATE_FACT_TABLES_V7 = """
CREATE TABLE scan_session_v7 (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at_ns      INTEGER NOT NULL,
  finished_at_ns     INTEGER,
  root_file_path_id  INTEGER NOT NULL REFERENCES file_path(id),
  pymcap_cli_version TEXT NOT NULL
) STRICT;

CREATE TABLE content_v7 (
  id                         INTEGER PRIMARY KEY AUTOINCREMENT,
  summary_fingerprint        TEXT    NOT NULL UNIQUE,
  size_bytes                 INTEGER NOT NULL,
  library_id                 INTEGER REFERENCES library(id),
  profile_id                 INTEGER REFERENCES profile(id),
  message_count              INTEGER NOT NULL,
  schema_count               INTEGER NOT NULL,
  channel_count              INTEGER NOT NULL,
  attachment_count           INTEGER NOT NULL,
  metadata_count             INTEGER NOT NULL,
  chunk_count                INTEGER NOT NULL,
  message_start_time_ns      INTEGER NOT NULL,
  message_end_time_ns        INTEGER NOT NULL,
  sane_message_start_time_ns INTEGER,
  sane_message_end_time_ns   INTEGER,
  scan_kind                  TEXT    NOT NULL,
  first_seen_at_ns           INTEGER NOT NULL,
  first_seen_scan_session_id INTEGER NOT NULL REFERENCES scan_session_v7(id),
  compression                TEXT,
  compressed_size_bytes      INTEGER,
  uncompressed_size_bytes    INTEGER
) STRICT;

CREATE TABLE topic_v7 (
  id   INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE
) STRICT;

CREATE TABLE schema_v7 (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  schema_hash TEXT    NOT NULL UNIQUE,
  name        TEXT    NOT NULL,
  encoding    TEXT    NOT NULL,
  size_bytes  INTEGER NOT NULL
) STRICT;

CREATE TABLE channel_metadata_v7 (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  content_hash        TEXT NOT NULL UNIQUE,
  metadata_json_zlib  BLOB NOT NULL
) STRICT;

CREATE TABLE channel_signature_v7 (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id            INTEGER NOT NULL REFERENCES topic_v7(id),
  schema_id           INTEGER REFERENCES schema_v7(id),
  message_encoding    TEXT,
  channel_metadata_id INTEGER REFERENCES channel_metadata_v7(id)
) STRICT;

CREATE TABLE content_channel_v7 (
  content_id              INTEGER NOT NULL REFERENCES content_v7(id),
  mcap_channel_id         INTEGER NOT NULL,
  channel_signature_id    INTEGER NOT NULL REFERENCES channel_signature_v7(id),
  message_count           INTEGER,
  uncompressed_size_bytes INTEGER,
  message_start_time_ns   INTEGER,
  message_end_time_ns     INTEGER,
  distribution_blob       BLOB,
  PRIMARY KEY (content_id, mcap_channel_id)
) STRICT;

CREATE TABLE content_schema_v7 (
  content_id     INTEGER NOT NULL REFERENCES content_v7(id),
  mcap_schema_id INTEGER NOT NULL,
  schema_id      INTEGER NOT NULL REFERENCES schema_v7(id),
  PRIMARY KEY (content_id, mcap_schema_id)
) STRICT;

CREATE TABLE file_observation_v7 (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path_id     INTEGER NOT NULL REFERENCES file_path(id),
  size_bytes       INTEGER NOT NULL,
  mtime_ns         INTEGER NOT NULL,
  inode            INTEGER,
  file_fingerprint TEXT NOT NULL,
  content_id       INTEGER REFERENCES content_v7(id),
  is_deleted       INTEGER NOT NULL DEFAULT 0,
  scan_session_id  INTEGER NOT NULL REFERENCES scan_session_v7(id),
  observed_at_ns   INTEGER NOT NULL
) STRICT;

CREATE TABLE scan_error_v7 (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path_id    INTEGER NOT NULL REFERENCES file_path(id),
  size_bytes      INTEGER NOT NULL,
  mtime_ns        INTEGER NOT NULL,
  scan_session_id INTEGER NOT NULL REFERENCES scan_session_v7(id),
  observed_at_ns  INTEGER NOT NULL,
  error_kind      TEXT    NOT NULL,
  error_message   TEXT    NOT NULL
) STRICT;
"""


_POPULATE_DIMS = """
INSERT INTO library(name)
  SELECT DISTINCT library FROM content WHERE library IS NOT NULL;

INSERT INTO profile(name)
  SELECT DISTINCT profile FROM content WHERE profile IS NOT NULL;

INSERT INTO file_path(value)
  SELECT abs_path  FROM file_observation
  UNION
  SELECT abs_path  FROM scan_error
  UNION
  SELECT root_path FROM scan_session;
"""


_COPY_FACTS = """
INSERT INTO scan_session_v7
  (id, started_at_ns, finished_at_ns, root_file_path_id, pymcap_cli_version)
SELECT s.id, s.started_at, s.finished_at, fp.id, s.pymcap_cli_version
  FROM scan_session s
  JOIN file_path fp ON fp.value = s.root_path;

INSERT INTO content_v7
  (id, summary_fingerprint, size_bytes, library_id, profile_id,
   message_count, schema_count, channel_count,
   attachment_count, metadata_count, chunk_count,
   message_start_time_ns, message_end_time_ns,
   sane_message_start_time_ns, sane_message_end_time_ns,
   scan_kind, first_seen_at_ns, first_seen_scan_session_id,
   compression, compressed_size_bytes, uncompressed_size_bytes)
SELECT c.content_id, c.summary_fingerprint, c.size_bytes,
       lib.id, prof.id,
       c.message_count, c.schema_count, c.channel_count,
       c.attachment_count, c.metadata_count, c.chunk_count,
       c.message_start_time, c.message_end_time,
       c.sane_message_start_time, c.sane_message_end_time,
       c.scan_kind, c.first_seen_at, c.first_seen_session,
       c.compression, c.compressed_size_bytes, c.uncompressed_size_bytes
  FROM content c
  LEFT JOIN library lib ON lib.name = c.library
  LEFT JOIN profile prof ON prof.name = c.profile;

INSERT INTO topic_v7(id, name)
  SELECT topic_id, name FROM topic;

INSERT INTO schema_v7(id, schema_hash, name, encoding, size_bytes)
  SELECT schema_pk_id, schema_hash, name, encoding, schema_size FROM schema;

INSERT INTO channel_metadata_v7(id, content_hash, metadata_json_zlib)
  SELECT metadata_id, content_hash, blob_zlib FROM channel_metadata;

INSERT INTO channel_signature_v7
  (id, topic_id, schema_id, message_encoding, channel_metadata_id)
SELECT channel_sig_id, topic_id, schema_pk_id, message_encoding, channel_metadata_id
  FROM channel_sig;

INSERT INTO content_channel_v7
  (content_id, mcap_channel_id, channel_signature_id,
   message_count, uncompressed_size_bytes,
   message_start_time_ns, message_end_time_ns, distribution_blob)
SELECT content_id, channel_id, channel_sig_id,
       message_count, uncompressed_size_bytes,
       message_start_time, message_end_time, distribution_blob
  FROM content_channel;

INSERT INTO content_schema_v7(content_id, mcap_schema_id, schema_id)
  SELECT content_id, schema_id, schema_pk_id FROM content_schema;

INSERT INTO file_observation_v7
  (id, file_path_id, size_bytes, mtime_ns, inode,
   file_fingerprint, content_id, is_deleted,
   scan_session_id, observed_at_ns)
SELECT fo.id, fp.id, fo.size_bytes, fo.mtime_ns, fo.inode,
       fo.file_fingerprint, fo.content_id, fo.is_deleted,
       fo.session_id, fo.observed_at
  FROM file_observation fo
  JOIN file_path fp ON fp.value = fo.abs_path;

INSERT INTO scan_error_v7
  (id, file_path_id, size_bytes, mtime_ns,
   scan_session_id, observed_at_ns, error_kind, error_message)
SELECT se.id, fp.id, se.size_bytes, se.mtime_ns,
       se.session_id, se.observed_at, se.error_kind, se.error_message
  FROM scan_error se
  JOIN file_path fp ON fp.value = se.abs_path;
"""


_DROP_OLD = """
DROP VIEW  IF EXISTS current_file;
DROP TABLE file_observation;
DROP TABLE scan_error;
DROP TABLE content_channel;
DROP TABLE content_schema;
DROP TABLE channel_sig;
DROP TABLE channel_metadata;
DROP TABLE content;
DROP TABLE schema;
DROP TABLE topic;
DROP TABLE scan_session;
"""


_RENAME = """
ALTER TABLE scan_session_v7      RENAME TO scan_session;
ALTER TABLE content_v7           RENAME TO content;
ALTER TABLE topic_v7             RENAME TO topic;
ALTER TABLE schema_v7            RENAME TO schema;
ALTER TABLE channel_metadata_v7  RENAME TO channel_metadata;
ALTER TABLE channel_signature_v7 RENAME TO channel_signature;
ALTER TABLE content_channel_v7   RENAME TO content_channel;
ALTER TABLE content_schema_v7    RENAME TO content_schema;
ALTER TABLE file_observation_v7  RENAME TO file_observation;
ALTER TABLE scan_error_v7        RENAME TO scan_error;
"""


_INDEXES_AND_VIEW = """
CREATE INDEX file_observation_file_path_id     ON file_observation(file_path_id);
CREATE INDEX file_observation_content_id       ON file_observation(content_id);
CREATE INDEX file_observation_file_fingerprint ON file_observation(file_fingerprint);
CREATE INDEX file_observation_scan_session_id  ON file_observation(scan_session_id);

CREATE INDEX scan_error_file_path_id_stat
  ON scan_error(file_path_id, size_bytes, mtime_ns);

CREATE INDEX content_message_time_ns
  ON content(message_start_time_ns, message_end_time_ns);

CREATE INDEX content_channel_channel_signature_id ON content_channel(channel_signature_id);
CREATE INDEX content_schema_schema_id              ON content_schema(schema_id);
CREATE INDEX schema_name                           ON schema(name);

CREATE UNIQUE INDEX channel_signature_unique ON channel_signature (
  topic_id,
  COALESCE(schema_id, 0),
  COALESCE(message_encoding, ''),
  COALESCE(channel_metadata_id, 0)
);
CREATE INDEX channel_signature_topic_id            ON channel_signature(topic_id);
CREATE INDEX channel_signature_channel_metadata_id ON channel_signature(channel_metadata_id);

CREATE VIEW current_file AS
  SELECT obs.id, fp.value AS abs_path, obs.size_bytes, obs.mtime_ns, obs.inode,
         obs.file_fingerprint, obs.content_id, c.summary_fingerprint,
         obs.is_deleted, obs.scan_session_id, obs.observed_at_ns
    FROM file_observation AS obs
    JOIN file_path fp ON fp.id = obs.file_path_id
    LEFT JOIN content c ON c.id = obs.content_id
    JOIN (
      SELECT file_path_id, MAX(id) AS max_id
      FROM file_observation GROUP BY file_path_id
    ) latest ON latest.max_id = obs.id
   WHERE obs.is_deleted = 0;
"""


def _exec_script(conn: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement SQL string while staying in the outer txn.

    ``sqlite3.Connection.executescript`` issues an implicit COMMIT before it
    runs, which would break the migration framework's transaction. Split on
    ``;`` and execute each non-empty statement ourselves; strip ``--`` line
    comments first so a ``;`` inside a comment doesn't fragment a statement.
    """
    lines = [line.split("--", 1)[0] for line in script.splitlines()]
    cleaned = "\n".join(lines)
    for raw in cleaned.split(";"):
        stmt = raw.strip()
        if stmt:
            conn.execute(stmt)


def apply(conn: sqlite3.Connection) -> None:
    """Rewrite the catalog in place. Caller (migration runner) owns the txn."""
    conn.execute("PRAGMA defer_foreign_keys = ON")
    _exec_script(conn, _CREATE_DIMS)
    _exec_script(conn, _CREATE_FACT_TABLES_V7)
    _exec_script(conn, _POPULATE_DIMS)
    _exec_script(conn, _COPY_FACTS)
    _exec_script(conn, _DROP_OLD)
    _exec_script(conn, _RENAME)
    _exec_script(conn, _INDEXES_AND_VIEW)
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"foreign-key violations after consolidation: {violations[:5]}")
