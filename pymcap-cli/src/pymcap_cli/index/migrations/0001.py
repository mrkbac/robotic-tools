"""V1 schema — sidecar catalog tables, indexes, current_file view."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "initial schema"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version     INTEGER PRIMARY KEY,
  applied_at  INTEGER NOT NULL,
  description TEXT
);

CREATE TABLE IF NOT EXISTS scan_session (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at         INTEGER NOT NULL,
  finished_at        INTEGER,
  root_path          TEXT NOT NULL,
  pymcap_cli_version TEXT NOT NULL,
  extra              TEXT
);

CREATE TABLE IF NOT EXISTS content (
  summary_fingerprint TEXT PRIMARY KEY,
  size_bytes          INTEGER NOT NULL,
  library             TEXT,
  profile             TEXT,
  message_count       INTEGER,
  schema_count        INTEGER,
  channel_count       INTEGER,
  attachment_count    INTEGER,
  metadata_count      INTEGER,
  chunk_count         INTEGER,
  message_start_time  INTEGER,
  message_end_time    INTEGER,
  scan_kind           TEXT NOT NULL,
  first_seen_at       INTEGER NOT NULL,
  first_seen_session  INTEGER REFERENCES scan_session(id),
  extra               TEXT
);

CREATE TABLE IF NOT EXISTS content_channel (
  summary_fingerprint TEXT NOT NULL REFERENCES content(summary_fingerprint),
  channel_id          INTEGER NOT NULL,
  topic               TEXT NOT NULL,
  schema_id           INTEGER NOT NULL,
  message_encoding    TEXT,
  metadata            TEXT,
  message_count       INTEGER,
  extra               TEXT,
  PRIMARY KEY (summary_fingerprint, channel_id)
);

CREATE TABLE IF NOT EXISTS schema (
  schema_hash TEXT PRIMARY KEY,
  name        TEXT,
  encoding    TEXT,
  schema_size INTEGER,
  extra       TEXT
);

CREATE TABLE IF NOT EXISTS content_schema (
  summary_fingerprint TEXT NOT NULL REFERENCES content(summary_fingerprint),
  schema_id           INTEGER NOT NULL,
  schema_hash         TEXT NOT NULL REFERENCES schema(schema_hash),
  extra               TEXT,
  PRIMARY KEY (summary_fingerprint, schema_id)
);

CREATE TABLE IF NOT EXISTS content_chunk (
  summary_fingerprint TEXT NOT NULL REFERENCES content(summary_fingerprint),
  chunk_idx           INTEGER NOT NULL,
  message_start_time  INTEGER,
  message_end_time    INTEGER,
  chunk_start_offset  INTEGER,
  chunk_length        INTEGER,
  compression         TEXT,
  compressed_size     INTEGER,
  uncompressed_size   INTEGER,
  extra               TEXT,
  PRIMARY KEY (summary_fingerprint, chunk_idx)
);

CREATE TABLE IF NOT EXISTS file_observation (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  abs_path            TEXT NOT NULL,
  size_bytes          INTEGER NOT NULL,
  mtime_ns            INTEGER NOT NULL,
  inode               INTEGER,
  file_fingerprint    TEXT NOT NULL,
  summary_fingerprint TEXT REFERENCES content(summary_fingerprint),
  is_deleted          INTEGER NOT NULL DEFAULT 0,
  session_id          INTEGER NOT NULL REFERENCES scan_session(id),
  observed_at         INTEGER NOT NULL,
  extra               TEXT
);

CREATE TABLE IF NOT EXISTS scan_error (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  abs_path      TEXT NOT NULL,
  size_bytes    INTEGER,
  mtime_ns      INTEGER,
  session_id    INTEGER NOT NULL REFERENCES scan_session(id),
  observed_at   INTEGER NOT NULL,
  error_kind    TEXT NOT NULL,
  error_message TEXT,
  extra         TEXT
);

CREATE INDEX IF NOT EXISTS file_observation_path
  ON file_observation(abs_path);
CREATE INDEX IF NOT EXISTS file_observation_summary_fp
  ON file_observation(summary_fingerprint);
CREATE INDEX IF NOT EXISTS file_observation_file_fp
  ON file_observation(file_fingerprint);
CREATE INDEX IF NOT EXISTS file_observation_session
  ON file_observation(session_id);
CREATE INDEX IF NOT EXISTS file_observation_stat
  ON file_observation(abs_path, size_bytes, mtime_ns);
CREATE INDEX IF NOT EXISTS content_channel_topic
  ON content_channel(topic);
CREATE INDEX IF NOT EXISTS content_schema_hash
  ON content_schema(schema_hash);
CREATE INDEX IF NOT EXISTS schema_name
  ON schema(name);
CREATE INDEX IF NOT EXISTS content_times
  ON content(message_start_time, message_end_time);
CREATE INDEX IF NOT EXISTS content_chunk_times
  ON content_chunk(message_start_time, message_end_time);
CREATE INDEX IF NOT EXISTS scan_error_path
  ON scan_error(abs_path);
CREATE INDEX IF NOT EXISTS scan_error_path_stat
  ON scan_error(abs_path, size_bytes, mtime_ns);

CREATE VIEW IF NOT EXISTS current_file AS
  SELECT obs.* FROM file_observation AS obs
  JOIN (
    SELECT abs_path, MAX(id) AS max_id
    FROM file_observation GROUP BY abs_path
  ) latest ON latest.max_id = obs.id
  WHERE obs.is_deleted = 0;
"""


def apply(conn: sqlite3.Connection) -> None:
    """Apply the V1 schema to a fresh DB."""
    for statement in _SCHEMA.split(";"):
        sql = statement.strip()
        if sql:
            conn.execute(sql)
