"""V8 schema — make ``current_file`` filter-friendly."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "rewrite current_file view and add read-side indexes for large catalogs"

_CURRENT_FILE_VIEW = """
CREATE VIEW current_file AS
  SELECT obs.id, fp.value AS abs_path, obs.size_bytes, obs.mtime_ns, obs.inode,
         obs.file_fingerprint, obs.content_id, c.summary_fingerprint,
         obs.is_deleted, obs.scan_session_id, obs.observed_at_ns
    FROM file_path fp
    JOIN file_observation obs
      ON obs.id = (
        SELECT MAX(inner_obs.id)
          FROM file_observation inner_obs
         WHERE inner_obs.file_path_id = fp.id
      )
    LEFT JOIN content c ON c.id = obs.content_id
   WHERE obs.is_deleted = 0;
"""

_CONTENT_CURRENT_FILE_COUNT = """
CREATE TABLE IF NOT EXISTS content_current_file_count (
  content_id INTEGER PRIMARY KEY REFERENCES content(id),
  file_count INTEGER NOT NULL CHECK(file_count > 0)
) STRICT;
"""


def apply(conn: sqlite3.Connection) -> None:
    """Recreate only the read-side view; v7 tables and indexes are unchanged."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS file_observation_content_observed_id "
        "ON file_observation(content_id, observed_at_ns DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS content_first_seen_scan_session_id "
        "ON content(first_seen_scan_session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS scan_error_scan_session_id ON scan_error(scan_session_id)"
    )
    conn.execute("DROP VIEW IF EXISTS current_file")
    conn.execute(_CURRENT_FILE_VIEW)
    conn.execute(_CONTENT_CURRENT_FILE_COUNT)
    conn.execute("DELETE FROM content_current_file_count")
    conn.execute(
        "INSERT INTO content_current_file_count(content_id, file_count) "
        "SELECT content_id, COUNT(*) "
        "FROM current_file "
        "WHERE content_id IS NOT NULL "
        "GROUP BY content_id"
    )
