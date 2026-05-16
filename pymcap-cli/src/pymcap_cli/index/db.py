"""Sidecar index database — connection factory + session helpers.

Schema lives in :mod:`pymcap_cli.index.migrations` (one numbered file per
version). The schema is append-only — no row is ever updated except for
``scan_session.finished_at`` and ``file_observation.is_deleted`` tombstones.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_cache_dir

from pymcap_cli.index.migrations import _discover, apply_pending

if TYPE_CHECKING:
    from collections.abc import Iterator

CURRENT_SCHEMA_VERSION = max((v for v, _, _ in _discover()), default=0)


def default_db_path() -> Path:
    """Return the default sidecar DB path under the user cache directory."""
    return Path(user_cache_dir("pymcap-cli")) / "index.sqlite"


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the sidecar DB, creating parent directories and applying migrations.

    Returns a connection with WAL mode and foreign keys enabled. Caller closes.
    """
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        apply_pending(conn)
        return conn

    if not db_path.exists():
        raise FileNotFoundError(db_path)
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    # Read-side tuning. SQLite's defaults (2 MiB page cache, no mmap) are
    # tiny relative to a sidecar catalog that is typically 100–500 MiB and
    # gets aggregated repeatedly. ``query_only`` guards against accidental
    # writer calls on a connection meant for reads.
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def open_db(db_path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    """Context manager wrapping :func:`connect`."""
    conn = connect(db_path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def start_session(conn: sqlite3.Connection, root_path: Path, pymcap_cli_version: str) -> int:
    """Insert a scan_session row and return its id."""
    cur = conn.execute(
        "INSERT INTO scan_session(started_at, root_path, pymcap_cli_version) VALUES (?, ?, ?)",
        (time.time_ns(), str(root_path), pymcap_cli_version),
    )
    session_id = cur.lastrowid
    assert session_id is not None
    return session_id


def finish_session(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute(
        "UPDATE scan_session SET finished_at = ? WHERE id = ?",
        (time.time_ns(), session_id),
    )
