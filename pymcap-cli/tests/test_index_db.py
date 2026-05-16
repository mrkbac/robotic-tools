"""Tests for the sidecar index DB schema and migrations."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.index import migrations
from pymcap_cli.index.db import (
    CURRENT_SCHEMA_VERSION,
    IndexDbNeedsMigrationError,
    connect,
    default_db_path,
    finish_session,
    open_db,
    start_session,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_connect_creates_db_and_applies_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite"
    with open_db(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == CURRENT_SCHEMA_VERSION

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {
            "schema_migrations",
            "scan_session",
            "content",
            "content_channel",
            "schema",
            "content_schema",
            "topic",
            "channel_sig",
            "channel_metadata",
            "file_observation",
            "scan_error",
        }.issubset(tables)

        views = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
        }
        assert "current_file" in views

        applied = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        assert applied == [(version,) for version in range(1, CURRENT_SCHEMA_VERSION + 1)]


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite"
    with open_db(db_path):
        pass
    with open_db(db_path) as conn:
        migrations = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        assert migrations == CURRENT_SCHEMA_VERSION


def test_session_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite"
    with open_db(db_path) as conn:
        session_id = start_session(conn, tmp_path, "1.2.3")
        assert session_id >= 1
        row = conn.execute(
            "SELECT root_path, pymcap_cli_version, finished_at FROM scan_session WHERE id=?",
            (session_id,),
        ).fetchone()
        assert row[0] == str(tmp_path)
        assert row[1] == "1.2.3"
        assert row[2] is None

        finish_session(conn, session_id)
        finished_at = conn.execute(
            "SELECT finished_at FROM scan_session WHERE id=?", (session_id,)
        ).fetchone()[0]
        assert finished_at is not None


def test_read_only_open_requires_existing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.sqlite"
    try:
        connect(db_path, read_only=True)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_default_db_path_under_cache_dir() -> None:
    path = default_db_path()
    assert path.name == "index.sqlite"
    assert "pymcap-cli" in path.parts


def test_read_only_open_rejects_pending_migrations_without_mutating(tmp_path: Path) -> None:
    """Read commands should not mutate an older DB just to apply migrations."""
    db_path = tmp_path / "index.sqlite"

    raw = sqlite3.connect(db_path, isolation_level=None)
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        v1_version, v1_apply, v1_desc = sorted(migrations._discover())[0]
        assert v1_version == 1, "expected migration 0001 to be the first"
        raw.execute("BEGIN")
        v1_apply(raw)
        raw.execute(
            "INSERT INTO schema_migrations(version, applied_at, description) VALUES (?, 0, ?)",
            (v1_version, v1_desc),
        )
        raw.execute(f"PRAGMA user_version = {v1_version}")
        raw.execute("COMMIT")
    finally:
        raw.close()

    # Verify the on-disk DB is at v1 before the read-only open.
    probe = sqlite3.connect(db_path)
    try:
        assert probe.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        probe.close()

    with pytest.raises(IndexDbNeedsMigrationError) as excinfo:
        connect(db_path, read_only=True)

    assert excinfo.value.current_version == 1
    assert excinfo.value.expected_version == CURRENT_SCHEMA_VERSION

    # Still v1 after the failed read-only open.
    probe = sqlite3.connect(db_path)
    try:
        assert probe.execute("PRAGMA user_version").fetchone()[0] == 1
        applied = probe.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        assert applied == [(1,)]
    finally:
        probe.close()


def test_failed_migration_rolls_back_schema_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.sqlite"
    conn = sqlite3.connect(db_path, isolation_level=None)

    def _fail_after_schema(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE should_rollback(id INTEGER)")
        raise RuntimeError("boom")

    monkeypatch.setattr(migrations, "_discover", lambda: [(1, _fail_after_schema, "fail")])

    try:
        with pytest.raises(RuntimeError, match="boom"):
            migrations.apply_pending(conn)

        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='should_rollback'"
        ).fetchone()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert table is None
        assert version == 0
    finally:
        conn.close()
