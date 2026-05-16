"""Integration tests for the sidecar index scanner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.index import scanner as index_scanner
from pymcap_cli.index.db import open_db
from pymcap_cli.index.scanner import scan
from small_mcap import rebuild_summary

from tests.fixtures.mcap_generator import create_multi_topic_mcap, create_simple_mcap

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path


def _make_mcap_tree(root: Path, *, count: int = 3) -> list[Path]:
    paths: list[Path] = []
    for i in range(count):
        sub = root / f"sub{i}"
        sub.mkdir()
        p = sub / f"rec_{i}.mcap"
        p.write_bytes(create_simple_mcap(num_messages=10 + i))
        paths.append(p)
    return paths


def _count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608


def _truncate_after_data_section(path: Path) -> None:
    with path.open("r+b") as stream:
        info = rebuild_summary(
            stream,
            validate_crc=False,
            calculate_channel_sizes=False,
            exact_sizes=False,
        )
        stream.truncate(info.next_offset)


def test_first_scan_indexes_all(tmp_path: Path) -> None:
    files = _make_mcap_tree(tmp_path, count=3)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.discovered == 3
        assert stats.indexed == 3
        assert stats.stat_skipped == 0
        assert stats.errored == 0

        assert _count(conn, "content") == 3
        assert _count(conn, "file_observation") == 3
        assert _count(conn, "content_channel") >= len(files)
        assert _count(conn, "content_schema") >= 1
        # ``content.chunk_count`` is populated from the MCAP Statistics
        # record. Per-chunk rows no longer live in their own table.
        chunk_totals = conn.execute(
            "SELECT MIN(chunk_count), SUM(chunk_count) FROM content"
        ).fetchone()
        assert chunk_totals[0] >= 1
        assert chunk_totals[1] >= 3


def test_rescan_skips_unchanged_via_stat(tmp_path: Path, monkeypatch) -> None:
    _make_mcap_tree(tmp_path, count=2)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")

    summary_calls = {"n": 0}
    original = index_scanner.get_summary

    def _counting_get_summary(*args, **kwargs):
        summary_calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(index_scanner, "get_summary", _counting_get_summary)

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.discovered == 2
        assert stats.stat_skipped == 2
        assert stats.indexed == 0
        assert stats.fingerprint_reused == 0
        assert summary_calls["n"] == 0


def test_moved_file_reuses_fingerprint(tmp_path: Path) -> None:
    p = tmp_path / "orig.mcap"
    p.write_bytes(create_simple_mcap(num_messages=5))
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")
        assert _count(conn, "content") == 1

    moved = tmp_path / "moved.mcap"
    p.rename(moved)

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.deleted == 1
        assert stats.fingerprint_reused == 1
        assert stats.indexed == 0
        assert _count(conn, "content") == 1  # still one content row
        # Three observations: original + tombstone + alias.
        assert _count(conn, "file_observation") == 3
        current_paths = {
            row[0] for row in conn.execute("SELECT abs_path FROM current_file").fetchall()
        }
        assert current_paths == {str(moved.resolve())}


def test_deleted_file_is_removed_from_current_file(tmp_path: Path) -> None:
    p = tmp_path / "rec.mcap"
    p.write_bytes(create_simple_mcap(num_messages=5))
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")

    p.unlink()

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.deleted == 1
        current = conn.execute(
            "SELECT abs_path FROM current_file WHERE abs_path=?",
            (str(p.resolve()),),
        ).fetchone()
        assert current is None
        tombstone = conn.execute(
            "SELECT is_deleted FROM file_observation WHERE abs_path=? ORDER BY id DESC LIMIT 1",
            (str(p.resolve()),),
        ).fetchone()
        assert tombstone == (1,)


def test_duplicate_file_in_same_scan_reuses_single_content_row(tmp_path: Path) -> None:
    payload = create_simple_mcap(num_messages=5)
    (tmp_path / "a.mcap").write_bytes(payload)
    (tmp_path / "b.mcap").write_bytes(payload)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.indexed == 1
        assert stats.fingerprint_reused == 1
        assert _count(conn, "content") == 1
        assert _count(conn, "file_observation") == 2


def test_corrupt_file_records_scan_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mcap"
    bad.write_bytes(b"not a real mcap" * 100)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.errored == 1
        assert stats.indexed == 0
        assert stats.errored_by_kind == {"corrupt": 1}
        errs = conn.execute("SELECT abs_path, error_kind FROM scan_error").fetchall()
        assert len(errs) == 1
        assert errs[0][0] == str(bad.resolve())
        current = conn.execute(
            "SELECT abs_path, summary_fingerprint FROM current_file WHERE abs_path=?",
            (str(bad.resolve()),),
        ).fetchone()
        assert current == (str(bad.resolve()), None)


def test_error_file_skipped_on_rescan(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mcap"
    bad.write_bytes(b"not a real mcap" * 100)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.error_skipped == 1
        assert stats.errored == 0


def test_summaryless_file_is_not_rebuilt_by_default(tmp_path: Path) -> None:
    path = tmp_path / "summaryless.mcap"
    path.write_bytes(create_simple_mcap(num_messages=5))
    _truncate_after_data_section(path)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.discovered == 1
        assert stats.indexed == 0
        assert stats.errored == 1
        assert stats.errored_by_kind == {"no_summary": 1}
        assert _count(conn, "content") == 0
        current = conn.execute(
            "SELECT summary_fingerprint FROM current_file WHERE abs_path=?",
            (str(path.resolve()),),
        ).fetchone()
        assert current == (None,)


def test_summaryless_file_rebuild_requires_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "summaryless.mcap"
    path.write_bytes(create_simple_mcap(num_messages=5))
    _truncate_after_data_section(path)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test", rebuild_missing=True)
        assert stats.indexed == 1
        assert stats.error_skipped == 0
        assert stats.errored == 0
        scan_kind = conn.execute("SELECT scan_kind FROM content").fetchone()[0]
        assert scan_kind == "rebuilt"


def test_corrupt_rescan_clears_current_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "rec.mcap"
    path.write_bytes(create_simple_mcap(num_messages=5))
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")

    path.write_bytes(b"not a real mcap" * 100)

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.errored == 1
        current = conn.execute(
            "SELECT summary_fingerprint FROM current_file WHERE abs_path=?",
            (str(path.resolve()),),
        ).fetchone()
        assert current == (None,)


def test_multi_topic_records_channels_and_chunks(tmp_path: Path) -> None:
    p = tmp_path / "multi.mcap"
    p.write_bytes(create_multi_topic_mcap(["/a", "/b", "/c"], messages_per_topic=5))
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")
        topics = {row[0] for row in conn.execute("SELECT name FROM topic").fetchall()}
        assert topics == {"/a", "/b", "/c"}
        # ``chunk_count`` from the MCAP Statistics record lives on ``content``.
        assert conn.execute("SELECT MAX(chunk_count) FROM content").fetchone()[0] >= 1
        schemas = conn.execute("SELECT name FROM schema").fetchall()
        assert len(schemas) == 1


def test_identical_schemas_are_deduplicated_across_content(tmp_path: Path) -> None:
    """Two distinct MCAPs (different fingerprints) that share schema bytes
    must collapse to one row in the global `schema` table."""
    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"
    a.write_bytes(create_simple_mcap(num_messages=3))
    b.write_bytes(create_simple_mcap(num_messages=7))
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")
        # Two distinct content rows, but only one shared schema.
        assert _count(conn, "content") == 2
        assert _count(conn, "schema") == 1
        # content_schema has one mapping per (content_id, schema_id).
        rows = conn.execute(
            "SELECT cs.content_id, cs.schema_pk_id FROM content_schema cs"
        ).fetchall()
        assert len(rows) == 2
        assert len({pk for _cid, pk in rows}) == 1


def test_byte_cache_rescan_skips_rebuild(tmp_path: Path, monkeypatch) -> None:
    """A byte-identical file under a new path must be aliased via file_fingerprint,
    not re-summarised."""
    payload = create_simple_mcap(num_messages=4)
    (tmp_path / "orig.mcap").write_bytes(payload)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")

    # Now copy the same bytes to a new location and ensure the rebuild path
    # never runs on the second scan.
    moved = tmp_path / "copy.mcap"
    moved.write_bytes(payload)

    rebuild_calls = {"n": 0}
    original_load = index_scanner._load_summary_info

    def _counting_load(*args, **kwargs):
        rebuild_calls["n"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(index_scanner, "_load_summary_info", _counting_load)

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        # First file's stat hits the stat-skip path; the new copy goes through
        # the byte cache and aliases without re-loading the summary.
        assert stats.stat_skipped == 1
        assert stats.fingerprint_reused == 1
        assert rebuild_calls["n"] == 0
        # Still only one content row — the new file aliases the existing one.
        assert _count(conn, "content") == 1


def test_retry_errors_rescans_failed_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mcap"
    bad.write_bytes(b"not a real mcap" * 100)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        scan(tmp_path, conn, pymcap_cli_version="test")

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test")
        assert stats.error_skipped == 1
        assert stats.errored == 0

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test", retry_errors=True)
        assert stats.error_skipped == 0
        assert stats.errored == 1
        assert _count(conn, "scan_error") == 2


def test_scan_with_parallel_jobs(tmp_path: Path) -> None:
    _make_mcap_tree(tmp_path, count=8)
    db = tmp_path / "index.sqlite"

    with open_db(db) as conn:
        stats = scan(tmp_path, conn, pymcap_cli_version="test", jobs=4)
        assert stats.discovered == 8
        assert stats.indexed == 8
        assert stats.errored == 0
        assert _count(conn, "content") == 8
        assert _count(conn, "file_observation") == 8


def test_session_finished_on_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_mcap_tree(tmp_path, count=2)
    db = tmp_path / "index.sqlite"

    original = index_scanner._record_observation
    calls = {"n": 0}

    def _explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(index_scanner, "_record_observation", _explode)

    with open_db(db) as conn:
        with pytest.raises(RuntimeError, match="boom"):
            scan(tmp_path, conn, pymcap_cli_version="test")

        row = conn.execute(
            "SELECT finished_at FROM scan_session ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] is not None
