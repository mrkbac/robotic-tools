"""Synthetic benchmarks for index discovery and unchanged rescans.

Run:
  uv run pytest \
    pymcap-cli/tests/benchmark/test_index_scan_performance.py \
    --benchmark-only -q
"""

from __future__ import annotations

import importlib
import io
import itertools
import os
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.index.scan_cmd import scan_cmd
from pymcap_cli.cmd.index.stats_cmd import stats_cmd
from pymcap_cli.index import scanner as index_scanner
from pymcap_cli.index.db import open_db
from pymcap_cli.index.scanner import ScanStats, _iter_mcap_files, scan
from rich.console import Console
from small_mcap import McapWriter

from tests.fixtures.mcap_generator import create_simple_mcap

pytestmark = pytest.mark.benchmark

if TYPE_CHECKING:
    from pathlib import Path

_FILE_COUNT = 2_000
_DIRECTORY_COUNT = 100
_UNIQUE_FILE_COUNT = 500
_CHANGED_FILE_COUNT = 100
_HISTORICAL_CONTENT_COUNT = 20_000
_scan_cmd_module = importlib.import_module("pymcap_cli.cmd.index.scan_cmd")


def _create_value_mcap(value: int) -> bytes:
    output = io.BytesIO()
    writer = McapWriter(output)
    writer.start()
    writer.add_schema(1, "sample", "json", b"{}")
    writer.add_channel(1, "/sample", "json", 1)
    writer.add_message(1, value, f'{{"value":{value}}}'.encode(), value)
    writer.finish()
    return output.getvalue()


@pytest.fixture(scope="session")
def indexed_tree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("index_scan_bench")
    payload = create_simple_mcap(num_messages=1)
    directories = [root / f"group_{index:03d}" for index in range(_DIRECTORY_COUNT)]
    for directory in directories:
        directory.mkdir()
    for index in range(_FILE_COUNT):
        (directories[index % _DIRECTORY_COUNT] / f"recording_{index:06d}.mcap").write_bytes(payload)

    db_path = root / "index.sqlite"
    with open_db(db_path) as conn:
        stats = scan(root, conn, pymcap_cli_version="benchmark", jobs=8)
    assert stats.discovered == _FILE_COUNT
    return root, db_path


@pytest.fixture(scope="session")
def unique_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("index_scan_unique_bench")
    directories = [root / f"group_{index:03d}" for index in range(25)]
    for directory in directories:
        directory.mkdir()
    for index in range(_UNIQUE_FILE_COUNT):
        path = directories[index % len(directories)] / f"recording_{index:06d}.mcap"
        path.write_bytes(_create_value_mcap(index))
    return root


@pytest.fixture(scope="session")
def large_history_catalog(indexed_tree: tuple[Path, Path]) -> tuple[Path, Path]:
    root, _ = indexed_tree
    db_path = root / "large_history.sqlite"
    with open_db(db_path) as conn:
        stats = scan(root, conn, pymcap_cli_version="benchmark", jobs=8)
        assert stats.discovered == _FILE_COUNT
        session_id = conn.execute("SELECT MAX(id) FROM scan_session").fetchone()[0]
        history_path_id = conn.execute(
            "INSERT INTO file_path(value) VALUES (?)",
            ("/outside-benchmark/history.mcap",),
        ).lastrowid
        assert history_path_id is not None
        conn.executemany(
            "INSERT INTO content("
            "summary_fingerprint, size_bytes, message_count, schema_count, channel_count, "
            "attachment_count, metadata_count, chunk_count, message_start_time_ns, "
            "message_end_time_ns, scan_kind, first_seen_at_ns, first_seen_scan_session_id"
            ") VALUES (?, 1, 0, 0, 0, 0, 0, 0, 0, 0, 'summary', 1, ?)",
            ((f"history:{index:032x}", session_id) for index in range(_HISTORICAL_CONTENT_COUNT)),
        )
        history_content_ids = conn.execute(
            "SELECT id FROM content WHERE summary_fingerprint LIKE 'history:%' ORDER BY id"
        ).fetchall()
        conn.executemany(
            "INSERT INTO file_observation("
            "file_path_id, size_bytes, mtime_ns, inode, file_fingerprint, content_id, "
            "scan_session_id, observed_at_ns"
            ") VALUES (?, 1, 1, NULL, ?, ?, ?, 1)",
            (
                (history_path_id, f"history-file:{index:032x}", content_id, session_id)
                for index, (content_id,) in enumerate(history_content_ids)
            ),
        )
    return root, db_path


def test_benchmark_index_scan_unchanged(benchmark, indexed_tree: tuple[Path, Path]) -> None:
    root, db_path = indexed_tree

    def run_scan() -> ScanStats:
        with open_db(db_path) as conn:
            return scan(root, conn, pymcap_cli_version="benchmark", jobs=8)

    benchmark.group = "index_scan_unchanged"
    stats = benchmark.pedantic(run_scan, rounds=10, iterations=1)
    assert stats.discovered == _FILE_COUNT
    assert stats.stat_skipped == _FILE_COUNT


def test_benchmark_index_scan_cmd_unchanged(
    benchmark,
    indexed_tree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db_path = indexed_tree
    output = io.StringIO()
    monkeypatch.setattr(
        _scan_cmd_module,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )

    def setup() -> tuple[tuple[()], dict[str, str]]:
        output.seek(0)
        output.truncate()
        return (), {}

    benchmark.group = "index_scan_cmd_unchanged"
    result = benchmark.pedantic(
        lambda: scan_cmd(root, db=db_path, jobs=8),
        setup=setup,
        rounds=10,
        iterations=1,
    )
    assert result == 0


def test_benchmark_index_scan_unchanged_large_history(
    benchmark,
    large_history_catalog: tuple[Path, Path],
) -> None:
    root, db_path = large_history_catalog

    def run_scan() -> ScanStats:
        with open_db(db_path) as conn:
            return scan(root, conn, pymcap_cli_version="benchmark", jobs=8)

    benchmark.group = "index_scan_unchanged_large_history"
    stats = benchmark.pedantic(run_scan, rounds=10, iterations=1)
    assert stats.discovered == _FILE_COUNT
    assert stats.stat_skipped == _FILE_COUNT


def test_benchmark_index_scan_new(
    benchmark,
    indexed_tree: tuple[Path, Path],
) -> None:
    root, _ = indexed_tree
    run_ids = itertools.count()

    def setup() -> tuple[tuple[Path], dict[str, str]]:
        db_path = root / f"new_scan_{next(run_ids)}.sqlite"
        with open_db(db_path):
            pass
        return (db_path,), {}

    def run_scan(db_path: Path) -> ScanStats:
        with open_db(db_path) as conn:
            return scan(root, conn, pymcap_cli_version="benchmark", jobs=8)

    benchmark.group = "index_scan_new"
    stats = benchmark.pedantic(run_scan, setup=setup, rounds=10, iterations=1)
    assert stats.discovered == _FILE_COUNT
    assert stats.indexed == 1
    assert stats.fingerprint_reused == _FILE_COUNT - 1


def test_benchmark_index_scan_new_unique(
    benchmark,
    unique_tree: Path,
) -> None:
    run_ids = itertools.count()

    def setup() -> tuple[tuple[Path], dict[str, str]]:
        db_path = unique_tree / f"new_unique_scan_{next(run_ids)}.sqlite"
        with open_db(db_path):
            pass
        return (db_path,), {}

    def run_scan(db_path: Path) -> ScanStats:
        with open_db(db_path) as conn:
            return scan(unique_tree, conn, pymcap_cli_version="benchmark", jobs=8)

    benchmark.group = "index_scan_new_unique"
    stats = benchmark.pedantic(run_scan, setup=setup, rounds=10, iterations=1)
    assert stats.discovered == _UNIQUE_FILE_COUNT
    assert stats.indexed == _UNIQUE_FILE_COUNT


@pytest.mark.parametrize("jobs", [1, 2, 4, 8])
def test_benchmark_index_scan_changed_subset(
    benchmark,
    indexed_tree: tuple[Path, Path],
    jobs: int,
) -> None:
    root, db_path = indexed_tree
    changed_files = sorted(root.rglob("*.mcap"))[:_CHANGED_FILE_COUNT]
    timestamps = itertools.count(time.time_ns())

    def setup() -> tuple[tuple[()], dict[str, str]]:
        mtime_ns = next(timestamps)
        for path in changed_files:
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, mtime_ns))
        return (), {}

    def run_scan() -> ScanStats:
        with open_db(db_path) as conn:
            return scan(root, conn, pymcap_cli_version="benchmark", jobs=jobs)

    benchmark.group = "index_scan_changed_subset"
    stats = benchmark.pedantic(run_scan, setup=setup, rounds=10, iterations=1)
    assert stats.discovered == _FILE_COUNT
    assert stats.stat_skipped == _FILE_COUNT - _CHANGED_FILE_COUNT
    assert stats.fingerprint_reused == _CHANGED_FILE_COUNT


def test_benchmark_index_stats_command(benchmark, indexed_tree: tuple[Path, Path]) -> None:
    root, db_path = indexed_tree

    def run_stats() -> int:
        with redirect_stdout(io.StringIO()):
            return stats_cmd(
                root,
                query=["maximum=/test.i.@@max"],
                format="json",
                db=db_path,
            )

    benchmark.group = "index_stats"
    result = benchmark.pedantic(run_stats, rounds=3, iterations=1)
    assert result == 0


@pytest.mark.parametrize("workers", [None, 1, 8])
def test_benchmark_index_walk(
    benchmark,
    indexed_tree: tuple[Path, Path],
    workers: int | None,
) -> None:
    root, _ = indexed_tree
    benchmark.group = "index_walk"
    paths = benchmark.pedantic(
        lambda: list(_iter_mcap_files(root, recurse=True, walker_workers=workers)),
        rounds=10,
        iterations=1,
    )
    assert len(paths) == _FILE_COUNT


@pytest.mark.parametrize("workers", [None, 1, 8])
def test_benchmark_index_walk_slow_mount(
    benchmark,
    indexed_tree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    workers: int | None,
) -> None:
    root, _ = indexed_tree
    scandir = index_scanner.os.scandir

    def delayed_scandir(path):
        time.sleep(0.002)
        return scandir(path)

    monkeypatch.setattr(index_scanner.os, "scandir", delayed_scandir)
    benchmark.group = "index_walk_slow_mount"
    paths = benchmark.pedantic(
        lambda: list(_iter_mcap_files(root, recurse=True, walker_workers=workers)),
        rounds=5,
        iterations=1,
    )
    assert len(paths) == _FILE_COUNT
