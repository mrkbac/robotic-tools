"""Synthetic benchmarks for the index ``current_file`` read path.

Run:
  uv run pytest \
    pymcap-cli/tests/benchmark/test_index_current_file_performance.py \
    --benchmark-only -q
"""

from __future__ import annotations

import importlib
import io
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.index._helpers import _path_prefix_where
from pymcap_cli.index.db import open_db
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_PATH_COUNT = 20_000
_OBSERVATIONS_PER_PATH = 3
_GROUP_COUNT = 100
_TARGET_GROUP = 42
_TARGET_INDEX = 1_042
_TARGET_FOLDER = Path(f"/bench/root/group_{_TARGET_GROUP:03d}")
_TARGET_ROOT = Path("/bench/root")
_TARGET_PATH = f"{_TARGET_FOLDER}/file_{_TARGET_INDEX:06d}.mcap"
_EXPECTED_GROUP_COUNT = _PATH_COUNT // _GROUP_COUNT

pytestmark = pytest.mark.usefixtures("_silent_index_command_output")


def _bench_path(index: int) -> str:
    return f"/bench/root/group_{index % _GROUP_COUNT:03d}/file_{index:06d}.mcap"


@pytest.fixture(scope="session")
def index_benchmark_db(tmp_path_factory) -> Path:
    """Create a deterministic v7+ catalog with many observed files."""
    db_path = tmp_path_factory.mktemp("index_current_file_bench") / "index.sqlite"
    with open_db(db_path) as conn:
        conn.execute("BEGIN")
        try:
            root_path = "/bench/root"
            root_id = conn.execute(
                "INSERT INTO file_path(value) VALUES (?)",
                (root_path,),
            ).lastrowid
            assert root_id is not None
            session_id = conn.execute(
                "INSERT INTO scan_session("
                "started_at_ns, root_file_path_id, pymcap_cli_version"
                ") VALUES (?, ?, ?)",
                (1, root_id, "benchmark"),
            ).lastrowid
            assert session_id is not None

            topic_id = conn.execute(
                "INSERT INTO topic(name) VALUES (?)", ("/bench/topic",)
            ).lastrowid
            schema_id = conn.execute(
                "INSERT INTO schema(schema_hash, name, encoding, size_bytes) VALUES (?, ?, ?, ?)",
                ("bench-schema", "bench_msgs/msg/Sample", "ros2msg", 64),
            ).lastrowid
            assert topic_id is not None
            assert schema_id is not None
            channel_signature_id = conn.execute(
                "INSERT INTO channel_signature("
                "topic_id, schema_id, message_encoding, channel_metadata_id"
                ") VALUES (?, ?, ?, ?)",
                (topic_id, schema_id, "cdr", None),
            ).lastrowid
            assert channel_signature_id is not None

            conn.executemany(
                "INSERT INTO content("
                "summary_fingerprint, size_bytes, message_count, schema_count, channel_count, "
                "attachment_count, metadata_count, chunk_count, message_start_time_ns, "
                "message_end_time_ns, sane_message_start_time_ns, sane_message_end_time_ns, "
                "scan_kind, first_seen_at_ns, first_seen_scan_session_id"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        f"s1:{index:032x}",
                        4_096 + index,
                        100,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1_700_000_000_000_000_000 + index,
                        1_700_000_000_000_001_000 + index,
                        1_700_000_000_000_000_000 + index,
                        1_700_000_000_000_001_000 + index,
                        "summary",
                        1,
                        session_id,
                    )
                    for index in range(_PATH_COUNT)
                ),
            )
            conn.executemany(
                "INSERT INTO content_channel("
                "content_id, mcap_channel_id, channel_signature_id, message_count"
                ") VALUES (?, ?, ?, ?)",
                (
                    (content_id, 1, channel_signature_id, 100)
                    for content_id in range(1, _PATH_COUNT + 1)
                ),
            )
            conn.executemany(
                "INSERT INTO content_schema(content_id, mcap_schema_id, schema_id) "
                "VALUES (?, ?, ?)",
                ((content_id, 1, schema_id) for content_id in range(1, _PATH_COUNT + 1)),
            )
            conn.executemany(
                "INSERT INTO file_path(value) VALUES (?)",
                ((_bench_path(index),) for index in range(_PATH_COUNT)),
            )

            path_id_by_index: dict[int, int] = {}
            for path_id, value in conn.execute(
                "SELECT id, value FROM file_path WHERE value <> ?",
                (root_path,),
            ):
                index = int(value.removesuffix(".mcap").rsplit("_", 1)[1])
                path_id_by_index[index] = path_id

            conn.executemany(
                "INSERT INTO file_observation("
                "file_path_id, size_bytes, mtime_ns, inode, file_fingerprint, content_id, "
                "is_deleted, scan_session_id, observed_at_ns"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        path_id_by_index[index],
                        4_096 + index,
                        observation_round,
                        10_000_000 + index,
                        f"ff:{index % 5_000:08x}",
                        index + 1,
                        0,
                        session_id,
                        observation_round,
                    )
                    for observation_round in range(_OBSERVATIONS_PER_PATH)
                    for index in range(_PATH_COUNT)
                ),
            )
            conn.executemany(
                "INSERT INTO scan_error("
                "file_path_id, size_bytes, mtime_ns, scan_session_id, observed_at_ns, "
                "error_kind, error_message"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        path_id_by_index[index],
                        4_096 + index,
                        _OBSERVATIONS_PER_PATH - 1,
                        session_id,
                        _OBSERVATIONS_PER_PATH,
                        "no_summary",
                        "synthetic benchmark error",
                    )
                    for index in range(0, _PATH_COUNT, 50)
                ),
            )
            conn.executemany(
                "INSERT INTO content_current_file_count(content_id, file_count) VALUES (?, ?)",
                ((content_id, 1) for content_id in range(1, _PATH_COUNT + 1)),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return db_path


@pytest.fixture
def index_benchmark_conn(index_benchmark_db: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(index_benchmark_db)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-65536")
        yield conn
    finally:
        conn.close()


@pytest.fixture
def _silent_index_command_output() -> Iterator[None]:
    modules = [
        importlib.import_module(name)
        for name in (
            "pymcap_cli.cmd.index.duplicates_cmd",
            "pymcap_cli.cmd.index.errors_cmd",
            "pymcap_cli.cmd.index.info_cmd",
            "pymcap_cli.cmd.index.query_cmd",
            "pymcap_cli.cmd.index.schemas_cmd",
            "pymcap_cli.cmd.index.sessions_cmd",
            "pymcap_cli.cmd.index.status_cmd",
            "pymcap_cli.cmd.index.timeline_cmd",
            "pymcap_cli.cmd.index.topics_cmd",
            "pymcap_cli.cmd.index.tree_cmd",
        )
    ]
    originals = [(module, module.console) for module in modules]
    try:
        for module in modules:
            module.console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
        yield
    finally:
        for module, console in originals:
            module.console = console


def _run_silent(call: Callable[[], int]) -> int:
    stream = io.StringIO()
    with redirect_stdout(stream):
        return call()


def _benchmark_zero_exit(
    benchmark,
    group: str,
    call: Callable[[], int],
    *,
    rounds: int = 5,
    iterations: int = 2,
) -> None:
    benchmark.group = group
    result = benchmark.pedantic(lambda: _run_silent(call), rounds=rounds, iterations=iterations)
    assert result == 0


def test_benchmark_current_file_exact_lookup(benchmark, index_benchmark_conn: sqlite3.Connection):
    benchmark.group = "index_current_file_exact_lookup"
    result = benchmark.pedantic(
        lambda: index_benchmark_conn.execute(
            "SELECT summary_fingerprint FROM current_file WHERE abs_path = ?",
            (_TARGET_PATH,),
        ).fetchone()[0],
        rounds=10,
        iterations=20,
    )
    assert result == f"s1:{_TARGET_INDEX:032x}"


def test_benchmark_current_file_prefix_count(benchmark, index_benchmark_conn: sqlite3.Connection):
    benchmark.group = "index_current_file_prefix_count"
    where, params = _path_prefix_where(_TARGET_FOLDER)
    result = benchmark.pedantic(
        lambda: index_benchmark_conn.execute(
            f"SELECT COUNT(*) FROM current_file {where}",  # noqa: S608
            params,
        ).fetchone()[0],
        rounds=10,
        iterations=10,
    )
    assert result == _EXPECTED_GROUP_COUNT


def test_benchmark_current_file_full_count(benchmark, index_benchmark_conn: sqlite3.Connection):
    benchmark.group = "index_current_file_full_count"
    result = benchmark.pedantic(
        lambda: index_benchmark_conn.execute("SELECT COUNT(*) FROM current_file").fetchone()[0],
        rounds=10,
        iterations=5,
    )
    assert result == _PATH_COUNT


def test_benchmark_current_file_topic_fanout(benchmark, index_benchmark_conn: sqlite3.Connection):
    benchmark.group = "index_current_file_topic_fanout"
    where, params = _path_prefix_where(_TARGET_FOLDER)
    predicate = where.removeprefix("WHERE ").replace("abs_path", "cf.abs_path")
    sql = (
        "SELECT COUNT(DISTINCT cs.topic_id), COUNT(*) "  # noqa: S608
        "FROM current_file cf "
        "JOIN content_channel cc ON cc.content_id = cf.content_id "
        "JOIN channel_signature cs ON cs.id = cc.channel_signature_id "
        f"WHERE {predicate}"
    )
    result = benchmark.pedantic(
        lambda: index_benchmark_conn.execute(sql, params).fetchone(),
        rounds=10,
        iterations=10,
    )
    assert result == (1, _EXPECTED_GROUP_COUNT)


def test_benchmark_index_status_command(
    benchmark,
    index_benchmark_db: Path,
):
    status_module = importlib.import_module("pymcap_cli.cmd.index.status_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_status",
        lambda: status_module.status_cmd(_TARGET_FOLDER, db=index_benchmark_db),
    )


def test_benchmark_index_query_command(
    benchmark,
    index_benchmark_db: Path,
):
    query_module = importlib.import_module("pymcap_cli.cmd.index.query_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_query",
        lambda: query_module.query_cmd(
            _TARGET_FOLDER,
            topic="/bench/topic",
            sort_by="path",
            limit=25,
            format="json",
            db=index_benchmark_db,
        ),
    )


def test_benchmark_index_topics_command(
    benchmark,
    index_benchmark_db: Path,
):
    topics_module = importlib.import_module("pymcap_cli.cmd.index.topics_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_topics",
        lambda: topics_module.topics_cmd(
            prefix="/bench",
            limit=25,
            format="json",
            db=index_benchmark_db,
        ),
        iterations=1,
    )


def test_benchmark_index_schemas_command(
    benchmark,
    index_benchmark_db: Path,
):
    schemas_module = importlib.import_module("pymcap_cli.cmd.index.schemas_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_schemas",
        lambda: schemas_module.schemas_cmd(
            prefix="bench_msgs",
            limit=25,
            format="json",
            db=index_benchmark_db,
        ),
        iterations=1,
    )


def test_benchmark_index_timeline_command(
    benchmark,
    index_benchmark_db: Path,
):
    timeline_module = importlib.import_module("pymcap_cli.cmd.index.timeline_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_timeline",
        lambda: timeline_module.timeline_cmd(
            _TARGET_FOLDER,
            bucket="day",
            limit=25,
            db=index_benchmark_db,
        ),
        iterations=1,
    )


def test_benchmark_index_duplicates_command(
    benchmark,
    index_benchmark_db: Path,
):
    duplicates_module = importlib.import_module("pymcap_cli.cmd.index.duplicates_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_duplicates",
        lambda: duplicates_module.duplicates_cmd(
            _TARGET_FOLDER,
            min_copies=2,
            limit=25,
            format="json",
            db=index_benchmark_db,
        ),
        iterations=1,
    )


def test_benchmark_index_errors_command(
    benchmark,
    index_benchmark_db: Path,
):
    errors_module = importlib.import_module("pymcap_cli.cmd.index.errors_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_errors",
        lambda: errors_module.errors_cmd(
            _TARGET_ROOT,
            limit=25,
            format="json",
            db=index_benchmark_db,
        ),
        iterations=1,
    )


def test_benchmark_index_sessions_command(
    benchmark,
    index_benchmark_db: Path,
):
    sessions_module = importlib.import_module("pymcap_cli.cmd.index.sessions_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_sessions",
        lambda: sessions_module.sessions_cmd(
            _TARGET_ROOT,
            limit=25,
            format="json",
            db=index_benchmark_db,
        ),
        iterations=1,
    )


def test_benchmark_index_info_command(
    benchmark,
    index_benchmark_db: Path,
):
    info_module = importlib.import_module("pymcap_cli.cmd.index.info_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_info",
        lambda: info_module.info_cmd(
            _TARGET_PATH,
            format="json",
            db=index_benchmark_db,
        ),
    )


def test_benchmark_index_tree_command(
    benchmark,
    index_benchmark_db: Path,
):
    tree_module = importlib.import_module("pymcap_cli.cmd.index.tree_cmd")
    _benchmark_zero_exit(
        benchmark,
        "index_command_tree",
        lambda: tree_module.tree_cmd(
            _TARGET_FOLDER,
            max_depth=2,
            min_files=1,
            db=index_benchmark_db,
        ),
        iterations=1,
    )
