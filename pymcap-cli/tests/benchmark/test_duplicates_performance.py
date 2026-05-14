"""Performance benchmarks for `pymcap-cli duplicates`.

The duplicates command has four stages (summary scan, candidate pair-scan,
message-index read, indexed/payload comparison). These benchmarks record
wall time for the two user-visible modes — with and without
``--compare-payloads`` — over a directory of duplicate MCAPs, so each
optimization tier can be measured against a stable baseline.

Run:
  uv run pytest pymcap-cli/tests/benchmark/test_duplicates_performance.py --benchmark-only -q
"""

from __future__ import annotations

import io
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd import duplicates_cmd
from rich.console import Console
from typing_extensions import Self

if TYPE_CHECKING:
    from pathlib import Path

_SUMMARY_COPIES = 16
_PAYLOAD_COPIES = 8


def _require_mcap(fixture: Path) -> None:
    if not fixture.exists():
        pytest.skip(f"Benchmark fixture missing: {fixture}")


def _clone_or_copy(src: Path, dst: Path) -> None:
    """Clone via APFS reflink (``cp -c``) when supported; otherwise copy."""
    try:
        subprocess.run(["cp", "-c", str(src), str(dst)], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copyfile(src, dst)


class _DummyProgress:
    """Drop-in for ``rich.progress.Progress`` that does nothing.

    Suppresses per-iteration progress redraw cost so the benchmark measures
    the pipeline, not the UI.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def add_task(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def update(self, *_args: object, **_kwargs: object) -> None:
        return None


def _make_corpus(src: Path, dest_dir: Path, copies: int) -> Path:
    for index in range(copies):
        _clone_or_copy(src, dest_dir / f"copy_{index}.mcap")
    return dest_dir


@pytest.fixture(scope="session")
def duplicates_corpus_large(nuscenes_mcap: Path, tmp_path_factory) -> Path:
    """Multi-copy corpus of the nuScenes fixture for summary-stage timing."""
    _require_mcap(nuscenes_mcap)
    return _make_corpus(
        nuscenes_mcap,
        tmp_path_factory.mktemp("duplicates_corpus_large"),
        _SUMMARY_COPIES,
    )


@pytest.fixture(scope="session")
def duplicates_corpus_payload(test_fixtures, tmp_path_factory) -> Path:
    """Multi-copy corpus of a ~10 MB fixture for full payload-verification timing."""
    src = test_fixtures["large_10mb"]
    _require_mcap(src)
    return _make_corpus(
        src,
        tmp_path_factory.mktemp("duplicates_corpus_payload"),
        _PAYLOAD_COPIES,
    )


def _run_duplicates(corpus: Path, *, compare_payloads: bool) -> int:
    sink_console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=180)
    original_console = duplicates_cmd.console
    original_create_progress = duplicates_cmd._create_progress
    duplicates_cmd.console = sink_console
    duplicates_cmd._create_progress = _DummyProgress
    try:
        return duplicates_cmd.duplicates(
            [str(corpus)],
            include_all=False,
            rebuild_missing=True,
            compare_payloads=compare_payloads,
        )
    finally:
        duplicates_cmd.console = original_console
        duplicates_cmd._create_progress = original_create_progress


def test_benchmark_duplicates_summary_only(benchmark, duplicates_corpus_large: Path) -> None:
    """16 copies of nuScenes (~451 MB each), no payload verification."""
    benchmark.group = "duplicates_summary_only"
    benchmark.pedantic(
        lambda: _run_duplicates(duplicates_corpus_large, compare_payloads=False),
        rounds=3,
        iterations=1,
    )


def test_benchmark_duplicates_compare_payloads(benchmark, duplicates_corpus_payload: Path) -> None:
    """8 copies of the ~10 MB fixture, with ``--compare-payloads`` enabled."""
    benchmark.group = "duplicates_compare_payloads"
    benchmark.pedantic(
        lambda: _run_duplicates(duplicates_corpus_payload, compare_payloads=True),
        rounds=3,
        iterations=1,
    )
