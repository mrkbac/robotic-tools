"""Performance benchmarks comparing pymcap-cli with the official `mcap` CLI.

Each scenario defines two tests — one invoking `pymcap-cli` via its Python API
(no interpreter-startup cost) and one invoking the Go `mcap` binary via
subprocess. Both are assigned the same `benchmark.group`, so `pytest-benchmark`
renders a side-by-side comparison table.

Run:
  uv run pytest tests/benchmark/test_performance.py -q
  uv run pytest tests/benchmark/test_performance.py -q --benchmark-only
  uv run pytest tests/benchmark/test_performance.py -q --benchmark-save=baseline

Skipping behaviour:
  - `mcap_cli` comparisons skip automatically if the `mcap` binary is missing.
  - All benchmarks skip if the nuScenes fixture file isn't available (large
    file, gated behind repo-local `data/data/...`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)

MCAP_CLI_AVAILABLE = shutil.which("mcap") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_mcap(fixture: Path) -> None:
    if not fixture.exists():
        pytest.skip(f"Benchmark fixture missing: {fixture}")


def _run_pymcap(
    source: Path,
    output_file: Path,
    *,
    output_compression: str,
    include_topic_regex: list[str] | None = None,
) -> None:
    """Run one full pymcap-cli transform through the Python API."""
    with source.open("rb") as input_stream, output_file.open("wb") as output_stream:
        options = ProcessingOptions(
            inputs=[
                InputFile(
                    stream=input_stream,
                    size=source.stat().st_size,
                    options=InputOptions.from_args(
                        include_topic_regex=include_topic_regex or [],
                    ),
                ),
            ],
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(
                compression=output_compression,
                chunk_size=4 * 1024 * 1024,
            ),
        )
        McapProcessor(options).process(output_stream)
    if output_file.exists():
        output_file.unlink()


def _run_mcap_cli(args: list[str], output_file: Path) -> None:
    """Invoke the Go `mcap` binary. Skips if it isn't on PATH."""
    if not MCAP_CLI_AVAILABLE:
        pytest.skip("`mcap` binary not available on PATH")
    subprocess.run(args, check=True, capture_output=True)
    if output_file.exists():
        output_file.unlink()


# ---------------------------------------------------------------------------
# recover (already present — kept for continuity)
# ---------------------------------------------------------------------------


def test_benchmark_recover_nuscenes_pymcap(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """pymcap-cli recover on the nuScenes fixture."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "recover_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_pymcap(nuscenes_mcap, output_file, output_compression="zstd"),
        rounds=3,
        iterations=1,
    )


def test_benchmark_recover_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Go mcap recover on the nuScenes fixture."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "recover_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_mcap_cli(
            [
                "mcap",
                "recover",
                str(nuscenes_mcap),
                "-o",
                str(output_file),
                "--compression",
                "zstd",
            ],
            output_file,
        ),
        rounds=3,
        iterations=1,
    )


# ---------------------------------------------------------------------------
# filter (kept for continuity)
# ---------------------------------------------------------------------------


def test_benchmark_filter_nuscenes_pymcap(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """pymcap-cli filter: keep CAM_ topics only (zstd → zstd)."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "filter_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_pymcap(
            nuscenes_mcap,
            output_file,
            output_compression="zstd",
            include_topic_regex=["/CAM_.*"],
        ),
        rounds=3,
        iterations=1,
    )


def test_benchmark_filter_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Go mcap filter: keep CAM_ topics only (zstd → zstd)."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "filter_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_mcap_cli(
            [
                "mcap",
                "filter",
                str(nuscenes_mcap),
                "-o",
                str(output_file),
                "-y",
                "/CAM_.*",
                "--output-compression",
                "zstd",
            ],
            output_file,
        ),
        rounds=3,
        iterations=1,
    )


# ---------------------------------------------------------------------------
# compression matrix (fast-copy / recompress / decompress / compress)
# ---------------------------------------------------------------------------


def test_benchmark_filter_z2z_nuscenes_pymcap(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """pymcap-cli: full-pass zstd → zstd (exercises fast-copy `add_chunk_raw`)."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "filter_z2z_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_pymcap(nuscenes_mcap, output_file, output_compression="zstd"),
        rounds=3,
        iterations=1,
    )


def test_benchmark_filter_z2z_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Go mcap: full-pass zstd → zstd."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "filter_z2z_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_mcap_cli(
            [
                "mcap",
                "filter",
                str(nuscenes_mcap),
                "-o",
                str(output_file),
                "--output-compression",
                "zstd",
                "--chunk-size",
                "4194304",
            ],
            output_file,
        ),
        rounds=3,
        iterations=1,
    )


def test_benchmark_decompress_nuscenes_pymcap(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """pymcap-cli: zstd → uncompressed (exercises parallel RECOMPRESS / DECODE)."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "decompress_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_pymcap(nuscenes_mcap, output_file, output_compression="none"),
        rounds=3,
        iterations=1,
    )


def test_benchmark_decompress_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Go mcap: zstd → uncompressed."""
    _require_mcap(nuscenes_mcap)
    benchmark.group = "decompress_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_mcap_cli(
            ["mcap", "decompress", str(nuscenes_mcap), "-o", str(output_file)],
            output_file,
        ),
        rounds=3,
        iterations=1,
    )


# ---------------------------------------------------------------------------
# compress (needs an uncompressed source) — materialise a one-off fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def nuscenes_uncompressed(nuscenes_mcap: Path, tmp_path_factory) -> Path:
    """Decompress nuScenes once per session so compress benches have a fast source."""
    _require_mcap(nuscenes_mcap)
    dest = tmp_path_factory.mktemp("bench") / "nuscenes-uncompressed.mcap"
    if not dest.exists():
        _run_pymcap(nuscenes_mcap, dest, output_compression="none")
        # `_run_pymcap` deletes the output after use — re-create so the fixture
        # stays available for the benchmarks that follow.
        with nuscenes_mcap.open("rb") as input_stream, dest.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=nuscenes_mcap.stat().st_size,
                        options=InputOptions.from_args(),
                    ),
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    compression="none",
                    chunk_size=4 * 1024 * 1024,
                ),
            )
            McapProcessor(options).process(output_stream)
    return dest


def test_benchmark_compress_nuscenes_pymcap(benchmark, nuscenes_uncompressed: Path, tmp_path: Path):
    """pymcap-cli: uncompressed → zstd (exercises parallel compression workers)."""
    benchmark.group = "compress_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_pymcap(nuscenes_uncompressed, output_file, output_compression="zstd"),
        rounds=3,
        iterations=1,
    )


def test_benchmark_compress_nuscenes_mcap_cli(
    benchmark, nuscenes_uncompressed: Path, tmp_path: Path
):
    """Go mcap: uncompressed → zstd."""
    benchmark.group = "compress_nuscenes"
    output_file = tmp_path / "output.mcap"
    benchmark.pedantic(
        lambda: _run_mcap_cli(
            [
                "mcap",
                "compress",
                str(nuscenes_uncompressed),
                "-o",
                str(output_file),
                "--chunk-size",
                "4194304",
            ],
            output_file,
        ),
        rounds=3,
        iterations=1,
    )
