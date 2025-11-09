"""Performance benchmarks comparing pymcap-cli with official mcap CLI."""

import subprocess
from pathlib import Path

from pymcap_cli.mcap_processor import (
    McapProcessor,
    ProcessingOptions,
    compile_topic_patterns,
)


def test_benchmark_recover_nuscenes_pymcap(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Benchmark pymcap-cli recover on real-world nuScenes file."""
    benchmark.group = "recover_nuscenes"
    output_file = tmp_path / "output.mcap"

    def run_recover():
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )
        processor = McapProcessor(options)
        file_size = nuscenes_mcap.stat().st_size

        with nuscenes_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            processor.process([input_stream], output_stream, [file_size])

        # Clean up for next iteration
        if output_file.exists():
            output_file.unlink()

    return benchmark.pedantic(run_recover, rounds=3, iterations=1)


def test_benchmark_recover_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Benchmark official mcap CLI recover on real-world nuScenes file."""
    benchmark.group = "recover_nuscenes"
    output_file = tmp_path / "output.mcap"

    def run_recover():
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "mcap",
                "recover",
                str(nuscenes_mcap),
                "-o",
                str(output_file),
                "--compression",
                "zstd",
            ],
            check=True,
            capture_output=True,
        )
        # Clean up for next iteration
        if output_file.exists():
            output_file.unlink()

    return benchmark.pedantic(run_recover, rounds=3, iterations=1)


def test_benchmark_filter_nuscenes_pymcap(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Benchmark pymcap-cli filter on real-world nuScenes file (camera topics only)."""
    benchmark.group = "filter_nuscenes"
    output_file = tmp_path / "output.mcap"

    def run_filter():
        include_patterns = compile_topic_patterns(["/CAM_.*"])
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            include_topics=include_patterns,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )
        processor = McapProcessor(options)
        file_size = nuscenes_mcap.stat().st_size

        with nuscenes_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            processor.process([input_stream], output_stream, [file_size])

        # Clean up for next iteration
        if output_file.exists():
            output_file.unlink()

    return benchmark.pedantic(run_filter, rounds=3, iterations=1)


def test_benchmark_filter_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Benchmark official mcap CLI filter on real-world nuScenes file (camera topics only)."""
    benchmark.group = "filter_nuscenes"
    output_file = tmp_path / "output.mcap"

    def run_filter():
        subprocess.run(  # noqa: S603
            [  # noqa: S607
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
            check=True,
            capture_output=True,
        )
        # Clean up for next iteration
        if output_file.exists():
            output_file.unlink()

    return benchmark.pedantic(run_filter, rounds=3, iterations=1)
