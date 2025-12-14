"""Performance benchmarks comparing pymcap-cli with official mcap CLI."""

import subprocess
from pathlib import Path

from pymcap_cli.mcap_processor import (
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)


def test_benchmark_recover_nuscenes_pymcap(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Benchmark pymcap-cli recover on real-world nuScenes file."""
    benchmark.group = "recover_nuscenes"
    output_file = tmp_path / "output.mcap"

    def run_recover():
        file_size = nuscenes_mcap.stat().st_size
        with nuscenes_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )
            processor = McapProcessor(options)
            processor.process(output_stream)

        # Clean up for next iteration
        if output_file.exists():
            output_file.unlink()

    return benchmark.pedantic(run_recover, rounds=3, iterations=1)


def test_benchmark_recover_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Benchmark official mcap CLI recover on real-world nuScenes file."""
    benchmark.group = "recover_nuscenes"
    output_file = tmp_path / "output.mcap"

    def run_recover():
        subprocess.run(
            [
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
        file_size = nuscenes_mcap.stat().st_size

        with nuscenes_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_topic_regex=["/CAM_.*"],
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )
            processor = McapProcessor(options)
            processor.process(output_stream)

        # Clean up for next iteration
        if output_file.exists():
            output_file.unlink()

    return benchmark.pedantic(run_filter, rounds=3, iterations=1)


def test_benchmark_filter_nuscenes_mcap_cli(benchmark, nuscenes_mcap: Path, tmp_path: Path):
    """Benchmark official mcap CLI filter on real-world nuScenes file (camera topics only)."""
    benchmark.group = "filter_nuscenes"
    output_file = tmp_path / "output.mcap"

    def run_filter():
        subprocess.run(
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
            check=True,
            capture_output=True,
        )
        # Clean up for next iteration
        if output_file.exists():
            output_file.unlink()

    return benchmark.pedantic(run_filter, rounds=3, iterations=1)
