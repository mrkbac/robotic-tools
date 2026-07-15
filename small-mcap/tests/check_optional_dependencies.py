"""Smoke-test the built small-mcap wheel and each compression extra in isolation."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Scenario:
    extra: str | None
    script: str


ROUND_TRIP = """
import io
from small_mcap import CompressionType, McapWriter, read_message

buffer = io.BytesIO()
writer = McapWriter(buffer, compression=CompressionType.__COMPRESSION__, chunk_size=64)
writer.start()
writer.add_schema(1, "Test", "json", b"{}")
writer.add_channel(1, "/test", "json", 1)
writer.add_message(1, 1, b"{}", 1)
writer.finish()
buffer.seek(0)
assert next(read_message(buffer))[2].data == b"{}"
"""

SCENARIOS = (
    Scenario(None, "import small_mcap; assert small_mcap.CompressionType.NONE.value == ''"),
    Scenario("lz4", ROUND_TRIP.replace("__COMPRESSION__", "LZ4")),
    Scenario("zstd", ROUND_TRIP.replace("__COMPRESSION__", "ZSTD")),
    Scenario(
        "compression",
        "import lz4.frame; import zstandard; " + ROUND_TRIP.replace("__COMPRESSION__", "ZSTD"),
    ),
    Scenario(
        "dev",
        "import mcap.reader; import pybag.mcap_reader; import rosbags.rosbag2",
    ),
)


def _requirement(wheel: Path, extra: str | None) -> str:
    name = "small-mcap" if extra is None else f"small-mcap[{extra}]"
    return f"{name} @ {wheel.resolve().as_uri()}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    wheel: Path = args.wheel
    if not wheel.is_file():
        raise FileNotFoundError(wheel)

    for scenario in SCENARIOS:
        label = scenario.extra or "base"
        print(f"checking small-mcap {label}", flush=True)
        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--with",
                _requirement(wheel, scenario.extra),
                "python",
                "-c",
                scenario.script,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
