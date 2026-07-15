"""Smoke-test the built codec-support wheel and each extra in isolation."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Scenario:
    extra: str | None
    script: str


SCENARIOS = (
    Scenario(None, "import mcap_codec_support; import mcap_codec_support._schemas"),
    Scenario(
        "ros2",
        "import mcap_codec_support; from mcap_ros2_support_fast.decoder import DecoderFactory",
    ),
    Scenario(
        "video",
        "from mcap_codec_support.video import VideoDecompressFactory; VideoDecompressFactory()",
    ),
    Scenario(
        "pointcloud",
        "from mcap_codec_support.pointcloud import PointCloudDecompressFactory; "
        "PointCloudDecompressFactory()",
    ),
    Scenario(
        "draco",
        "from mcap_codec_support.pointcloud import DracoPointCloudCompressor; "
        "DracoPointCloudCompressor()",
    ),
    Scenario(
        "all",
        "from mcap_codec_support import create_decoder_factories; create_decoder_factories()",
    ),
)


def _requirement(wheel: Path, extra: str | None) -> str:
    name = "mcap-codec-support" if extra is None else f"mcap-codec-support[{extra}]"
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
        print(f"checking mcap-codec-support {label}", flush=True)
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
