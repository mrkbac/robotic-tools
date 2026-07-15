"""Smoke-test the built wheel with the base install and each extra in isolation."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Scenario:
    extra: str | None
    command: tuple[str, ...]


SCENARIOS = (
    Scenario("bridge", ("pymcap-cli", "bridge", "--help")),
    Scenario("bridge-proxy", ("pymcap-cli", "bridge", "proxy", "--help")),
    Scenario("draco", ("python", "-c", "import DracoPy; import mcap_codec_support.pointcloud")),
    Scenario("image", ("pymcap-cli", "export-images", "--help")),
    Scenario("parquet", ("pymcap-cli", "export-parquet", "--help")),
    Scenario("plot", ("pymcap-cli", "plot", "--help")),
    Scenario("pointcloud", ("pymcap-cli", "export-pcd", "--help")),
    Scenario("serve", ("pymcap-cli", "index", "serve", "--help")),
    Scenario("video", ("pymcap-cli", "video", "--help")),
    Scenario("xxhash", ("pymcap-cli", "index", "--help")),
    Scenario("lite", ("pymcap-cli", "bridge", "--help")),
    Scenario("all", ("pymcap-cli", "--help")),
)

CREATE_MCAP = """
import sys
from pathlib import Path
from small_mcap import McapWriter

with Path(sys.argv[1]).open("wb") as stream:
    writer = McapWriter(stream)
    writer.start(profile="test", library="optional-dependency-smoke")
    writer.finish()
"""


def _requirement(wheel: Path, extra: str | None = None) -> str:
    name = "pymcap-cli" if extra is None else f"pymcap-cli[{extra}]"
    return f"{name} @ {wheel.resolve().as_uri()}"


def _run(wheel: Path, scenario: Scenario) -> None:
    requirement = _requirement(wheel, scenario.extra)
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        requirement,
        *scenario.command,
    ]
    label = scenario.extra or "base"
    print(f"checking {label}: {' '.join(scenario.command)}", flush=True)
    subprocess.run(command, check=True)


def _run_base_compress(wheel: Path, directory: Path) -> None:
    requirement = _requirement(wheel)
    source = directory / "input.mcap"
    output = directory / "compressed.mcap"
    prefix = ["uv", "run", "--isolated", "--no-project", "--with", requirement]

    subprocess.run([*prefix, "python", "-c", CREATE_MCAP, str(source)], check=True)
    subprocess.run(
        [*prefix, "pymcap-cli", "compress", str(source), str(output)],
        check=True,
    )
    if not output.is_file():
        raise RuntimeError("base compress smoke test did not create an output MCAP")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    wheel: Path = args.wheel
    if not wheel.is_file():
        raise FileNotFoundError(wheel)

    with tempfile.TemporaryDirectory() as temp_dir:
        _run_base_compress(wheel, Path(temp_dir))
    for scenario in SCENARIOS:
        _run(wheel, scenario)


if __name__ == "__main__":
    main()
