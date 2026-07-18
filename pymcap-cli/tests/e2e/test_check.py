from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.e2e
def test_check_valid_spec_exits_zero(simple_mcap: Path, tmp_path: Path) -> None:
    spec = tmp_path / "recording.yaml"
    spec.write_text(
        """\
version: 1
topics:
  sample:
    topic: /test
    schema:
      name: test
      encoding: json
    message_encoding: json
"""
    )

    result = subprocess.run(
        ["pymcap-cli", "check", str(simple_mcap), "--spec", str(spec)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "sample/expected" in result.stdout
    assert re.search(r"\b0\s+ERROR\b", result.stdout)


@pytest.mark.e2e
def test_check_error_exits_one(simple_mcap: Path, tmp_path: Path) -> None:
    spec = tmp_path / "recording.yaml"
    spec.write_text(
        """\
version: 1
topics:
  imu:
    topic: /imu
"""
    )

    result = subprocess.run(
        ["pymcap-cli", "check", str(simple_mcap), "--spec", str(spec)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "expected topic is missing" in result.stdout
    assert re.search(r"\b1\s+ERROR\b", result.stdout)


@pytest.mark.e2e
def test_check_nuscenes_example(nuscenes_mcap: Path) -> None:
    if not nuscenes_mcap.exists():
        pytest.skip(f"nuScenes fixture missing: {nuscenes_mcap}")
    spec = nuscenes_mcap.parents[2] / "pymcap-cli/examples/check/nuscenes.yaml"

    result = subprocess.run(
        ["pymcap-cli", "check", str(nuscenes_mcap), "--spec", str(spec)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"\b0\s+ERROR\b", result.stdout)
