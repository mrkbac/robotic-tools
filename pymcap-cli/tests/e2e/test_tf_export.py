"""End-to-end tests for `pymcap-cli tf-export`."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import pytest

from tests.fixtures.mcap_generator import create_tf_mcap

if TYPE_CHECKING:
    from pathlib import Path


def _write_tf_mcap(tmp_path: Path) -> Path:
    path = tmp_path / "tf.mcap"
    path.write_bytes(
        create_tf_mcap(
            static_edges=[
                ("base_link", "wheel_left", (0.0, 0.3, 0.0)),
                ("base_link", "wheel_right", (0.0, -0.3, 0.0)),
                ("base_link", "camera", (0.5, 0.0, 0.2)),
            ]
        )
    )
    return path


@pytest.mark.e2e
def test_tf_export_urdf_to_file(tmp_path: Path) -> None:
    mcap = _write_tf_mcap(tmp_path)
    out = tmp_path / "robot.urdf"

    result = subprocess.run(
        ["pymcap-cli", "tf-export", str(mcap), "-o", str(out), "--robot-name", "bot"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out.exists()
    parsed = ET.parse(out).getroot()  # noqa: S314 — file we just wrote
    assert parsed.tag == "robot"
    assert parsed.attrib["name"] == "bot"


@pytest.mark.e2e
def test_tf_export_json_to_stdout(tmp_path: Path) -> None:
    mcap = _write_tf_mcap(tmp_path)

    result = subprocess.run(
        ["pymcap-cli", "tf-export", str(mcap), "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"parent": "base_link"' in result.stdout


@pytest.mark.e2e
def test_tf_export_help_lists_command() -> None:
    result = subprocess.run(
        ["pymcap-cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "tf-export" in result.stdout
