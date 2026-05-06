from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.e2e
def test_doctor_valid_file_exits_zero(simple_mcap: Path) -> None:
    result = subprocess.run(
        ["pymcap-cli", "doctor", str(simple_mcap)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "passed MCAP doctor checks" in result.stdout


@pytest.mark.e2e
def test_doctor_missing_file_exits_one(tmp_path: Path) -> None:
    result = subprocess.run(
        ["pymcap-cli", "doctor", str(tmp_path / "missing.mcap")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Doctor command failed" in result.stderr
