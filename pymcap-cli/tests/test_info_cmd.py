"""Smoke tests for the ``info`` command path and QoS handling."""

from __future__ import annotations

import io
import json
import shutil
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd import info_cmd
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path


def _info_json(files: list[str], **kwargs: object) -> dict:
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.stdout", out)
        code = info_cmd.info(files, json_output=True, **kwargs)
    assert code == 0
    return json.loads(out.getvalue())


def test_info_accepts_ros2_bag_directory(tmp_path: Path, image_fixtures: dict[str, Path]) -> None:
    bag = tmp_path / "mybag"
    bag.mkdir()
    inner = bag / "mybag.mcap"
    shutil.copy(image_fixtures["image_small"], inner)

    data = _info_json([str(bag)])
    assert data["file"]["path"] == str(inner)


def test_info_qos_column_hidden_by_default(image_fixtures: dict[str, Path]) -> None:
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, color_system=None, width=200)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(info_cmd, "console", console)
        code = info_cmd.info([str(image_fixtures["image_small"])])
    assert code == 0
    assert "QoS" not in out.getvalue()
