from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.roscompress_cmd import roscompress

if TYPE_CHECKING:
    from pathlib import Path


def test_roscompress_defaults_to_auto_backend() -> None:
    backend = inspect.signature(roscompress).parameters["backend"]

    assert backend.default == "auto"


def test_roscompress_accepts_explicit_auto_backend(simple_mcap: Path, tmp_path: Path) -> None:
    output = tmp_path / "compressed.mcap"

    result = roscompress(
        str(simple_mcap),
        output,
        force=True,
        backend="auto",
        pointcloud=False,
    )

    assert result == 0


def test_roscompress_cli_accepts_dash_prefixed_ffmpeg_arguments(
    simple_mcap: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "compressed.mcap"

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "roscompress",
                str(simple_mcap),
                "-o",
                str(output),
                "--force",
                "--no-pointcloud",
                "--backend",
                "ffmpeg-cli",
                "--ffmpeg-args=-preset medium",
            ]
        )

    assert exc_info.value.code == 0
    assert output.exists()


def test_roscompress_cli_renders_per_topic_value_validation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = "/lidar/points:mode=invalid"

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "roscompress",
                "input.mcap",
                "output.mcap",
                "--pointcloud-topic-options",
                value,
            ]
        )

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 1
    assert "Invalid value" in output
    assert value in output
    assert "mode must be one of: default, keep" in output
    assert "does not exist" not in output
