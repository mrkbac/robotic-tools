from __future__ import annotations

import inspect
import shutil
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


def test_roscompress_batch_processes_recursive_tree(
    simple_mcap: Path,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    source = input_dir / "nested" / "run.mcap"
    source.parent.mkdir(parents=True)
    shutil.copyfile(simple_mcap, source)
    output_dir = tmp_path / "output"

    result = roscompress(
        str(input_dir),
        None,
        batch=True,
        output_dir=output_dir,
        image_format="none",
        pointcloud=False,
    )

    assert result == 0
    assert (output_dir / "nested" / "run.mcap").is_file()
    assert (output_dir / ".pymcap-roscompress-archive.jsonl").is_file()


def test_roscompress_cli_accepts_batch_without_single_output(
    simple_mcap: Path,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copyfile(simple_mcap, input_dir / "run.mcap")
    output_dir = tmp_path / "output"

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "roscompress",
                str(input_dir),
                "--batch",
                "--output-dir",
                str(output_dir),
                "--image-format",
                "none",
                "--no-pointcloud",
            ]
        )

    assert exc_info.value.code == 0
    assert (output_dir / "run.mcap").is_file()


def test_roscompress_batch_rejects_single_output_and_source_deletion(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    assert (
        roscompress(
            str(input_dir),
            tmp_path / "single.mcap",
            batch=True,
            output_dir=tmp_path / "output",
        )
        == 1
    )
    assert (
        roscompress(
            str(input_dir),
            None,
            batch=True,
            output_dir=tmp_path / "output",
            delete_source=True,
        )
        == 1
    )
