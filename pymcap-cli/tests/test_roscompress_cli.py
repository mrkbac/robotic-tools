from __future__ import annotations

import inspect
import shutil
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd import roscompress_cmd
from pymcap_cli.cmd.roscompress_cmd import roscompress
from pymcap_cli.core import batch as batch_core

from tests.fixtures.mcap_generator import create_multi_topic_mcap
from tests.helpers import mcap_message_count

if TYPE_CHECKING:
    from pathlib import Path

    from pymcap_cli.core.batch import JsonValue


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


def test_roscompress_batch_invokes_worker_without_command_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    public_calls = 0
    worker_calls = 0
    original_roscompress = roscompress_cmd.roscompress

    def unexpected_recursive_call(*_args, **_kwargs) -> int:
        nonlocal public_calls
        public_calls += 1
        return 0

    def fake_worker(*_args, **_kwargs) -> int:
        nonlocal worker_calls
        worker_calls += 1
        return 0

    def fake_run_batch(_input_dir, _output_dir, run_one, **_kwargs):
        assert run_one(input_dir / "run.mcap", tmp_path / "partial.mcap") == 0
        return batch_core.BatchRunResult([])

    monkeypatch.setattr(roscompress_cmd, "roscompress", unexpected_recursive_call)
    monkeypatch.setattr(roscompress_cmd, "_run_roscompress", fake_worker, raising=False)
    monkeypatch.setattr(roscompress_cmd, "run_batch", fake_run_batch)

    result = original_roscompress(
        str(input_dir),
        batch=True,
        output_dir=tmp_path / "output",
        image_format="none",
        pointcloud=False,
    )

    assert result == 0
    assert public_calls == 0
    assert worker_calls == 1


def test_roscompress_batch_validates_preserved_topics_with_lossy_topic_regex(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "run.mcap").write_bytes(
        create_multi_topic_mcap(["/keep", "/drop"], messages_per_topic=2)
    )
    output_dir = tmp_path / "output"

    result = roscompress(
        str(input_dir),
        None,
        batch=True,
        output_dir=output_dir,
        image_format="none",
        pointcloud=False,
        exclude_topic=["/drop"],
    )

    assert result == 0
    assert mcap_message_count(output_dir / "run.mcap") == 2


def test_roscompress_batch_recipe_does_not_depend_on_package_version(
    simple_mcap: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copyfile(simple_mcap, input_dir / "run.mcap")
    output_dir = tmp_path / "output"
    captured_recipe: list[str] = []
    original_recipe_digest = batch_core._recipe_digest

    def capture_recipe(recipe: JsonValue) -> str:
        assert isinstance(recipe, str)
        captured_recipe.append(recipe)
        return original_recipe_digest(recipe)

    monkeypatch.setattr(batch_core, "_recipe_digest", capture_recipe)
    monkeypatch.setattr(
        roscompress_cmd,
        "version",
        lambda _package: pytest.fail("package version must not affect the recipe"),
        raising=False,
    )

    result = roscompress(
        str(input_dir),
        None,
        batch=True,
        output_dir=output_dir,
        image_format="none",
        pointcloud=False,
    )

    assert result == 0
    assert "schema_version" not in captured_recipe[0]


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
