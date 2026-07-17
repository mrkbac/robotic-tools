"""Rendered-help contracts for CLI features shared across commands."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pymcap_cli.cli import app

CMD_DIR = Path(__file__).parents[1] / "src" / "pymcap_cli" / "cmd"


def _parameter_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "Parameter":
            continue
        for keyword in node.keywords:
            if keyword.arg != "name":
                continue
            values = keyword.value.elts if isinstance(keyword.value, ast.List) else [keyword.value]
            names.update(
                value.value
                for value in values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            )
    return names


def _help(capsys: pytest.CaptureFixture[str], *command: str) -> str:
    with pytest.raises(SystemExit) as exc_info:
        app([*command, "--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    return captured.out + captured.err


@pytest.mark.parametrize("command", ["cat", "record", "delay", "diag"])
def test_bridge_topic_filters_use_canonical_names(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(capsys, "bridge", command)

    assert "--topic " in output
    assert "--topics" not in output


def test_bridge_cat_uses_file_cat_filter_and_render_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    file_help = _help(capsys, "cat")
    bridge_help = _help(capsys, "bridge", "cat")

    for option in (
        "--topic",
        "--exclude-topic",
        "--query",
        "--grep",
        "--grep-ignore-case",
        "--limit",
        "--bytes",
        "--flat",
        "--changed",
    ):
        assert option in file_help
        assert option in bridge_help


def test_bridge_record_uses_canonical_limit_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, "bridge", "record")

    assert "--limit" in output
    assert "--message-limit" not in output


def test_bridge_proxy_uses_roscompress_option_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    roscompress_help = _help(capsys, "roscompress")
    proxy_help = _help(capsys, "bridge", "proxy")

    for option in (
        "--codec",
        "--quality",
        "--encoder",
        "--scale",
        "--backend",
        "--image-format",
        "--jpeg-quality",
        "--pointcloud",
        "--resolution",
        "--pc-format",
        "--pc-schema",
        "--pc-encoding",
        "--pc-compression",
        "--draco-compression-level",
        "--pointcloud-drop-invalid",
        "--pointcloud-sort-field",
    ):
        assert option in roscompress_help
        assert option in proxy_help

    for bridge_only_name in (
        "--image-codec",
        "--image-quality",
        "--image-encoder",
        "--image-scale",
        "--image-backend",
    ):
        assert bridge_only_name not in proxy_help


@pytest.mark.parametrize("command", ["play", "serve"])
def test_bridge_playback_exposes_ros_transform_presets(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, "bridge", command)

    assert "--transform" in output
    if command == "play":
        assert "--only-subscribed" in output
    for option in (
        "--image-format",
        "--codec",
        "--quality",
        "--encoder",
        "--backend",
        "--scale",
        "--jpeg-quality",
        "--video",
        "--video-format",
        "--pointcloud",
        "--resolution",
        "--pc-format",
        "--pc-schema",
        "--pc-encoding",
        "--pc-compression",
        "--draco-compression-level",
        "--pointcloud-drop-invalid",
        "--pointcloud-sort-field",
    ):
        assert option in output


def test_shared_options_are_declared_only_in_central_lookup() -> None:
    shared_names = {
        "--always-decode-chunk",
        "--attachments",
        "--backend",
        "--bytes",
        "--call-timeout",
        "--changed",
        "--codec",
        "--connect-timeout",
        "--db",
        "--dedup-identical",
        "--discover-seconds",
        "--distro",
        "--early-bail",
        "--end",
        "--exclude-topic",
        "--extra-path",
        "--flat",
        "--grep",
        "--grep-ignore-case",
        "--host",
        "--include-blobs",
        "--incompressible-schema-pattern",
        "--latch",
        "--latch-from-metadata",
        "--metadata",
        "--no-browser",
        "--num-workers",
        "--pointcloud-drop-invalid",
        "--pointcloud-sort-field",
        "--query",
        "--select",
        "--spec",
        "--split-at",
        "--start",
        "--var",
    }
    declarations: dict[str, list[Path]] = {name: [] for name in shared_names}

    for path in CMD_DIR.rglob("*.py"):
        if path.name == "_cli_options.py":
            continue
        for name in _parameter_names(path) & shared_names:
            declarations[name].append(path.relative_to(CMD_DIR))

    assert {name: paths for name, paths in declarations.items() if paths} == {}
    assert not list((CMD_DIR / "_options").glob("*.py"))


def test_file_and_bridge_check_share_spec_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    file_help = _help(capsys, "check")
    bridge_help = _help(capsys, "bridge", "check")

    for output in (file_help, bridge_help):
        assert "--spec" in output
        # Rich wraps help text to the console width; compare space-normalized.
        normalized = " ".join(output.replace("│", " ").split())
        assert "Version 1 YAML recording and live-system contract." in normalized
