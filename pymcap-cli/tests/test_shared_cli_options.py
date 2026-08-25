"""Rendered-help contracts for CLI features shared across commands."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pymcap_cli.cli import app

CMD_DIR = Path(__file__).parents[1] / "src" / "pymcap_cli" / "cmd"
README = Path(__file__).parents[1] / "README.md"


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


@pytest.mark.parametrize("command", ["cat", "record", "delay", "diag", "hz", "bw", "stats"])
def test_bridge_topic_filters_use_canonical_names(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(capsys, "bridge", command)

    assert "--topic " in output
    assert "--topics" not in output


@pytest.mark.parametrize("command", ["hz", "bw", "stats"])
def test_bridge_topic_monitor_commands_share_live_measurement_options(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(capsys, "bridge", command)

    for option in (
        "--topic",
        "--all",
        "--exclude-topic",
        "--window",
        "--interval",
        "--duration",
        "--json",
        "--connect-timeout",
    ):
        assert option in output


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


def test_roscompress_exposes_per_topic_compression_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, "roscompress")

    assert "--video-topic-options" in output
    assert "--pointcloud-topic-options" in output


def test_roscompress_exposes_transactional_batch_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, "roscompress")
    normalized_output = " ".join(output.split())

    for description in (
        "Process a directory recursively",
        "Directory for batch outputs",
        "Continue batch processing after a file fails",
    ):
        assert description in normalized_output
    assert "--archive" not in output
    assert "--ffmpeg-args" in output
    assert "--video-topic-ffmpeg-args" in output


@pytest.mark.parametrize(
    "command",
    [
        "compress",
        "filter",
        "merge",
        "process",
        "rechunk",
        "recover",
        "roscompress",
        "rosdecompress",
        "split",
    ],
)
def test_compressed_output_commands_share_compression_worker_configuration(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, command)

    assert "--compression-workers" in output
    assert "PYMCAP_COMPRESSION_WORKERS" in output
    assert "MCAP_COMPRESS_WORKERS" in output


@pytest.mark.parametrize("command", ["process", "roscompress"])
def test_video_compression_commands_share_decode_worker_configuration(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, command)

    assert "--video-decode-workers" in output
    assert "PYMCAP_VIDEO_DECODE_WORKERS" in output
    assert "VC_DECODE" in output


def test_bridge_target_help_documents_its_environment_variable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "PYMCAP_BRIDGE" in _help(capsys, "bridge", "play")


def test_message_path_help_documents_environment_variable_pattern(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "PYMCAP_VAR_NAME" in _help(capsys, "cat")


def test_message_definition_help_documents_ros_search_environment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "AMENT_PREFIX_PATH" in _help(capsys, "msg", "def")


def test_canonical_worker_environment_variables_parse_as_typed_options(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PYMCAP_COMPRESSION_WORKERS", "6")
    monkeypatch.setenv("PYMCAP_VIDEO_DECODE_WORKERS", "5")
    monkeypatch.setenv("MCAP_COMPRESS_WORKERS", "8")
    monkeypatch.setenv("VC_DECODE", "9")

    _command, compress_arguments, _ignored = app.parse_args(
        ["compress", "in.mcap", "-o", "out.mcap"]
    )
    _command, roscompress_arguments, _ignored = app.parse_args(
        ["roscompress", "in.mcap", "-o", "out.mcap"]
    )

    assert compress_arguments.arguments["compression_workers"] == 6
    assert roscompress_arguments.arguments["video_decode_workers"] == 5
    assert "deprecated" not in caplog.text


def test_legacy_worker_environment_variables_parse_with_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MCAP_COMPRESS_WORKERS", "6")
    monkeypatch.setenv("VC_DECODE", "5")

    _command, compress_arguments, _ignored = app.parse_args(
        ["compress", "in.mcap", "-o", "out.mcap"]
    )
    _command, roscompress_arguments, _ignored = app.parse_args(
        ["roscompress", "in.mcap", "-o", "out.mcap"]
    )

    assert compress_arguments.arguments["compression_workers"] == 6
    assert roscompress_arguments.arguments["video_decode_workers"] == 5
    assert "MCAP_COMPRESS_WORKERS is deprecated" in caplog.text
    assert "VC_DECODE is deprecated" in caplog.text


def test_worker_cli_options_override_canonical_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PYMCAP_COMPRESSION_WORKERS", "6")
    monkeypatch.setenv("PYMCAP_VIDEO_DECODE_WORKERS", "5")
    monkeypatch.setenv("MCAP_COMPRESS_WORKERS", "8")
    monkeypatch.setenv("VC_DECODE", "9")

    _command, compress_arguments, _ignored = app.parse_args(
        ["compress", "in.mcap", "-o", "out.mcap", "--compression-workers", "2"]
    )
    _command, roscompress_arguments, _ignored = app.parse_args(
        ["roscompress", "in.mcap", "-o", "out.mcap", "--video-decode-workers", "3"]
    )

    assert compress_arguments.arguments["compression_workers"] == 2
    assert roscompress_arguments.arguments["video_decode_workers"] == 3
    assert "deprecated" not in caplog.text


@pytest.mark.parametrize(
    "env_name",
    [
        "PYMCAP_BRIDGE",
        "PYMCAP_COMPRESSION_WORKERS",
        "PYMCAP_VIDEO_DECODE_WORKERS",
        "PYMCAP_VAR_<NAME>",
        "MCAP_COMPRESS_WORKERS",
        "VC_DECODE",
        "AMENT_PREFIX_PATH",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "WT_SESSION",
        "ConEmuPID",
        "ConEmuBuild",
        "TERM_PROGRAM",
    ],
)
def test_readme_documents_supported_environment(env_name: str) -> None:
    assert f"`{env_name}`" in README.read_text()


@pytest.mark.parametrize("command", ["roscompress", "rosdecompress"])
def test_ros_transforms_expose_delete_source(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "--delete-source" in _help(capsys, command)


@pytest.mark.parametrize("command", ["play", "serve"])
def test_bridge_playback_exposes_ros_transform_presets(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = _help(capsys, "bridge", command)

    assert "--preset" in output
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
        "--compression-workers",
        "--connect-timeout",
        "--db",
        "--dedup-identical",
        "--discover-seconds",
        "--distro",
        "--early-bail",
        "--end",
        "--exclude-attachments",
        "--exclude-metadata",
        "--exclude-topic",
        "--extra-path",
        "--flat",
        "--grep",
        "--grep-ignore-case",
        "--host",
        "--include-blobs",
        "--incompressible-schema-pattern",
        "--interval",
        "--latch",
        "--latch-from-metadata",
        "--metadata",
        "--no-browser",
        "--no-chunks",
        "--no-crc",
        "--num-workers",
        "--order",
        "--pointcloud-drop-invalid",
        "--pointcloud-sort-field",
        "--query",
        "--select",
        "--spec",
        "--split-at",
        "--start",
        "--var",
        "--video-decode-workers",
        "--window",
    }
    # Sanctioned overrides: commands that deliberately redeclare a shared name with different
    # semantics the central scalar alias cannot express. `bridge serve` uses a list-valued
    # `--host` so the flag may be given bare to bind every interface (vite-style).
    allowed_overrides = {"--host": {Path("bridge/serve.py")}}
    declarations: dict[str, list[Path]] = {name: [] for name in shared_names}

    for path in CMD_DIR.rglob("*.py"):
        if path.name == "_cli_options.py":
            continue
        relative = path.relative_to(CMD_DIR)
        for name in _parameter_names(path) & shared_names:
            if relative in allowed_overrides.get(name, set()):
                continue
            declarations[name].append(relative)

    assert {name: paths for name, paths in declarations.items() if paths} == {}
    assert not list((CMD_DIR / "_options").glob("*.py"))


@pytest.mark.parametrize("command", ["filter", "sort"])
def test_negative_output_flags_do_not_expose_double_negative_aliases(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _help(capsys, command)

    assert "--no-crc" in output
    assert "--no-chunks" in output
    assert "--no-no-crc" not in output
    assert "--no-no-chunks" not in output
    if command == "filter":
        assert "--no-exclude-metadata" not in output
        assert "--no-exclude-attachments" not in output


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
