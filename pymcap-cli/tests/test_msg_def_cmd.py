"""Tests for `pymcap-cli msg def`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.msg import def_cmd as msg_def_cmd
from pymcap_cli.core.msg_resolver import ROS2Distro

if TYPE_CHECKING:
    from pathlib import Path


def test_msg_def_prints_raw_definition_when_stdout_is_not_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, ROS2Distro, tuple[Path, ...] | None]] = []

    def fake_get_message_definition(
        msg_type: str,
        distro: ROS2Distro = ROS2Distro.HUMBLE,
        extra_paths: tuple[Path, ...] | None = None,
    ) -> str:
        calls.append((msg_type, distro, extra_paths))
        return "uint32 height\n"

    monkeypatch.setattr(msg_def_cmd, "get_message_definition", fake_get_message_definition)

    rc = msg_def_cmd.msg_def(
        "sensor_msgs/msg/Image",
        distro=ROS2Distro.JAZZY,
        extra_path=[tmp_path],
    )

    assert rc == 0
    assert capsys.readouterr().out == "uint32 height\n"
    assert calls == [("sensor_msgs/msg/Image", ROS2Distro.JAZZY, (tmp_path,))]


def test_msg_def_uses_colored_rendering_for_tty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_message_definition(
        msg_type: str,
        distro: ROS2Distro = ROS2Distro.HUMBLE,
        extra_paths: tuple[Path, ...] | None = None,
    ) -> str:
        _ = (msg_type, distro, extra_paths)
        return "uint32 height\n"

    monkeypatch.setattr(msg_def_cmd, "get_message_definition", fake_get_message_definition)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    rc = msg_def_cmd.msg_def("sensor_msgs/msg/Image")

    out = capsys.readouterr().out
    assert rc == 0
    assert "\x1b[" in out
    assert "uint32" in out
    assert "height" in out


def test_msg_def_compact_strips_comments_constants_and_blank_lines(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    definition = (
        "# root comment\n"
        'string label "keep # and ="  # inline comment\n'
        "uint8 FLAG=1\n"
        "\n"
        "========================================\n"
        "MSG: example_msgs/Dependency\n"
        "# dependency comment\n"
        "int32 value  # dependency inline\n"
        "uint8 OTHER=2\n"
    )
    monkeypatch.setattr(msg_def_cmd, "get_message_definition", lambda *_args, **_kwargs: definition)

    rc = msg_def_cmd.msg_def("example_msgs/Root", compact=True)

    assert rc == 0
    assert capsys.readouterr().out == (
        'string label "keep # and ="\n'
        "========================================\n"
        "MSG: example_msgs/Dependency\n"
        "int32 value\n"
    )


def test_msg_def_root_only_uses_direct_definition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, ROS2Distro, tuple[Path, ...]]] = []

    def fake_get_message_text(
        msg_type: str,
        distro: ROS2Distro = ROS2Distro.HUMBLE,
        extra_paths: tuple[Path, ...] = (),
    ) -> tuple[str, list[str]]:
        calls.append((msg_type, distro, extra_paths))
        return "uint8 root\n", ["example_msgs/Dependency"]

    def unexpected_recursive_lookup(*_args, **_kwargs) -> str:
        pytest.fail("root-only must not resolve recursive dependencies")

    monkeypatch.setattr(msg_def_cmd, "get_message_text", fake_get_message_text)
    monkeypatch.setattr(msg_def_cmd, "get_message_definition", unexpected_recursive_lookup)

    rc = msg_def_cmd.msg_def("example_msgs/Root", root_only=True)

    assert rc == 0
    assert capsys.readouterr().out == "uint8 root\n"
    assert calls == [("example_msgs/Root", ROS2Distro.HUMBLE, ())]


def test_msg_def_returns_one_when_definition_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_get_message_definition(
        msg_type: str,
        distro: ROS2Distro = ROS2Distro.HUMBLE,
        extra_paths: tuple[Path, ...] | None = None,
    ) -> str | None:
        _ = (msg_type, distro, extra_paths)
        return None

    monkeypatch.setattr(msg_def_cmd, "get_message_definition", fake_get_message_definition)

    rc = msg_def_cmd.msg_def("missing_msgs/msg/Thing")

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "could not resolve" in captured.err
    assert "missing_msgs/msg/Thing" in captured.err


@pytest.mark.parametrize(
    ("error", "expected_return_code"),
    [(KeyboardInterrupt(), 0), (RuntimeError("lookup failed"), 1)],
)
def test_msg_def_handles_resolution_errors(
    error: BaseException,
    expected_return_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs) -> None:
        raise error

    monkeypatch.setattr(msg_def_cmd, "get_message_definition", fail)

    assert msg_def_cmd.msg_def("example_msgs/Value") == expected_return_code


def test_msg_def_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["msg", "def", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli msg def" in output
    assert "Print a resolved ROS2" in output
    assert "--compact" in output
    assert "--root-only" in output
