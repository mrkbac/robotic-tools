"""Tests for `pymcap-cli msg list`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.msg import list_cmd as msg_list_cmd
from pymcap_cli.core.msg_resolver import ROS2Distro

if TYPE_CHECKING:
    from pathlib import Path


def test_msg_list_prints_one_per_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, ROS2Distro, tuple[Path, ...]]] = []

    def fake_list_package_messages(
        package_name: str,
        distro: ROS2Distro = ROS2Distro.HUMBLE,
        extra_paths: tuple[Path, ...] = (),
    ) -> list[str]:
        calls.append((package_name, distro, extra_paths))
        return ["Image", "PointCloud2", "Temperature"]

    monkeypatch.setattr(msg_list_cmd, "list_package_messages", fake_list_package_messages)

    rc = msg_list_cmd.msg_list(
        "sensor_msgs",
        distro=ROS2Distro.JAZZY,
        extra_path=[tmp_path],
    )

    assert rc == 0
    assert capsys.readouterr().out == "Image\nPointCloud2\nTemperature\n"
    assert calls == [("sensor_msgs", ROS2Distro.JAZZY, (tmp_path,))]


def test_msg_list_empty_package_succeeds_with_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        msg_list_cmd,
        "list_package_messages",
        lambda *_args, **_kwargs: [],
    )

    rc = msg_list_cmd.msg_list("rclpy")

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err == ""


def test_msg_list_returns_one_when_package_unknown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        msg_list_cmd,
        "list_package_messages",
        lambda *_args, **_kwargs: None,
    )

    rc = msg_list_cmd.msg_list("not_a_package_msgs")

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "could not resolve package" in captured.err
    assert "not_a_package_msgs" in captured.err


def test_msg_list_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["msg", "list", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli msg list" in output
