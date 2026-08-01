"""Tests for `pymcap-cli msg hash`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.msg import hash_cmd as msg_hash_cmd
from pymcap_cli.core.msg_resolver import ROS2Distro

if TYPE_CHECKING:
    from pathlib import Path


def test_msg_hash_prints_rihs01(
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
        return "uint32 value\n"

    monkeypatch.setattr(msg_hash_cmd, "get_message_definition", fake_get_message_definition)
    monkeypatch.setattr(
        msg_hash_cmd,
        "compute_rihs01",
        lambda msg_type, data: f"RIHS01_{msg_type}_{data.decode().strip()}",
    )

    rc = msg_hash_cmd.msg_hash(
        "example_msgs/Value",
        distro=ROS2Distro.JAZZY,
        extra_path=[tmp_path],
    )

    assert rc == 0
    assert capsys.readouterr().out == "RIHS01_example_msgs/Value_uint32 value\n"
    assert calls == [("example_msgs/Value", ROS2Distro.JAZZY, (tmp_path,))]


def test_msg_hash_returns_one_when_definition_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(msg_hash_cmd, "get_message_definition", lambda *_args, **_kwargs: None)

    rc = msg_hash_cmd.msg_hash("missing_msgs/Thing")

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "could not resolve" in captured.err


def test_msg_hash_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["msg", "hash", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli msg hash" in output
    assert "RIHS01" in output
