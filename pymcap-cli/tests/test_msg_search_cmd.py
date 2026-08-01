"""Tests for `pymcap-cli msg search`."""

from __future__ import annotations

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.msg import search_cmd as msg_search_cmd
from pymcap_cli.core.msg_resolver import MessageSearchResult


def test_msg_search_prints_matching_types(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        msg_search_cmd,
        "search_message_definitions",
        lambda *_args, **_kwargs: [
            MessageSearchResult("sensor_msgs/msg/PointCloud2", "uint32 height\n", None),
            MessageSearchResult("sensor_msgs/msg/PointCloud2Modifier", "", None),
        ],
    )

    rc = msg_search_cmd.msg_search("pointcloud2")

    assert rc == 0
    assert capsys.readouterr().out == (
        "sensor_msgs/msg/PointCloud2\nsensor_msgs/msg/PointCloud2Modifier\n"
    )


def test_msg_search_can_show_resolved_definition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        msg_search_cmd,
        "search_message_definitions",
        lambda *_args, **_kwargs: [
            MessageSearchResult("sensor_msgs/msg/PointCloud2", "uint32 height\n", None),
        ],
    )
    monkeypatch.setattr(
        msg_search_cmd,
        "get_message_definition",
        lambda *_args, **_kwargs: "uint32 height\nuint32 width\n",
    )

    rc = msg_search_cmd.msg_search("pointcloud2", show_definition=True)

    assert rc == 0
    assert capsys.readouterr().out == "uint32 height\nuint32 width\n"


def test_msg_search_no_match_explains_remote_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(msg_search_cmd, "search_message_definitions", lambda *_args, **_kwargs: [])

    rc = msg_search_cmd.msg_search("pointcloud2")

    captured = capsys.readouterr()
    assert rc == 0
    assert "No message definitions matched" in captured.err
    assert "--remote" in captured.err


def test_msg_search_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["msg", "search", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli msg search" in output
    assert "--show-definition" in output
    assert "--remote" in output
