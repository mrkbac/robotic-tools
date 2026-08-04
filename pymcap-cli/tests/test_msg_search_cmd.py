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


@pytest.mark.parametrize(
    ("error", "expected_return_code"),
    [(KeyboardInterrupt(), 0), (RuntimeError("search failed"), 1)],
)
def test_msg_search_handles_search_errors(
    error: BaseException,
    expected_return_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs) -> None:
        raise error

    monkeypatch.setattr(msg_search_cmd, "search_message_definitions", fail)

    assert msg_search_cmd.msg_search("pointcloud2") == expected_return_code


def test_msg_search_reports_definition_that_disappears(
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
    monkeypatch.setattr(msg_search_cmd, "get_message_definition", lambda *_args, **_kwargs: None)

    assert msg_search_cmd.msg_search("pointcloud2", show_definition=True) == 1
    assert "could not resolve" in capsys.readouterr().err


def test_msg_search_prints_headers_between_multiple_definitions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        MessageSearchResult("example_msgs/msg/First", "uint8 first\n", None),
        MessageSearchResult("example_msgs/msg/Second", "uint8 second\n", None),
    ]
    monkeypatch.setattr(
        msg_search_cmd, "search_message_definitions", lambda *_args, **_kwargs: results
    )
    monkeypatch.setattr(
        msg_search_cmd,
        "get_message_definition",
        lambda msg_type, **_kwargs: f"uint8 {msg_type.rsplit('/', 1)[-1].lower()}\n",
    )

    assert msg_search_cmd.msg_search("example", show_definition=True) == 0
    assert capsys.readouterr().out == (
        "# example_msgs/msg/First\nuint8 first\n\n# example_msgs/msg/Second\nuint8 second\n"
    )


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
