"""The file-reading commands expose one topic/time filtering convention."""

from __future__ import annotations

import pytest
from pymcap_cli.cli import _normalize_time_filter_tokens, app

COMMANDS = (
    "video",
    "plot",
    "export-csv",
    "export-json",
    "export-parquet",
    "export-pcd",
    "export-images",
    "export-geo",
    "cat",
    "filter",
    "process",
    "roscompress",
    "diag",
    "tftree",
    "tf-get",
    "tf-export",
)

CANONICAL_OPTIONS = (
    "--topic",
    "--exclude-topic",
    "--start",
    "--end",
)

REMOVED_OPTIONS = (
    "--topics",
    "--exclude-topics",
    "--start-secs",
    "--end-secs",
    "--start-nsecs",
    "--end-nsecs",
    "--include-topic-regex",
    "--exclude-topic-regex",
    "--topic-glob",
    "--exclude-topic-glob",
)


def test_short_time_options_with_equals_are_normalized() -> None:
    assert _normalize_time_filter_tokens(("cat", "recording.mcap", "-S=+10s", "-E=-30s")) == (
        "cat",
        "recording.mcap",
        "--start=+10s",
        "--end=-30s",
    )


@pytest.mark.parametrize("command", COMMANDS)
def test_file_reader_help_uses_canonical_filter_options(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app([command, "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    for option in CANONICAL_OPTIONS:
        assert option in output
    for option in REMOVED_OPTIONS:
        assert option not in output
