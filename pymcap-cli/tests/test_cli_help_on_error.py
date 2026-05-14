"""CLI parser errors include the relevant command help."""

from __future__ import annotations

import pytest
from pymcap_cli.cli import app


def test_process_missing_file_shows_command_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["process"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 1
    assert "Usage: pymcap-cli process" in output
    assert "Process MCAP files" in output
    assert 'Command "process" parameter "--file" requires an argument.' in output
