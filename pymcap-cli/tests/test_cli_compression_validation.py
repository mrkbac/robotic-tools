"""CLI parser validation for shared compression options."""

from __future__ import annotations

import pytest
from pymcap_cli.cli import app


@pytest.mark.parametrize(
    "tokens",
    [
        ["bag2mcap", "input.bag", "-o", "out.mcap"],
        ["bridge", "record", "localhost:8765", "-o", "out.mcap"],
        ["compress", "input.mcap", "-o", "out.mcap"],
        ["convert", "input.db3", "-o", "out.mcap"],
        ["filter", "input.mcap", "-o", "out.mcap"],
        ["merge", "left.mcap", "right.mcap", "-o", "out.mcap"],
        ["process", "input.mcap", "-o", "out.mcap"],
        ["recover", "input.mcap", "-o", "out.mcap"],
        ["rechunk", "input.mcap", "-o", "out.mcap"],
        ["split", "input.mcap", "--duration", "1s"],
    ],
)
def test_compression_option_rejects_invalid_values_before_command_runs(
    tokens: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app([*tokens, "--compression", "brotli"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 1
    assert "Invalid value" in output
    assert "--compression" in output
    assert "brotli" in output
    for choice in ("zstd", "lz4", "none"):
        assert choice in output
    assert "does not exist" not in output
