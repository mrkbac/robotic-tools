"""CLI-wiring regression tests for cross-argument constraints across commands.

Parse-time group validators raise ``cyclopts.ValidationError``; value/data-conditional gates
run in the command body and exit non-zero. Both are checked before any file/network IO.
"""

import cyclopts
import pytest
from pymcap_cli.cli import app


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["cat", "x.mcap", "--grep-ignore-case"], "--grep-ignore-case requires --grep"),
        (["cat", "x.mcap", "--var", "a=1"], "--var requires --query"),
        (["export-csv", "x.mcap", "out", "--var", "a=1"], "--var requires --select"),
        (["export-json", "x.mcap", "out", "--var", "a=1"], "--var requires --select"),
        (["export-parquet", "x.mcap", "out", "--var", "a=1"], "--var requires --select"),
        (["info", "x.mcap", "--compress"], "--compress requires --json"),
        (["info", "x.mcap", "--watch", "--json"], "--watch is incompatible with --json"),
        (["info", "x.mcap", "--watch-interval", "1"], "--watch-interval requires --watch"),
        (["info", "x.mcap", "--link", "--json"], "--link is incompatible with --json"),
        (["split", "x.mcap"], "Specify at least one of"),
        (["split", "x.mcap", "--hysteresis", "2s"], "--hysteresis requires --expression"),
        (["split", "x.mcap", "--var", "a=1"], "--var requires --expression"),
        (["rechunk", "x.mcap", "o.mcap", "--max-groups", "0"], "Must be >= 1"),
        (
            ["process", "x.mcap", "-o", "o.mcap", "--var", "a=1"],
            "--var requires --split-expression",
        ),
        (
            ["roscompress", "x.mcap", "o.mcap", "--image-format", "none", "--quality", "20"],
            "--quality requires --image-format video",
        ),
        (
            ["roscompress", "x.mcap", "o.mcap", "--no-pointcloud", "--resolution", "0.05"],
            "--resolution requires --pointcloud enabled",
        ),
        (["roscompress", "x.mcap", "o.mcap", "--jpeg-quality", "200"], "Must be <= 100"),
        (
            ["rosdecompress", "x.mcap", "o.mcap", "--no-video", "--jpeg-quality", "50"],
            "--jpeg-quality requires --video enabled",
        ),
        (["bridge", "record", "t", "-o", "o.mcap"], "Specify at least one of"),
        (
            ["bridge", "record", "t", "-o", "o.mcap", "--all", "-t", "/x"],
            "Mutually exclusive",
        ),
        (["bridge", "delay", "t", "--against", "bridge"], "--against requires --topic"),
        (["bridge", "cat", "t", "--var", "a=1"], "--var requires --query"),
        (["bridge", "info", "t", "--compress"], "--compress requires --json"),
        (["bridge", "params", "t", "/foo", "--set", "/bar:=1"], "Mutually exclusive"),
        (
            ["bridge", "proxy", "t", "--image-format", "none", "--codec", "h265"],
            "--codec requires --image-format video",
        ),
        (["index", "query", "--at", "5", "--since", "3"], "--at is incompatible with --since"),
    ],
)
def test_parse_time_constraint_rejected(argv: list[str], match: str):
    with pytest.raises(cyclopts.ValidationError, match=match):
        app(argv, exit_on_error=False)


@pytest.mark.parametrize(
    "argv",
    [
        # value-conditional gates enforced in the command body (before file/network IO)
        ["tf-export", "x.mcap", "--format", "json", "--robot-name", "foo"],
        ["rechunk", "x.mcap", "o.mcap", "-p", "/foo"],
        ["bridge", "pub", "t", "/topic", "--rate", "5"],
    ],
)
def test_body_gate_exits_nonzero(argv: list[str]):
    with pytest.raises(SystemExit) as exc_info:
        app(argv, exit_on_error=False)
    assert exc_info.value.code not in (0, None)
