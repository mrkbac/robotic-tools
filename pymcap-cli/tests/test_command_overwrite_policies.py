from pathlib import Path

import cyclopts
import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd import (
    compress_cmd,
    filter_cmd,
    merge_cmd,
    process_cmd,
    rechunk_cmd,
    recover_cmd,
)
from pymcap_cli.core.mcap_processor import OverwriteCollisionPolicy

from tests.helpers import empty_processor_result


def _fake_run_processor(seen: list[OverwriteCollisionPolicy]):
    def fake_run_processor(*, files, output, input_options, output_options):
        _ = files, input_options
        assert output == Path("out.mcap")
        seen.append(output_options.overwrite_policy)
        return empty_processor_result()

    return fake_run_processor


@pytest.mark.parametrize(
    ("module", "func_name", "kwargs"),
    [
        (compress_cmd, "compress", {"file": "input.mcap", "output": Path("out.mcap")}),
        (
            filter_cmd,
            "filter_cmd",
            {"file": "input.mcap", "output": Path("out.mcap")},
        ),
        (
            merge_cmd,
            "merge",
            {"files": ["a.mcap", "b.mcap"], "output": Path("out.mcap")},
        ),
        (
            process_cmd,
            "process",
            {"file": ["input.mcap"], "output": Path("out.mcap")},
        ),
        (
            recover_cmd,
            "recover",
            {"file": "input.mcap", "output": Path("out.mcap")},
        ),
        (
            rechunk_cmd,
            "rechunk",
            {"file": "input.mcap", "output": Path("out.mcap")},
        ),
    ],
)
def test_commands_pass_force_as_overwrite_policy(module, func_name: str, kwargs, monkeypatch):
    seen: list[OverwriteCollisionPolicy] = []

    monkeypatch.setattr(module, "run_processor", _fake_run_processor(seen))

    exit_code = getattr(module, func_name)(**kwargs, force=True)

    assert exit_code == 0
    assert seen == [OverwriteCollisionPolicy.OVERWRITE]


@pytest.mark.parametrize(
    ("module", "func_name", "kwargs"),
    [
        (compress_cmd, "compress", {"file": "input.mcap", "output": Path("out.mcap")}),
        (
            filter_cmd,
            "filter_cmd",
            {"file": "input.mcap", "output": Path("out.mcap")},
        ),
        (
            merge_cmd,
            "merge",
            {"files": ["a.mcap", "b.mcap"], "output": Path("out.mcap")},
        ),
        (
            process_cmd,
            "process",
            {"file": ["input.mcap"], "output": Path("out.mcap")},
        ),
        (
            recover_cmd,
            "recover",
            {"file": "input.mcap", "output": Path("out.mcap")},
        ),
        (
            rechunk_cmd,
            "rechunk",
            {"file": "input.mcap", "output": Path("out.mcap")},
        ),
    ],
)
def test_commands_pass_no_clobber_as_error_policy(module, func_name: str, kwargs, monkeypatch):
    seen: list[OverwriteCollisionPolicy] = []

    monkeypatch.setattr(module, "run_processor", _fake_run_processor(seen))

    exit_code = getattr(module, func_name)(**kwargs, no_clobber=True)

    assert exit_code == 0
    assert seen == [OverwriteCollisionPolicy.ERROR]


@pytest.mark.parametrize(
    "argv",
    [
        ["compress", "input.mcap", "-o", "out.mcap"],
        ["filter", "input.mcap", "out.mcap"],
        ["merge", "a.mcap", "b.mcap", "-o", "out.mcap"],
        ["process", "input.mcap", "-o", "out.mcap"],
        ["recover", "input.mcap", "out.mcap"],
        ["rechunk", "input.mcap", "out.mcap"],
    ],
)
def test_commands_reject_force_and_no_clobber(argv: list[str]):
    # --force / --no-clobber mutual exclusion is enforced by the shared OVERWRITE_CONSTRAINT
    # group validator at parse time, before any command body runs.
    with pytest.raises(cyclopts.ValidationError, match="Mutually exclusive"):
        app([*argv, "--force", "--no-clobber"], exit_on_error=False)


def test_rechunk_returns_one_when_processor_raises(monkeypatch):
    def fake_run_processor(*, files, output, input_options, output_options) -> None:
        _ = files, output, input_options, output_options
        raise RuntimeError("boom")

    monkeypatch.setattr(rechunk_cmd, "run_processor", fake_run_processor)

    exit_code = rechunk_cmd.rechunk(file="input.mcap", output=Path("out.mcap"), force=True)

    assert exit_code == 1


def test_filter_cmd_passes_early_bail_option(monkeypatch):
    seen: list[bool] = []

    def fake_run_processor(*, files, output, input_options, output_options):
        _ = files, output, output_options
        seen.append(input_options.is_early_bail_enabled)
        return empty_processor_result()

    monkeypatch.setattr(filter_cmd, "run_processor", fake_run_processor)

    exit_code = filter_cmd.filter_cmd(
        file="input.mcap",
        output=Path("out.mcap"),
        end="100",
        early_bail=True,
        force=True,
    )

    assert exit_code == 0
    assert seen == [True]
