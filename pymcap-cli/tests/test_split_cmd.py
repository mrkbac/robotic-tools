import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd import split_cmd
from pymcap_cli.core.mcap_processor import OverwriteCollisionPolicy

from tests.helpers import empty_processor_result


def _fake_run_processor_multi(seen: list[OverwriteCollisionPolicy]):
    def fake_run_processor_multi(*, files: list[str], output_options, input_options=None):
        _ = input_options
        assert files == ["input.mcap"]
        seen.append(output_options.overwrite_policy)
        return empty_processor_result(segments={})

    return fake_run_processor_multi


def test_split_passes_force_as_overwrite_policy(monkeypatch):
    seen: list[OverwriteCollisionPolicy] = []

    monkeypatch.setattr(split_cmd, "run_processor_multi", _fake_run_processor_multi(seen))

    exit_code = split_cmd.split(file="input.mcap", duration="1s", force=True)

    assert exit_code == 0
    assert seen == [OverwriteCollisionPolicy.OVERWRITE]


def test_split_passes_no_clobber_as_error_policy(monkeypatch):
    seen: list[OverwriteCollisionPolicy] = []

    monkeypatch.setattr(split_cmd, "run_processor_multi", _fake_run_processor_multi(seen))

    exit_code = split_cmd.split(file="input.mcap", duration="1s", no_clobber=True)

    assert exit_code == 0
    assert seen == [OverwriteCollisionPolicy.ERROR]


def test_split_rejects_force_and_no_clobber():
    exit_code = split_cmd.split(file="input.mcap", duration="1s", force=True, no_clobber=True)

    assert exit_code == 1


@pytest.mark.parametrize(
    "tokens",
    [
        ["--hysteresis", "0s"],
        ["--hysteresis-count", "0"],
        ["--keep-trailing-context", "0s"],
        ["--keep-trailing-count", "0"],
    ],
)
def test_split_rejects_zero_expression_options_before_command_runs(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tokens: list[str],
):
    called = False

    def fake_run_processor_multi(*, files: list[str], output_options, input_options=None):
        nonlocal called
        _ = files, output_options, input_options
        called = True
        return empty_processor_result(segments={})

    monkeypatch.setattr(split_cmd, "run_processor_multi", fake_run_processor_multi)

    with pytest.raises(SystemExit) as exc_info:
        app(["split", "input.mcap", "--expression", "/state.msg", *tokens])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 1
    assert "Must be > 0" in output
    assert not called


def test_split_returns_one_when_processor_raises(monkeypatch):
    def fake_run_processor_multi(*, files: list[str], output_options, input_options=None) -> None:
        _ = files, output_options, input_options
        raise RuntimeError("boom")

    monkeypatch.setattr(split_cmd, "run_processor_multi", fake_run_processor_multi)

    exit_code = split_cmd.split(file="input.mcap", duration="1s", force=True)

    assert exit_code == 1
