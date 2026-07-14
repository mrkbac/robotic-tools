import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd import split_cmd
from pymcap_cli.core.mcap_processor import OverwriteCollisionPolicy
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.size_split import SizeSplitProcessor

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


def test_split_requires_some_split_mode():
    exit_code = split_cmd.split(file="input.mcap", force=True)
    assert exit_code == 1


def test_split_max_size_constructs_size_processor(monkeypatch):
    seen: list[list] = []

    def fake_run_processor_multi(*, files: list[str], output_options, input_options=None):
        _ = files, input_options
        seen.append(list(output_options.routers))
        return empty_processor_result(segments={})

    monkeypatch.setattr(split_cmd, "run_processor_multi", fake_run_processor_multi)

    exit_code = split_cmd.split(file="input.mcap", max_size="500MB", force=True)

    assert exit_code == 0
    assert len(seen) == 1
    procs = seen[0]
    assert len(procs) == 1
    assert isinstance(procs[0], SizeSplitProcessor)
    assert procs[0].max_size_bytes == 500_000_000


def test_split_max_size_rejects_invalid_value(monkeypatch):
    monkeypatch.setattr(
        split_cmd,
        "run_processor_multi",
        lambda **_: empty_processor_result(segments={}),
    )
    exit_code = split_cmd.split(file="input.mcap", max_size="not-a-size", force=True)
    assert exit_code == 1


def test_split_passes_skip_values_to_expression_processor(monkeypatch):
    seen: list[ExpressionSplitProcessor] = []

    def fake_run_processor_multi(*, files: list[str], output_options, input_options=None):
        _ = files, input_options
        seen.extend(
            router
            for router in output_options.routers
            if isinstance(router, ExpressionSplitProcessor)
        )
        return empty_processor_result(segments={})

    monkeypatch.setattr(split_cmd, "run_processor_multi", fake_run_processor_multi)

    exit_code = split_cmd.split(
        file="input.mcap",
        expression="/state.direction",
        skip_value=["0", "-1"],
        output_template="drive_{value:+d}_{index:03d}.mcap",
        force=True,
    )

    assert exit_code == 0
    assert len(seen) == 1
    assert seen[0].skip_values == (0, -1)
    assert seen[0].require_value is True


def test_split_rejects_skip_value_without_expression() -> None:
    exit_code = split_cmd.split(file="input.mcap", duration="1s", skip_value=["0"])
    assert exit_code == 1


def test_split_returns_one_when_processor_raises(monkeypatch):
    def fake_run_processor_multi(*, files: list[str], output_options, input_options=None) -> None:
        _ = files, output_options, input_options
        raise RuntimeError("boom")

    monkeypatch.setattr(split_cmd, "run_processor_multi", fake_run_processor_multi)

    exit_code = split_cmd.split(file="input.mcap", duration="1s", force=True)

    assert exit_code == 1
