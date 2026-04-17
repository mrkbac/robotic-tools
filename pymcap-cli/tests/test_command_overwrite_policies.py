from pathlib import Path
from types import SimpleNamespace

import pytest
from pymcap_cli.cmd import (
    compress_cmd,
    filter_cmd,
    merge_cmd,
    process_cmd,
    rechunk_cmd,
    recover_cmd,
)
from pymcap_cli.core.mcap_processor import OverwriteCollisionPolicy


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

    def fake_run_processor(*, files, output, input_options, output_options):
        _ = files, input_options
        assert output == Path("out.mcap")
        seen.append(output_options.overwrite_policy)
        return SimpleNamespace(
            stats=SimpleNamespace(
                messages_processed=0,
                writer_statistics=SimpleNamespace(message_count=0),
            ),
            processor=SimpleNamespace(
                large_channels=[],
                output_manager=SimpleNamespace(segments={0: SimpleNamespace(rechunk_groups=[])}),
            ),
        )

    monkeypatch.setattr(module, "run_processor", fake_run_processor)

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

    def fake_run_processor(*, files, output, input_options, output_options):
        _ = files, input_options
        assert output == Path("out.mcap")
        seen.append(output_options.overwrite_policy)
        return SimpleNamespace(
            stats=SimpleNamespace(
                messages_processed=0,
                writer_statistics=SimpleNamespace(message_count=0),
            ),
            processor=SimpleNamespace(
                large_channels=[],
                output_manager=SimpleNamespace(segments={0: SimpleNamespace(rechunk_groups=[])}),
            ),
        )

    monkeypatch.setattr(module, "run_processor", fake_run_processor)

    exit_code = getattr(module, func_name)(**kwargs, no_clobber=True)

    assert exit_code == 0
    assert seen == [OverwriteCollisionPolicy.ERROR]


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
def test_commands_reject_force_and_no_clobber(module, func_name: str, kwargs):
    exit_code = getattr(module, func_name)(**kwargs, force=True, no_clobber=True)

    assert exit_code == 1
