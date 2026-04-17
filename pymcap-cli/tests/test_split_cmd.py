from types import SimpleNamespace

from pymcap_cli.cmd import split_cmd
from pymcap_cli.core.mcap_processor import OverwriteCollisionPolicy


def test_split_passes_force_as_overwrite_policy(monkeypatch):
    seen: list[OverwriteCollisionPolicy] = []

    def fake_run_processor_multi(*, files: list[str], output_options) -> SimpleNamespace:
        assert files == ["input.mcap"]
        seen.append(output_options.overwrite_policy)
        return SimpleNamespace(
            stats=SimpleNamespace(
                messages_processed=0,
                writer_statistics=SimpleNamespace(
                    message_count=0,
                    message_start_time=0,
                    message_end_time=0,
                ),
            ),
            processor=SimpleNamespace(output_manager=SimpleNamespace(segments={})),
        )

    monkeypatch.setattr(split_cmd, "run_processor_multi", fake_run_processor_multi)

    exit_code = split_cmd.split(file="input.mcap", duration="1s", force=True)

    assert exit_code == 0
    assert seen == [OverwriteCollisionPolicy.OVERWRITE]


def test_split_passes_no_clobber_as_error_policy(monkeypatch):
    seen: list[OverwriteCollisionPolicy] = []

    def fake_run_processor_multi(*, files: list[str], output_options) -> SimpleNamespace:
        assert files == ["input.mcap"]
        seen.append(output_options.overwrite_policy)
        return SimpleNamespace(
            stats=SimpleNamespace(
                messages_processed=0,
                writer_statistics=SimpleNamespace(
                    message_count=0,
                    message_start_time=0,
                    message_end_time=0,
                ),
            ),
            processor=SimpleNamespace(output_manager=SimpleNamespace(segments={})),
        )

    monkeypatch.setattr(split_cmd, "run_processor_multi", fake_run_processor_multi)

    exit_code = split_cmd.split(file="input.mcap", duration="1s", no_clobber=True)

    assert exit_code == 0
    assert seen == [OverwriteCollisionPolicy.ERROR]


def test_split_rejects_force_and_no_clobber():
    exit_code = split_cmd.split(file="input.mcap", duration="1s", force=True, no_clobber=True)

    assert exit_code == 1
