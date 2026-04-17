from pathlib import Path

import pytest
from pymcap_cli.cmd._run_processor import _open_output_stream, resolve_overwrite_policy
from pymcap_cli.core.mcap_processor import OverwriteCollisionPolicy


def test_resolve_overwrite_policy_defaults_to_ask():
    assert resolve_overwrite_policy(force=False, no_clobber=False) == OverwriteCollisionPolicy.ASK


def test_resolve_overwrite_policy_returns_overwrite_for_force():
    assert (
        resolve_overwrite_policy(force=True, no_clobber=False) == OverwriteCollisionPolicy.OVERWRITE
    )


def test_resolve_overwrite_policy_returns_error_for_no_clobber():
    assert resolve_overwrite_policy(force=False, no_clobber=True) == OverwriteCollisionPolicy.ERROR


def test_resolve_overwrite_policy_rejects_conflicting_flags():
    assert resolve_overwrite_policy(force=True, no_clobber=True) is None


def test_open_output_stream_prompts_in_ask_mode(tmp_path: Path, monkeypatch):
    output = tmp_path / "output.mcap"
    output.write_bytes(b"existing")
    seen: list[Path] = []

    def fake_confirm(path: Path, force: bool) -> None:
        seen.append(path)
        assert force is False

    monkeypatch.setattr("pymcap_cli.cmd._run_processor.confirm_output_overwrite", fake_confirm)

    with _open_output_stream(output, OverwriteCollisionPolicy.ASK) as stream:
        stream.write(b"new")

    assert seen == [output]
    assert output.read_bytes() == b"new"


def test_open_output_stream_fails_in_error_mode(tmp_path: Path):
    output = tmp_path / "output.mcap"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match=str(output)):
        _open_output_stream(output, OverwriteCollisionPolicy.ERROR)


def test_open_output_stream_overwrites_in_overwrite_mode(tmp_path: Path, monkeypatch):
    output = tmp_path / "output.mcap"
    output.write_bytes(b"existing")
    called = False

    def fake_confirm(_path: Path, _force: bool) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("pymcap_cli.cmd._run_processor.confirm_output_overwrite", fake_confirm)

    with _open_output_stream(output, OverwriteCollisionPolicy.OVERWRITE) as stream:
        stream.write(b"new")

    assert called is False
    assert output.read_bytes() == b"new"
