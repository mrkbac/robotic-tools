from contextlib import contextmanager
from pathlib import Path

import pymcap_cli.cmd._run_processor as run_processor_module
import pytest
from pymcap_cli.cmd._run_processor import (
    _open_output_stream,
    resolve_overwrite_policy,
    run_processor,
)
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from small_mcap import McapWriter


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


def test_open_output_stream_async_preserves_order_and_logical_position(tmp_path: Path):
    output = tmp_path / "output.mcap"

    with _open_output_stream(
        output,
        OverwriteCollisionPolicy.OVERWRITE,
        async_buffer_bytes=16,
    ) as stream:
        assert stream.tell() == 0
        stream.write(b"first")
        assert stream.tell() == 5
        stream.write(b"-second")
        assert stream.tell() == 12

    assert output.read_bytes() == b"first-second"


def test_run_processor_uses_configured_input_buffer(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "output.mcap"
    with source.open("wb") as stream:
        writer = McapWriter(stream)
        writer.start()
        writer.finish()

    observed: list[int] = []
    real_open_input = run_processor_module.open_input

    @contextmanager
    def recording_open_input(path: str, buffering: int = 8192):
        observed.append(buffering)
        with real_open_input(path, buffering=buffering) as opened:
            yield opened

    monkeypatch.setattr(run_processor_module, "open_input", recording_open_input)

    run_processor(
        files=[str(source)],
        output=output,
        input_options=InputOptions.from_args(
            include_metadata=False,
            include_attachments=False,
        ),
        output_options=OutputOptions(overwrite_policy=OverwriteCollisionPolicy.OVERWRITE),
        input_buffer_bytes=123_456,
    )

    assert observed == [123_456]
