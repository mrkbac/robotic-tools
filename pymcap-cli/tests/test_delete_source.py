"""Unit tests for `--delete-source` helpers in `_run_processor`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pymcap_cli.cmd import _run_processor
from pymcap_cli.cmd._run_processor import (
    delete_source_files,
    finalize_delete_source,
    validate_mcap_output,
)
from pymcap_cli.cmd.compress_cmd import compress
from pymcap_cli.cmd.merge_cmd import merge


def test_validate_mcap_output_returns_true_for_valid_file(simple_mcap: Path) -> None:
    assert validate_mcap_output(simple_mcap) is True


def test_validate_mcap_output_returns_false_for_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.mcap"
    bad.write_bytes(b"not an mcap file at all")
    assert validate_mcap_output(bad) is False


def test_validate_mcap_output_returns_false_for_truncated(truncated_mcap: Path) -> None:
    assert validate_mcap_output(truncated_mcap) is False


def test_validate_mcap_output_returns_false_for_missing(tmp_path: Path) -> None:
    assert validate_mcap_output(tmp_path / "does-not-exist.mcap") is False


def test_delete_source_files_removes_local_paths(tmp_path: Path) -> None:
    src1 = tmp_path / "a.mcap"
    src2 = tmp_path / "b.mcap"
    src1.write_bytes(b"x")
    src2.write_bytes(b"y")
    out = tmp_path / "out.mcap"
    out.write_bytes(b"z")

    delete_source_files([str(src1), str(src2)], [out])

    assert not src1.exists()
    assert not src2.exists()
    assert out.exists()


def test_delete_source_files_skips_http_urls(tmp_path: Path) -> None:
    """URL inputs must not trigger any filesystem operation."""
    out = tmp_path / "out.mcap"
    out.write_bytes(b"z")
    # Should not raise (e.g. trying to call .unlink() on a URL string).
    delete_source_files(["http://example.com/foo.mcap", "https://x/y.mcap"], [out])
    assert out.exists()


def test_delete_source_files_skips_path_equal_to_output(tmp_path: Path) -> None:
    src = tmp_path / "same.mcap"
    src.write_bytes(b"x")

    delete_source_files([str(src)], [src])

    assert src.exists()


def test_delete_source_files_skips_resolved_match(tmp_path: Path) -> None:
    """Symlink/relative-path equivalence: source resolves to output."""
    real = tmp_path / "real.mcap"
    real.write_bytes(b"x")
    link = tmp_path / "link.mcap"
    link.symlink_to(real)

    delete_source_files([str(link)], [real])

    assert real.exists()
    assert link.exists()


def test_delete_source_files_swallows_permission_denied(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "x.mcap"
    src.write_bytes(b"x")
    out = tmp_path / "out.mcap"

    def boom(_self: Path) -> None:
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "unlink", boom)

    # Must not raise — the helper logs the error and continues.
    delete_source_files([str(src)], [out])
    assert src.exists()


def test_finalize_delete_source_deletes_when_outputs_valid(
    simple_mcap: Path, tmp_path: Path
) -> None:
    src_copy = tmp_path / "copy.mcap"
    src_copy.write_bytes(simple_mcap.read_bytes())

    rc = finalize_delete_source(sources=[str(src_copy)], outputs=[simple_mcap])

    assert rc == 0
    assert not src_copy.exists()


def test_finalize_delete_source_preserves_when_output_invalid(tmp_path: Path) -> None:
    src = tmp_path / "src.mcap"
    src.write_bytes(b"x")
    bad_out = tmp_path / "bad.mcap"
    bad_out.write_bytes(b"not mcap")

    rc = finalize_delete_source(sources=[str(src)], outputs=[bad_out])

    assert rc == 1
    assert src.exists()


@pytest.fixture
def simple_mcap_copy(simple_mcap: Path, tmp_path: Path) -> Path:
    """A throwaway copy of `simple_mcap` we can delete in tests."""
    dst = tmp_path / "src.mcap"
    dst.write_bytes(simple_mcap.read_bytes())
    return dst


def test_compress_command_deletes_source_on_success(simple_mcap_copy: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.mcap"
    rc = compress(
        str(simple_mcap_copy),
        output,
        compression="zstd",
        force=True,
        delete_source=True,
    )

    assert rc == 0
    assert not simple_mcap_copy.exists()
    assert output.exists()
    assert validate_mcap_output(output)


def test_compress_command_keeps_source_on_validation_failure(
    simple_mcap_copy: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(_run_processor, "validate_mcap_output", lambda _path: False)

    output = tmp_path / "out.mcap"
    rc = compress(
        str(simple_mcap_copy),
        output,
        compression="zstd",
        force=True,
        delete_source=True,
    )

    assert rc == 1
    assert simple_mcap_copy.exists()


def test_merge_command_deletes_all_sources(simple_mcap: Path, tmp_path: Path) -> None:
    s1 = tmp_path / "s1.mcap"
    s2 = tmp_path / "s2.mcap"
    s1.write_bytes(simple_mcap.read_bytes())
    s2.write_bytes(simple_mcap.read_bytes())
    output = tmp_path / "merged.mcap"

    rc = merge([str(s1), str(s2)], output, force=True, delete_source=True)

    assert rc == 0
    assert not s1.exists()
    assert not s2.exists()
    assert validate_mcap_output(output)
