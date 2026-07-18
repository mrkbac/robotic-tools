"""Tests for pymcap_cli.core.input_handler path resolution and debug I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.core import input_handler
from pymcap_cli.core.input_handler import open_input, resolve_mcap_path

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_mcap_path_single_split_directory(tmp_path: Path) -> None:
    bag = tmp_path / "mybag"
    bag.mkdir()
    inner = bag / "mybag_0.mcap"
    inner.write_bytes(b"\x89MCAP0\r\n")

    assert resolve_mcap_path(str(bag)) == str(inner)


def test_resolve_mcap_path_legacy_single_file_directory(tmp_path: Path) -> None:
    bag = tmp_path / "mybag"
    bag.mkdir()
    inner = bag / "mybag.mcap"
    inner.write_bytes(b"\x89MCAP0\r\n")

    assert resolve_mcap_path(str(bag)) == str(inner)


def test_resolve_mcap_path_regular_file_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "recording.mcap"
    f.write_bytes(b"\x89MCAP0\r\n")

    assert resolve_mcap_path(str(f)) == str(f)


def test_resolve_mcap_path_directory_without_mcap_raises(tmp_path: Path) -> None:
    bag = tmp_path / "mybag"
    bag.mkdir()
    (bag / "other.mcap").write_bytes(b"\x89MCAP0\r\n")

    with pytest.raises(ValueError, match="mybag"):
        resolve_mcap_path(str(bag))


def test_resolve_mcap_path_url_unchanged() -> None:
    url = "https://example.com/recording.mcap"
    assert resolve_mcap_path(url) == url


def test_resolve_mcap_path_nonexistent_unchanged(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mcap"
    assert resolve_mcap_path(str(missing)) == str(missing)


def test_open_input_no_debug_by_default(
    simple_mcap: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with open_input(str(simple_mcap)) as (stream, size):
        stream.read(16)

    assert size == simple_mcap.stat().st_size
    assert "Debug I/O Statistics" not in capsys.readouterr().out


def test_open_input_explicit_debug_prints_stats(
    simple_mcap: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with open_input(str(simple_mcap), debug=True) as (stream, _size):
        stream.read(16)

    assert "Debug I/O Statistics" in capsys.readouterr().out


def test_open_input_debug_io_default_prints_stats(
    simple_mcap: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(input_handler._input_config, "is_debug_io_enabled", True)

    with open_input(str(simple_mcap)) as (stream, _size):
        stream.read(16)

    output = capsys.readouterr().out
    assert "Debug I/O Statistics" in output
    assert simple_mcap.name in "".join(output.split())


def test_cli_debug_io_flag_applies_to_any_command(
    simple_mcap: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pymcap_cli.cli import app  # noqa: PLC0415

    monkeypatch.setattr(input_handler._input_config, "is_debug_io_enabled", False)

    with pytest.raises(SystemExit) as exc_info:
        app.meta(["--debug-io", "du", str(simple_mcap)])

    assert exc_info.value.code == 0
    assert "Debug I/O Statistics" in capsys.readouterr().out
