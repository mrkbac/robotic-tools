"""Tests for pymcap_cli.core.input_handler path resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.core.input_handler import resolve_mcap_path

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
