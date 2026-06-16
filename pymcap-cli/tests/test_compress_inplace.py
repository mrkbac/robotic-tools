"""Tests for `compress --in-place`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd import _run_processor, compress_cmd
from pymcap_cli.cmd._run_processor import (
    finalize_replace_source,
    in_place_temp_path,
    validate_mcap_output,
)
from pymcap_cli.cmd.compress_cmd import compress
from pymcap_cli.utils import read_info

from tests.fixtures.mcap_generator import create_simple_mcap

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def simple_mcap_copy(simple_mcap: Path, tmp_path: Path) -> Path:
    """A throwaway copy of `simple_mcap` we can replace in tests."""
    dst = tmp_path / "src.mcap"
    dst.write_bytes(simple_mcap.read_bytes())
    return dst


def test_compress_in_place_replaces_source(simple_mcap_copy: Path) -> None:
    rc = compress(str(simple_mcap_copy), compression="lz4", in_place=True)

    assert rc == 0
    assert simple_mcap_copy.exists()
    assert validate_mcap_output(simple_mcap_copy)
    with simple_mcap_copy.open("rb") as f:
        info = read_info(f)
    assert all(ci.compression == "lz4" for ci in info.summary.chunk_indexes)
    assert not in_place_temp_path(simple_mcap_copy).exists()


def test_compress_in_place_rejects_output(simple_mcap_copy: Path, tmp_path: Path) -> None:
    original = simple_mcap_copy.read_bytes()

    rc = compress(str(simple_mcap_copy), tmp_path / "out.mcap", in_place=True)

    assert rc == 1
    assert simple_mcap_copy.read_bytes() == original


def test_compress_in_place_rejects_delete_source(simple_mcap_copy: Path) -> None:
    rc = compress(str(simple_mcap_copy), delete_source=True, in_place=True)
    assert rc == 1


def test_compress_in_place_rejects_url() -> None:
    rc = compress("https://example.com/foo.mcap", in_place=True)
    assert rc == 1


def test_compress_requires_output_or_in_place(simple_mcap_copy: Path) -> None:
    rc = compress(str(simple_mcap_copy))
    assert rc == 1


def test_compress_in_place_preserves_source_on_validation_failure(
    simple_mcap_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_run_processor, "validate_mcap_output", lambda _path: False)
    original = simple_mcap_copy.read_bytes()

    rc = compress(str(simple_mcap_copy), compression="lz4", in_place=True)

    assert rc == 1
    assert simple_mcap_copy.read_bytes() == original
    assert not in_place_temp_path(simple_mcap_copy).exists()


def test_finalize_replace_source_preserves_when_temp_empty(simple_mcap_copy: Path) -> None:
    """A valid-but-empty temp output must not atomically replace a non-empty source."""
    original = simple_mcap_copy.read_bytes()  # 200 messages
    tmp_output = in_place_temp_path(simple_mcap_copy)
    tmp_output.write_bytes(create_simple_mcap(num_messages=0))
    assert validate_mcap_output(tmp_output) is True  # valid, but 0 messages

    rc = finalize_replace_source(source=simple_mcap_copy, tmp_output=tmp_output)

    assert rc == 1
    assert simple_mcap_copy.read_bytes() == original
    assert not tmp_output.exists()


def test_compress_in_place_cleans_temp_on_processing_error(
    simple_mcap_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*, output: Path, **_kwargs: object) -> None:
        output.write_bytes(b"partial")
        raise RuntimeError("processing failed")

    monkeypatch.setattr(compress_cmd, "run_processor", boom)
    original = simple_mcap_copy.read_bytes()

    rc = compress(str(simple_mcap_copy), compression="lz4", in_place=True)

    assert rc == 1
    assert simple_mcap_copy.read_bytes() == original
    assert not in_place_temp_path(simple_mcap_copy).exists()
