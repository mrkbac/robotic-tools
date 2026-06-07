"""Tests for `compress --fast` / `--compression-level`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd._run_processor import validate_mcap_output
from pymcap_cli.cmd.compress_cmd import compress
from small_mcap import Channel, Message, get_summary, stream_reader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def uncompressed_mcap(simple_mcap: Path, tmp_path: Path) -> Path:
    """An uncompressed copy of simple_mcap, so compressing it to zstd takes the
    recompress path (where the zstd level actually applies, rather than a
    compression-matched fast-copy)."""
    out = tmp_path / "none.mcap"
    assert compress(str(simple_mcap), out, compression="none", force=True) == 0
    return out


def _chunk_compressions(path: Path) -> set[str]:
    with path.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    return {ci.compression for ci in summary.chunk_indexes}


def test_compress_fast_produces_valid_zstd(uncompressed_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "fast.mcap"
    rc = compress(str(uncompressed_mcap), out, compression="zstd", fast=True, force=True)
    assert rc == 0
    assert validate_mcap_output(out)
    assert _chunk_compressions(out) == {"zstd"}


def test_compress_explicit_level_produces_valid_zstd(
    uncompressed_mcap: Path, tmp_path: Path
) -> None:
    out = tmp_path / "lvl.mcap"
    rc = compress(str(uncompressed_mcap), out, compression="zstd", compression_level=-5, force=True)
    assert rc == 0
    assert validate_mcap_output(out)
    assert _chunk_compressions(out) == {"zstd"}


def test_compress_fast_and_level_together_errors(simple_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.mcap"
    rc = compress(
        str(simple_mcap), out, compression="zstd", fast=True, compression_level=3, force=True
    )
    assert rc == 1
    assert not out.exists()


def _message_counts_by_topic(path: Path) -> dict[str, int]:
    topics: dict[int, str] = {}
    counts: dict[str, int] = {}
    with path.open("rb") as f:
        for record in stream_reader(f):
            if isinstance(record, Channel):
                topics[record.id] = record.topic
            elif isinstance(record, Message):
                topic = topics[record.channel_id]
                counts[topic] = counts.get(topic, 0) + 1
    return counts


def test_compress_fast_roundtrips_messages(uncompressed_mcap: Path, tmp_path: Path) -> None:
    """--fast must preserve message content, only changing the zstd level."""
    out = tmp_path / "fast.mcap"
    assert compress(str(uncompressed_mcap), out, compression="zstd", fast=True, force=True) == 0
    assert _message_counts_by_topic(out) == _message_counts_by_topic(uncompressed_mcap)
