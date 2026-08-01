"""Smoke tests for the ``info`` command path and QoS handling."""

from __future__ import annotations

import io
import json
import shutil
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd import info_cmd
from rich.console import Console
from small_mcap import McapWriter, stream_reader
from small_mcap.records import OPCODE_AND_LEN_STRUCT, LazyChunk
from small_mcap.writer import CompressionType

if TYPE_CHECKING:
    from pathlib import Path


def _info_json(files: list[str], **kwargs: object) -> dict:
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.stdout", out)
        code = info_cmd.info(files, json_output=True, **kwargs)
    assert code == 0
    return json.loads(out.getvalue())


def test_info_accepts_ros2_bag_directory(tmp_path: Path, image_fixtures: dict[str, Path]) -> None:
    bag = tmp_path / "mybag"
    bag.mkdir()
    inner = bag / "mybag.mcap"
    shutil.copy(image_fixtures["image_small"], inner)

    data = _info_json([str(bag)])
    assert data["file"]["path"] == str(inner)


def test_info_qos_column_hidden_by_default(image_fixtures: dict[str, Path]) -> None:
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, color_system=None, width=200)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(info_cmd, "console", console)
        code = info_cmd.info([str(image_fixtures["image_small"])])
    assert code == 0
    assert "QoS" not in out.getvalue()


def test_info_rebuild_returns_partial_summary_for_truncated_final_chunk(tmp_path: Path) -> None:
    path = tmp_path / "truncated.mcap"
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=160, compression=CompressionType.NONE)
        writer.start(profile="test", library="test")
        writer.add_schema(1, "Test", "json", b"{}")
        writer.add_channel(1, "/test", "json", 1)
        for index in range(24):
            writer.add_message(1, index + 1, bytes([index]) * 48, index + 1)
        writer.finish()

    with path.open("rb") as stream:
        chunks = [
            record
            for record in stream_reader(stream, emit_chunks=True, lazy_chunks=True)
            if isinstance(record, LazyChunk)
        ]
    assert len(chunks) >= 2
    final_chunk = chunks[-1]
    final_chunk_data_start = (
        final_chunk.record_start
        + OPCODE_AND_LEN_STRUCT.size
        + 8
        + 8
        + 8
        + 4
        + 4
        + len(final_chunk.compression.encode())
        + 8
    )
    path.write_bytes(path.read_bytes()[: final_chunk_data_start + final_chunk.data_len // 2])

    data = _info_json([str(path)])

    assert data["statistics"]["message_count"] > 0
    assert data["statistics"]["message_count"] < 24
    assert data["channels"][0]["topic"] == "/test"
