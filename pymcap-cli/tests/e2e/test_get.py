"""End-to-end tests for `pymcap-cli get attachment` and `get metadata`."""

from __future__ import annotations

import io
import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from small_mcap import CompressionType, McapWriter

if TYPE_CHECKING:
    from pathlib import Path


def _build_fixture(path):
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=1024 * 1024, compression=CompressionType.NONE)
    writer.start()
    writer.add_schema(schema_id=1, name="x", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/x", message_encoding="json", schema_id=1)
    writer.add_message(channel_id=1, log_time=0, data=b"{}", publish_time=0)
    writer.add_attachment(
        log_time=0,
        create_time=0,
        name="blob.bin",
        media_type="application/octet-stream",
        data=b"\x01\x02\x03 round-trip",
    )
    writer.add_metadata("session", {"robot": "r2", "site": "warehouse"})
    writer.finish()
    path.write_bytes(buf.getvalue())
    return path


@pytest.fixture
def extras_mcap(tmp_path: Path) -> Path:
    return _build_fixture(tmp_path / "extras.mcap")


@pytest.mark.e2e
def test_get_attachment_round_trips_bytes(extras_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.bin"
    result = subprocess.run(
        [
            "pymcap-cli",
            "get",
            "attachment",
            "--name",
            "blob.bin",
            "--output",
            str(out),
            str(extras_mcap),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == b"\x01\x02\x03 round-trip"


@pytest.mark.e2e
def test_get_metadata_prints_json(extras_mcap: Path) -> None:
    result = subprocess.run(
        ["pymcap-cli", "get", "metadata", "--name", "session", str(extras_mcap)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {"robot": "r2", "site": "warehouse"}


@pytest.mark.e2e
def test_get_metadata_missing_name_exits_one(extras_mcap: Path) -> None:
    result = subprocess.run(
        ["pymcap-cli", "get", "metadata", "--name", "nope", str(extras_mcap)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "no metadata record named" in result.stderr
