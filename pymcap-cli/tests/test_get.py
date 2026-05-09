"""Tests for `pymcap-cli get attachment` and `get metadata`."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.get_cmd import attachment, metadata
from small_mcap import CompressionType, McapWriter, get_summary

if TYPE_CHECKING:
    from pathlib import Path


def _build_fixture(
    path: Path,
    *,
    attachments: list[tuple[str, bytes, str]] | None = None,
    metadata_records: list[tuple[str, dict[str, str]]] | None = None,
) -> Path:
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=1024 * 1024, compression=CompressionType.NONE)
    writer.start()
    writer.add_schema(schema_id=1, name="x", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/x", message_encoding="json", schema_id=1)
    writer.add_message(channel_id=1, log_time=0, data=b"{}", publish_time=0)

    for name, data, media_type in attachments or []:
        writer.add_attachment(
            log_time=0,
            create_time=0,
            name=name,
            media_type=media_type,
            data=data,
        )

    for name, payload in metadata_records or []:
        writer.add_metadata(name, payload)

    writer.finish()
    path.write_bytes(buf.getvalue())
    return path


@pytest.fixture
def fixture_with_extras(tmp_path: Path) -> Path:
    return _build_fixture(
        tmp_path / "extras.mcap",
        attachments=[
            ("calib.bin", b"\x00\x01\x02 hello", "application/octet-stream"),
            ("notes.txt", b"first note", "text/plain"),
            ("notes.txt", b"second note", "text/plain"),
        ],
        metadata_records=[
            ("session", {"robot": "r1", "site": "yard"}),
            ("split", {"part": "1"}),
            ("split", {"part": "2", "extra": "z"}),
        ],
    )


def test_attachment_writes_bytes_to_output(fixture_with_extras: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.bin"
    rc = attachment(str(fixture_with_extras), name="calib.bin", output=out)
    assert rc == 0
    assert out.read_bytes() == b"\x00\x01\x02 hello"


def test_attachment_missing_name_returns_one(fixture_with_extras: Path) -> None:
    rc = attachment(str(fixture_with_extras), name="does-not-exist")
    assert rc == 1


def test_attachment_duplicate_without_offset_errors(
    fixture_with_extras: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = attachment(str(fixture_with_extras), name="notes.txt")
    assert rc == 1
    err = capsys.readouterr().err
    assert "multiple attachments" in err
    assert "--offset" in err


def test_attachment_duplicate_with_offset_picks_right_one(
    fixture_with_extras: Path, tmp_path: Path
) -> None:
    with fixture_with_extras.open("rb") as f:
        summary = get_summary(f)
    assert summary is not None
    notes = [idx for idx in summary.attachment_indexes if idx.name == "notes.txt"]
    assert len(notes) == 2

    out_first = tmp_path / "first.bin"
    rc = attachment(
        str(fixture_with_extras),
        name="notes.txt",
        offset=notes[0].offset,
        output=out_first,
    )
    assert rc == 0
    assert out_first.read_bytes() == b"first note"

    out_second = tmp_path / "second.bin"
    rc = attachment(
        str(fixture_with_extras),
        name="notes.txt",
        offset=notes[1].offset,
        output=out_second,
    )
    assert rc == 0
    assert out_second.read_bytes() == b"second note"


def test_attachment_offset_not_found_errors(fixture_with_extras: Path) -> None:
    rc = attachment(str(fixture_with_extras), name="calib.bin", offset=999_999)
    assert rc == 1


def test_metadata_single_match_prints_json(
    fixture_with_extras: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = metadata(str(fixture_with_extras), name="session")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"robot": "r1", "site": "yard"}


def test_metadata_duplicate_records_merge(
    fixture_with_extras: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = metadata(str(fixture_with_extras), name="split")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"part": "2", "extra": "z"}


def test_metadata_missing_name_returns_one(fixture_with_extras: Path) -> None:
    rc = metadata(str(fixture_with_extras), name="nope")
    assert rc == 1
