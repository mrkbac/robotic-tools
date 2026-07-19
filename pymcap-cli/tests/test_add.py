"""Tests for the add attachment/metadata command."""

from pathlib import Path

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd import add_cmd
from small_mcap import (
    CompressionType,
    McapRecord,
    McapWriter,
    Message,
    get_summary,
    read_attachment,
    read_metadata,
)
from small_mcap.reader import stream_reader


def _build(path: Path) -> None:
    with path.open("wb") as f:
        w = McapWriter(f, compression=CompressionType.ZSTD)
        w.start(profile="", library="test")
        w.add_schema(1, "std_msgs/String", "ros2msg", b"string data")
        w.add_channel(1, "/chatter", "cdr", 1)
        w.add_message(1, 30, b"c30", 30, 0)
        w.add_message(1, 10, b"c10", 10, 0)
        w.add_message(1, 20, b"c20", 20, 0)
        w.finish()


def _stored_times(path: Path) -> list[int]:
    return [r.log_time for r in _read_all(path) if isinstance(r, Message)]


def _read_all(path: Path) -> list[McapRecord]:
    with path.open("rb") as f:
        return list(stream_reader(f))


def test_add_attachment_to_output(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    data = tmp_path / "logo.png"
    data.write_bytes(b"PNGDATA")
    _build(src)

    assert add_cmd.attachment(str(src), data=data, output=out) == 0

    with out.open("rb") as f:
        summary = get_summary(f)
        idx = next(i for i in summary.attachment_indexes if i.name == "logo.png")
        record = read_attachment(f, idx)
    assert record.data == b"PNGDATA"
    assert record.media_type == "image/png"


def test_add_attachment_cli_file_option(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    data = tmp_path / "note.txt"
    data.write_bytes(b"hello")
    _build(src)

    with pytest.raises(SystemExit) as exc_info:
        app(
            ["add", "attachment", str(src), "--file", str(data), "--output", str(out)],
            exit_on_error=False,
        )
    assert exc_info.value.code == 0

    with out.open("rb") as stream:
        summary = get_summary(stream)
    assert [index.name for index in summary.attachment_indexes] == ["note.txt"]


def test_add_attachment_preserves_message_order(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    data = tmp_path / "a.bin"
    data.write_bytes(b"x")
    _build(src)

    assert add_cmd.attachment(str(src), data=data, output=out) == 0

    # add must not reorder messages.
    assert _stored_times(out) == [30, 10, 20]


def test_add_attachment_in_place(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    data = tmp_path / "a.bin"
    data.write_bytes(b"y")
    _build(src)

    assert add_cmd.attachment(str(src), data=data, name="a.bin") == 0

    with src.open("rb") as f:
        summary = get_summary(f)
    assert [i.name for i in summary.attachment_indexes] == ["a.bin"]
    assert _stored_times(src) == [30, 10, 20]


def test_add_attachment_rejects_output_overwriting_input(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    data = tmp_path / "a.bin"
    data.write_bytes(b"x")
    _build(src)
    before = src.read_bytes()

    assert add_cmd.attachment(str(src), data=data, output=src, force=True) == 1

    assert src.read_bytes() == before


def test_add_metadata_to_output(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert add_cmd.metadata(str(src), name="run", key=["foo=bar", "n=2"], output=out) == 0

    with out.open("rb") as f:
        summary = get_summary(f)
        idx = next(i for i in summary.metadata_indexes if i.name == "run")
        record = read_metadata(f, idx)
    assert record.metadata == {"foo": "bar", "n": "2"}


def test_add_metadata_requires_a_key(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    _build(src)

    assert add_cmd.metadata(str(src), name="run", key=None, output=tmp_path / "o.mcap") == 1


def test_add_metadata_rejects_malformed_key(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    _build(src)

    assert add_cmd.metadata(str(src), name="run", key=["novalue"], output=tmp_path / "o.mcap") == 1
