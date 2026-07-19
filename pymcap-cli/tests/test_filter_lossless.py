"""Tests for filter lossless-by-default behavior and the new output flags."""

from pathlib import Path

from pymcap_cli.cmd import filter_cmd
from small_mcap import Chunk, CompressionType, McapWriter, get_summary
from small_mcap.reader import stream_reader


def _build(path: Path) -> None:
    with path.open("wb") as f:
        w = McapWriter(f, compression=CompressionType.ZSTD)
        w.start(profile="", library="test")
        w.add_schema(1, "std_msgs/String", "ros2msg", b"string data")
        w.add_channel(1, "/chatter", "cdr", 1)
        for i in range(3):
            w.add_message(1, i * 10, f"m{i}".encode(), i * 10, i)
        w.add_attachment(5, 5, "note.txt", "text/plain", b"hi")
        w.add_metadata("cfg", {"a": "1"})
        w.finish()


def _names(path: Path) -> tuple[list[str], list[str]]:
    with path.open("rb") as f:
        summary = get_summary(f)
    return (
        [i.name for i in summary.attachment_indexes],
        [i.name for i in summary.metadata_indexes],
    )


def test_filter_keeps_attachments_and_metadata_by_default(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert filter_cmd.filter_cmd(str(src), out) == 0

    assert _names(out) == (["note.txt"], ["cfg"])


def test_filter_exclude_flags_drop_records(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert (
        filter_cmd.filter_cmd(str(src), out, exclude_metadata=True, exclude_attachments=True) == 0
    )

    assert _names(out) == ([], [])


def test_filter_no_chunks_writes_unchunked(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert filter_cmd.filter_cmd(str(src), out, no_chunks=True) == 0

    with out.open("rb") as f:
        chunks = [r for r in stream_reader(f, emit_chunks=True) if isinstance(r, Chunk)]
    assert chunks == []


def test_filter_no_crc_zeroes_chunk_crc(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert filter_cmd.filter_cmd(str(src), out, no_crc=True) == 0

    with out.open("rb") as f:
        chunks = [r for r in stream_reader(f, emit_chunks=True) if isinstance(r, Chunk)]
    assert chunks
    assert all(c.uncompressed_crc == 0 for c in chunks)
