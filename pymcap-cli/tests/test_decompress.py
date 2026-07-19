"""Tests for the decompress command."""

from pathlib import Path

from pymcap_cli.cmd import decompress_cmd
from small_mcap import Chunk, CompressionType, McapWriter, read_message
from small_mcap.reader import stream_reader


def _build(path: Path) -> None:
    with path.open("wb") as f:
        w = McapWriter(f, compression=CompressionType.ZSTD)
        w.start(profile="", library="test")
        w.add_schema(1, "std_msgs/String", "ros2msg", b"string data")
        w.add_channel(1, "/chatter", "cdr", 1)
        for i in range(5):
            w.add_message(1, i * 10, f"m{i}".encode(), i * 10, i)
        w.finish()


def _chunk_compressions(path: Path) -> list[str]:
    with path.open("rb") as f:
        return [r.compression for r in stream_reader(f, emit_chunks=True) if isinstance(r, Chunk)]


def _messages(path: Path) -> list[tuple[int, bytes]]:
    with path.open("rb") as f:
        return [(m.log_time, m.data) for _, _, m in read_message(f)]


def test_decompress_produces_uncompressed_chunks(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert decompress_cmd.decompress(str(src), out) == 0

    assert all(c == "" for c in _chunk_compressions(out))


def test_decompress_preserves_messages(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert decompress_cmd.decompress(str(src), out) == 0

    assert _messages(out) == _messages(src)


def test_decompress_requires_output_or_in_place(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    _build(src)

    assert decompress_cmd.decompress(str(src), None) == 1


def test_decompress_in_place(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    _build(src)
    before = _messages(src)

    assert decompress_cmd.decompress(str(src), None, in_place=True) == 0

    assert all(c == "" for c in _chunk_compressions(src))
    assert _messages(src) == before
