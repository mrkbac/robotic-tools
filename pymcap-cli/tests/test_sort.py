"""Tests for the sort command and filter --order."""

from pathlib import Path

from pymcap_cli.cmd import filter_cmd, sort_cmd
from small_mcap import (
    Channel,
    CompressionType,
    McapWriter,
    Message,
    get_summary,
)
from small_mcap.reader import stream_reader


def _build(path: Path, compression: CompressionType = CompressionType.ZSTD) -> None:
    with path.open("wb") as f:
        w = McapWriter(f, compression=compression)
        w.start(profile="", library="test")
        w.add_schema(1, "std_msgs/String", "ros2msg", b"string data")
        w.add_channel(1, "/chatter", "cdr", 1)
        w.add_channel(2, "/other", "cdr", 1)
        # Deliberately out of log-time order and interleaved across topics.
        w.add_message(1, 30, b"c30", 30, 0)
        w.add_message(2, 10, b"o10", 10, 0)
        w.add_message(1, 20, b"c20", 20, 0)
        w.add_message(2, 40, b"o40", 40, 0)
        w.add_attachment(5, 5, "note.txt", "text/plain", b"hi")
        w.add_metadata("cfg", {"a": "1"})
        w.finish()


def _stored(path: Path) -> list[tuple[str, int]]:
    topics: dict[int, str] = {}
    out: list[tuple[str, int]] = []
    with path.open("rb") as f:
        for record in stream_reader(f):
            if isinstance(record, Channel):
                topics[record.id] = record.topic
            elif isinstance(record, Message):
                out.append((topics[record.channel_id], record.log_time))
    return out


def _record_names(path: Path) -> tuple[list[str], list[str]]:
    with path.open("rb") as f:
        summary = get_summary(f)
    return (
        [i.name for i in summary.attachment_indexes],
        [i.name for i in summary.metadata_indexes],
    )


def test_sort_log_time_orders_messages(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert sort_cmd.sort(str(src), out, order="log_time") == 0

    assert [t for _, t in _stored(out)] == [10, 20, 30, 40]


def test_sort_topic_groups_by_topic_then_time(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert sort_cmd.sort(str(src), out, order="topic") == 0

    assert _stored(out) == [
        ("/chatter", 20),
        ("/chatter", 30),
        ("/other", 10),
        ("/other", 40),
    ]


def test_sort_preserve_keeps_stored_order(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert sort_cmd.sort(str(src), out, order="preserve") == 0

    assert [time for _, time in _stored(out)] == [30, 10, 20, 40]


def test_sort_preserves_attachments_and_metadata(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert sort_cmd.sort(str(src), out, order="log_time") == 0

    assert _record_names(out) == (["note.txt"], ["cfg"])


def test_sort_in_place(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    _build(src)

    assert sort_cmd.sort(str(src), None, order="log_time", in_place=True) == 0

    assert [t for _, t in _stored(src)] == [10, 20, 30, 40]


def test_sort_requires_output_or_in_place(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    _build(src)

    assert sort_cmd.sort(str(src), None) == 1


def test_filter_order_reorders_output(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src)

    assert filter_cmd.filter_cmd(str(src), out, order="log_time") == 0

    assert [t for _, t in _stored(out)] == [10, 20, 30, 40]


def test_filter_default_order_preserves_stored_order(tmp_path: Path) -> None:
    src = tmp_path / "in.mcap"
    out = tmp_path / "out.mcap"
    _build(src, compression=CompressionType.NONE)

    assert filter_cmd.filter_cmd(str(src), out) == 0

    # No --order: stored order is unchanged from the source.
    assert [t for _, t in _stored(out)] == [30, 10, 20, 40]
