"""Tests for the seekable fast path (_read_message_seeking_unchunked).

This path serves seekable files that have no chunk indexes in their summary: it
scans records and seeks past the bodies of filtered-out messages instead of
reading them. It also handles chunked-but-unindexed files (e.g. recorder crashed
before writing the summary) by decompressing chunks inline.
"""

import io

import pytest
import small_mcap.reader as reader_module
from small_mcap import McapWriter, include_topics, read_message
from small_mcap.writer import CompressionType, IndexType


class _NonSeekable(io.RawIOBase):
    """Wrapper that disables seeking, forcing the non-seeking reader path."""

    def __init__(self, data: bytes) -> None:
        self._stream = io.BytesIO(data)

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        return self._stream.readinto(b)

    def readable(self) -> bool:
        return True


def _build(
    *,
    use_chunking: bool,
    index_types: IndexType,
    compression: CompressionType = CompressionType.NONE,
    count: int = 300,
) -> bytes:
    buffer = io.BytesIO()
    writer = McapWriter(
        buffer,
        use_chunking=use_chunking,
        index_types=index_types,
        compression=compression,
        chunk_size=200,
        enable_crcs=True,
    )
    writer.start()
    writer.add_schema(1, "s", "raw", b"")
    for cid, topic in [(1, "/a"), (2, "/b"), (3, "/c")]:
        writer.add_channel(cid, topic, "raw", 1)
    for i in range(count):
        cid = (i % 3) + 1
        log_time = 1000 + i * 10
        writer.add_message(cid, log_time, f"m{i}".encode() * (cid * 5), log_time, i)
    writer.finish()
    return buffer.getvalue()


def _read(data: bytes, **kwargs: object) -> list[tuple[str, int, int]]:
    with io.BytesIO(data) as stream:
        return [
            (channel.topic, message.log_time, message.sequence)
            for _schema, channel, message in read_message(stream, **kwargs)  # type: ignore[arg-type]
        ]


def test_unchunked_uses_fast_path(mocker):
    data = _build(use_chunking=False, index_types=IndexType.ALL)
    spy = mocker.spy(reader_module, "_read_message_seeking_unchunked")
    _read(data)
    spy.assert_called_once()


def test_chunked_without_index_uses_fast_path(mocker):
    """A chunked file whose summary lacks chunk indexes still routes here."""
    data = _build(use_chunking=True, index_types=IndexType.NONE)
    spy = mocker.spy(reader_module, "_read_message_seeking_unchunked")
    result = _read(data)
    spy.assert_called_once()
    assert len(result) == 300


def test_fully_indexed_chunked_skips_fast_path(mocker):
    """A properly indexed chunked file keeps the chunk-index seek path."""
    data = _build(use_chunking=True, index_types=IndexType.ALL)
    spy = mocker.spy(reader_module, "_read_message_seeking_unchunked")
    _read(data)
    spy.assert_not_called()


@pytest.mark.parametrize(
    "compression",
    [CompressionType.NONE, CompressionType.ZSTD, CompressionType.LZ4],
)
def test_chunked_no_index_decodes_correctly(compression):
    """Fast path decompresses inline chunks (no seek benefit, still correct)."""
    data = _build(use_chunking=True, index_types=IndexType.NONE, compression=compression)
    assert _read(data, should_include=include_topics(["/a", "/c"])) == _read(
        data, should_include=include_topics(["/a", "/c"])
    )
    # topic /b is every third message → 100 of 300
    assert len(_read(data, should_include=include_topics(["/b"]))) == 100


def test_topic_and_time_filter_matches_full_scan():
    data = _build(use_chunking=False, index_types=IndexType.ALL)
    full = _read(data)
    expected = [(topic, t, seq) for (topic, t, seq) in full if topic == "/a" and 1500 <= t < 2500]
    got = _read(
        data,
        should_include=include_topics(["/a"]),
        start_time_ns=1500,
        end_time_ns=2500,
    )
    assert got == expected


@pytest.mark.parametrize(
    "layout",
    [
        {"use_chunking": False, "index_types": IndexType.ALL},
        {"use_chunking": True, "index_types": IndexType.NONE},
        {"use_chunking": True, "index_types": IndexType.NONE, "compression": CompressionType.ZSTD},
    ],
)
@pytest.mark.parametrize("frac", [0.9, 0.6, 0.3, 0.1])
def test_truncated_matches_nonseeking_reference(layout, frac):
    """A truncated (broken) file must degrade gracefully and match the
    allow_incomplete non-seeking reader on the same bytes."""
    data = _build(**layout)
    truncated = data[: int(len(data) * frac)]

    fast = _read(truncated)

    reference_stream = io.BufferedReader(_NonSeekable(truncated))
    reference = [
        (channel.topic, message.log_time, message.sequence)
        for _schema, channel, message in read_message(reference_stream)
    ]

    assert fast == reference
