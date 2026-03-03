"""Tests for parallel prefetch chunk decompression (num_workers > 0)."""

import io

import pytest
from small_mcap import McapWriter, include_topics, read_message
from small_mcap.writer import CompressionType


class NonSeekableIO(io.RawIOBase):
    """Wrapper that disables seeking, forcing the non-seeking reader path."""

    def __init__(self, data: bytes):
        self._stream = io.BytesIO(data)

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        return self._stream.readinto(b)

    def readable(self) -> bool:
        return True


def _write_test_file(
    num_messages: int = 100,
    num_channels: int = 2,
    chunk_size: int = 200,
    compression: CompressionType = CompressionType.ZSTD,
) -> io.BytesIO:
    """Write a test MCAP file with multiple chunks."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer, chunk_size=chunk_size, compression=compression)
    writer.start()
    writer.add_schema(1, "Test", "json", b'{"type": "object"}')
    for ch in range(num_channels):
        writer.add_channel(ch + 1, f"/topic_{ch}", "json", 1)
    for i in range(num_messages):
        ch = (i % num_channels) + 1
        writer.add_message(ch, i * 1000, f'{{"value": {i}}}'.encode(), i * 1000)
    writer.finish()
    buffer.seek(0)
    return buffer


def _collect_messages(stream: io.IOBase, **kwargs) -> list[tuple[int, int, bytes]]:
    """Read all messages and return (channel_id, log_time, data) tuples."""
    if isinstance(stream, io.BytesIO):
        stream.seek(0)
    return [
        (msg.channel_id, msg.log_time, bytes(msg.data))
        for _schema, _channel, msg in read_message(stream, **kwargs)
    ]


def _make_nonseekable(buf: io.BytesIO) -> io.BufferedReader:
    """Create a non-seekable stream from a BytesIO buffer."""
    buf.seek(0)
    return io.BufferedReader(NonSeekableIO(buf.read()))


class TestPrefetchOutput:
    """Verify that num_workers > 0 produces identical output to num_workers=0."""

    def test_basic_identity(self):
        """Basic: prefetch output matches sequential output."""
        buf = _write_test_file()
        sequential = _collect_messages(buf, num_workers=0)
        prefetch = _collect_messages(buf, num_workers=2)
        assert sequential == prefetch

    def test_with_time_filtering(self):
        """Time filtering works correctly with prefetch."""
        buf = _write_test_file(num_messages=50)
        kwargs = {"start_time_ns": 10_000, "end_time_ns": 30_000}
        sequential = _collect_messages(buf, num_workers=0, **kwargs)
        prefetch = _collect_messages(buf, num_workers=2, **kwargs)
        assert sequential == prefetch
        assert len(sequential) > 0

    def test_with_topic_filtering(self):
        """Topic filtering works correctly with prefetch."""
        buf = _write_test_file(num_messages=50, num_channels=3)
        kwargs = {"should_include": include_topics("/topic_0")}
        sequential = _collect_messages(buf, num_workers=0, **kwargs)
        prefetch = _collect_messages(buf, num_workers=2, **kwargs)
        assert sequential == prefetch
        assert len(sequential) > 0

    def test_reverse(self):
        """Reverse ordering works correctly with prefetch."""
        buf = _write_test_file()
        sequential = _collect_messages(buf, num_workers=0, reverse=True)
        prefetch = _collect_messages(buf, num_workers=2, reverse=True)
        assert sequential == prefetch
        # Verify actually reversed
        times = [t for _, t, _ in prefetch]
        assert times == sorted(times, reverse=True)

    def test_single_chunk(self):
        """Single chunk file works with prefetch."""
        buf = _write_test_file(num_messages=3, chunk_size=10_000)
        sequential = _collect_messages(buf, num_workers=0)
        prefetch = _collect_messages(buf, num_workers=2)
        assert sequential == prefetch

    def test_lz4_compression(self):
        """LZ4 compressed chunks work with prefetch."""
        pytest.importorskip("lz4")
        buf = _write_test_file(compression=CompressionType.LZ4)
        sequential = _collect_messages(buf, num_workers=0)
        prefetch = _collect_messages(buf, num_workers=2)
        assert sequential == prefetch

    def test_uncompressed(self):
        """Uncompressed chunks work with prefetch."""
        buf = _write_test_file(compression=CompressionType.NONE)
        sequential = _collect_messages(buf, num_workers=0)
        prefetch = _collect_messages(buf, num_workers=2)
        assert sequential == prefetch

    def test_many_workers(self):
        """More workers than chunks still works correctly."""
        buf = _write_test_file(num_messages=10, chunk_size=200)
        sequential = _collect_messages(buf, num_workers=0)
        prefetch = _collect_messages(buf, num_workers=8)
        assert sequential == prefetch


class TestPrefetchFallback:
    """Verify non-seekable streams use the non-seeking path and produce correct output."""

    def test_non_seekable_matches_seekable(self):
        """Non-seekable stream output matches seekable sequential output."""
        buf = _write_test_file(num_messages=10)

        # Get reference output from seekable path
        sequential = _collect_messages(buf, num_workers=0)

        # Non-seekable stream with num_workers (uses non-seeking prefetch path)
        ns = _make_nonseekable(buf)
        result = _collect_messages(ns, num_workers=4)
        assert result == sequential


class TestNonSeekablePrefetch:
    """Verify that num_workers > 0 on non-seekable streams produces identical output."""

    def test_basic_identity(self):
        """Non-seekable: prefetch output matches sequential output."""
        buf = _write_test_file()
        ns_seq = _make_nonseekable(buf)
        sequential = _collect_messages(ns_seq, num_workers=0)
        ns_pre = _make_nonseekable(buf)
        prefetch = _collect_messages(ns_pre, num_workers=2)
        assert sequential == prefetch

    def test_with_time_filtering(self):
        """Time filtering works correctly with non-seekable prefetch."""
        buf = _write_test_file(num_messages=50)
        kwargs = {"start_time_ns": 10_000, "end_time_ns": 30_000}
        ns_seq = _make_nonseekable(buf)
        sequential = _collect_messages(ns_seq, num_workers=0, **kwargs)
        ns_pre = _make_nonseekable(buf)
        prefetch = _collect_messages(ns_pre, num_workers=2, **kwargs)
        assert sequential == prefetch
        assert len(sequential) > 0

    def test_with_topic_filtering(self):
        """Topic filtering works correctly with non-seekable prefetch."""
        buf = _write_test_file(num_messages=50, num_channels=3)
        kwargs = {"should_include": include_topics("/topic_0")}
        ns_seq = _make_nonseekable(buf)
        sequential = _collect_messages(ns_seq, num_workers=0, **kwargs)
        ns_pre = _make_nonseekable(buf)
        prefetch = _collect_messages(ns_pre, num_workers=2, **kwargs)
        assert sequential == prefetch
        assert len(sequential) > 0

    def test_many_workers(self):
        """More workers than chunks still works correctly on non-seekable."""
        buf = _write_test_file(num_messages=10, chunk_size=200)
        ns_seq = _make_nonseekable(buf)
        sequential = _collect_messages(ns_seq, num_workers=0)
        ns_pre = _make_nonseekable(buf)
        prefetch = _collect_messages(ns_pre, num_workers=8)
        assert sequential == prefetch

    def test_single_chunk(self):
        """Single chunk file works with non-seekable prefetch."""
        buf = _write_test_file(num_messages=3, chunk_size=10_000)
        ns_seq = _make_nonseekable(buf)
        sequential = _collect_messages(ns_seq, num_workers=0)
        ns_pre = _make_nonseekable(buf)
        prefetch = _collect_messages(ns_pre, num_workers=2)
        assert sequential == prefetch

    def test_lz4_compression(self):
        """LZ4 compressed chunks work with non-seekable prefetch."""
        pytest.importorskip("lz4")
        buf = _write_test_file(compression=CompressionType.LZ4)
        ns_seq = _make_nonseekable(buf)
        sequential = _collect_messages(ns_seq, num_workers=0)
        ns_pre = _make_nonseekable(buf)
        prefetch = _collect_messages(ns_pre, num_workers=2)
        assert sequential == prefetch

    def test_uncompressed(self):
        """Uncompressed chunks work with non-seekable prefetch."""
        buf = _write_test_file(compression=CompressionType.NONE)
        ns_seq = _make_nonseekable(buf)
        sequential = _collect_messages(ns_seq, num_workers=0)
        ns_pre = _make_nonseekable(buf)
        prefetch = _collect_messages(ns_pre, num_workers=2)
        assert sequential == prefetch
