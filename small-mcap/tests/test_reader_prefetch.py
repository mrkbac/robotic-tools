"""Tests for parallel prefetch chunk decompression (num_workers > 0)."""

import io

import pytest
from small_mcap import McapWriter, include_topics, read_message
from small_mcap.writer import CompressionType


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


def _collect_messages(stream: io.BytesIO, **kwargs) -> list[tuple[int, int, bytes]]:
    """Read all messages and return (channel_id, log_time, data) tuples."""
    stream.seek(0)
    return [
        (msg.channel_id, msg.log_time, bytes(msg.data))
        for _schema, _channel, msg in read_message(stream, **kwargs)
    ]


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
    """Verify fallback to sequential for non-seekable streams."""

    def test_non_seekable_ignores_num_workers(self):
        """Non-seekable streams should ignore num_workers and work sequentially."""
        buf = _write_test_file(num_messages=10)

        # Get reference output
        buf.seek(0)
        sequential = _collect_messages(buf, num_workers=0)

        # Create a non-seekable stream by wrapping BytesIO
        buf.seek(0)
        raw = buf.read()
        stream = io.BufferedReader(io.BytesIO(raw))
        # Patch seekable to return False
        stream.seekable = lambda: False  # type: ignore[assignment]
        result = [
            (msg.channel_id, msg.log_time, bytes(msg.data))
            for _schema, _channel, msg in read_message(stream, num_workers=4)
        ]
        assert result == sequential
