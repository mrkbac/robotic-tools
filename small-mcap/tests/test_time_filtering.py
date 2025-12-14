"""Tests for time filtering in read_message() with various MCAP configurations.

These tests exercise the _read_message_non_seeking() code path which is used when:
1. The stream is not seekable, OR
2. The MCAP file has no chunk indexes (unchunked files or chunked without index)

We use different MCAP configurations to trigger various code paths:
- Unchunked files (use_chunking=False) - no chunks at all
- Chunked files without indexes (index_types=NONE) - chunks but no index in summary
"""

import io

import pytest
import small_mcap.reader as reader_module
from pytest_mock import MockerFixture
from small_mcap import Channel, McapWriter, Schema, SeekRequiredError, read_message
from small_mcap.reader import _filter_message_index_by_time
from small_mcap.records import MessageIndex
from small_mcap.writer import IndexType


def _create_unchunked_mcap_with_timed_messages(
    messages: list[tuple[int, bytes]], schema_id: int = 1, channel_id: int = 1
) -> bytes:
    """Create an unchunked MCAP with messages at specific timestamps.

    Unchunked files have no chunk indexes, triggering _read_message_non_seeking path.

    Args:
        messages: List of (timestamp_ns, data) tuples
        schema_id: Schema ID to use
        channel_id: Channel ID to use

    Returns:
        MCAP file bytes
    """
    buffer = io.BytesIO()
    writer = McapWriter(buffer, use_chunking=False)
    writer.start()

    writer.add_schema(schema_id=schema_id, name="test", encoding="raw", data=b"")
    writer.add_channel(
        channel_id=channel_id,
        topic="/test",
        message_encoding="raw",
        schema_id=schema_id,
    )

    for log_time, data in messages:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=data,
        )

    writer.finish()
    return buffer.getvalue()


def _create_unchunked_mcap_multi_channel(
    channel_messages: dict[int, list[tuple[int, bytes]]],
) -> bytes:
    """Create an unchunked MCAP with multiple channels and timed messages.

    Args:
        channel_messages: Dict of channel_id -> list of (timestamp_ns, data)

    Returns:
        MCAP file bytes
    """
    buffer = io.BytesIO()
    writer = McapWriter(buffer, use_chunking=False)
    writer.start()

    writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")

    for channel_id in channel_messages:
        writer.add_channel(
            channel_id=channel_id,
            topic=f"/channel_{channel_id}",
            message_encoding="raw",
            schema_id=1,
        )

    # Interleave messages by time
    all_messages = []
    for channel_id, msgs in channel_messages.items():
        for log_time, data in msgs:
            all_messages.append((log_time, channel_id, data))
    all_messages.sort(key=lambda x: x[0])

    for log_time, channel_id, data in all_messages:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=data,
        )

    writer.finish()
    return buffer.getvalue()


def _create_chunked_mcap_with_timed_messages(
    messages: list[tuple[int, bytes]], schema_id: int = 1, channel_id: int = 1
) -> bytes:
    """Create a chunked MCAP with messages at specific timestamps.

    Args:
        messages: List of (timestamp_ns, data) tuples
        schema_id: Schema ID to use
        channel_id: Channel ID to use

    Returns:
        MCAP file bytes
    """
    buffer = io.BytesIO()
    writer = McapWriter(buffer, use_chunking=True)
    writer.start()

    writer.add_schema(schema_id=schema_id, name="test", encoding="raw", data=b"")
    writer.add_channel(
        channel_id=channel_id,
        topic="/test",
        message_encoding="raw",
        schema_id=schema_id,
    )

    for log_time, data in messages:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=data,
        )

    writer.finish()
    return buffer.getvalue()


def _create_chunked_mcap_no_index(
    messages: list[tuple[int, bytes]],
    schema_id: int = 1,
    channel_id: int = 1,
    chunk_size: int = 64,
) -> bytes:
    """Create a chunked MCAP without chunk/message indexes in the summary.

    This triggers _read_message_non_seeking() path with chunks.

    Args:
        messages: List of (timestamp_ns, data) tuples
        schema_id: Schema ID to use
        channel_id: Channel ID to use
        chunk_size: Chunk size to force multiple chunks

    Returns:
        MCAP file bytes
    """
    buffer = io.BytesIO()
    # Use IndexType.NONE to not write any indexes in summary
    writer = McapWriter(
        buffer, use_chunking=True, chunk_size=chunk_size, index_types=IndexType.NONE
    )
    writer.start()

    writer.add_schema(schema_id=schema_id, name="test", encoding="raw", data=b"")
    writer.add_channel(
        channel_id=channel_id,
        topic="/test",
        message_encoding="raw",
        schema_id=schema_id,
    )

    for log_time, data in messages:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=data,
        )

    writer.finish()
    return buffer.getvalue()


def _create_chunked_mcap_multi_channel_no_index(
    channel_messages: dict[int, list[tuple[int, bytes]]],
    chunk_size: int = 64,
) -> bytes:
    """Create a chunked MCAP with multiple channels without chunk indexes.

    Args:
        channel_messages: Dict of channel_id -> list of (timestamp_ns, data)
        chunk_size: Chunk size to force multiple chunks

    Returns:
        MCAP file bytes
    """
    buffer = io.BytesIO()
    writer = McapWriter(
        buffer, use_chunking=True, chunk_size=chunk_size, index_types=IndexType.NONE
    )
    writer.start()

    writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")

    for channel_id in channel_messages:
        writer.add_channel(
            channel_id=channel_id,
            topic=f"/channel_{channel_id}",
            message_encoding="raw",
            schema_id=1,
        )

    # Interleave messages by time
    all_messages = []
    for channel_id, msgs in channel_messages.items():
        for log_time, data in msgs:
            all_messages.append((log_time, channel_id, data))
    all_messages.sort(key=lambda x: x[0])

    for log_time, channel_id, data in all_messages:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=data,
        )

    writer.finish()
    return buffer.getvalue()


def _create_chunked_mcap_with_message_indexes(
    messages: list[tuple[int, bytes]],
    schema_id: int = 1,
    channel_id: int = 1,
    chunk_size: int = 64,
) -> bytes:
    """Create a chunked MCAP with message indexes but no chunk indexes in summary.

    This triggers the message index filtering path in _read_message_non_seeking.
    MessageIndex records are written after each chunk, but ChunkIndex records
    are not written to the summary section.

    Args:
        messages: List of (timestamp_ns, data) tuples
        schema_id: Schema ID to use
        channel_id: Channel ID to use
        chunk_size: Chunk size to force multiple chunks

    Returns:
        MCAP file bytes
    """
    buffer = io.BytesIO()
    # MESSAGE indexes written after chunks, but no CHUNK indexes in summary
    writer = McapWriter(
        buffer,
        use_chunking=True,
        chunk_size=chunk_size,
        index_types=IndexType.MESSAGE,  # Writes MessageIndex after chunks, no ChunkIndex in summary
    )
    writer.start()

    writer.add_schema(schema_id=schema_id, name="test", encoding="raw", data=b"")
    writer.add_channel(
        channel_id=channel_id,
        topic="/test",
        message_encoding="raw",
        schema_id=schema_id,
    )

    for log_time, data in messages:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=data,
        )

    writer.finish()
    return buffer.getvalue()


def _create_chunked_mcap_multi_channel_with_message_indexes(
    channel_messages: dict[int, list[tuple[int, bytes]]],
    chunk_size: int = 64,
) -> bytes:
    """Create a chunked MCAP with multiple channels, with message indexes but no chunk indexes.

    Args:
        channel_messages: Dict of channel_id -> list of (timestamp_ns, data)
        chunk_size: Chunk size to force multiple chunks

    Returns:
        MCAP file bytes
    """
    buffer = io.BytesIO()
    writer = McapWriter(
        buffer,
        use_chunking=True,
        chunk_size=chunk_size,
        index_types=IndexType.MESSAGE,
    )
    writer.start()

    writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")

    for channel_id in channel_messages:
        writer.add_channel(
            channel_id=channel_id,
            topic=f"/channel_{channel_id}",
            message_encoding="raw",
            schema_id=1,
        )

    # Interleave messages by time
    all_messages = []
    for channel_id, msgs in channel_messages.items():
        for log_time, data in msgs:
            all_messages.append((log_time, channel_id, data))
    all_messages.sort(key=lambda x: x[0])

    for log_time, channel_id, data in all_messages:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=data,
        )

    writer.finish()
    return buffer.getvalue()


def test_read_message_with_time_range_chunked(mocker: MockerFixture):
    """Time filtering should work on chunked MCAP files (uses seeking path)."""
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
        (3_000_000, b"msg3"),
        (4_000_000, b"msg4"),
        (5_000_000, b"msg5"),
    ]
    mcap_data = _create_chunked_mcap_with_timed_messages(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, start_time_ns=2_000_000, end_time_ns=4_000_000))
    spy.assert_not_called()

    # Should get messages at 2ms and 3ms (end_time is exclusive)
    assert len(results) == 2
    assert results[0][2].log_time == 2_000_000
    assert results[1][2].log_time == 3_000_000


def test_read_message_with_time_range_unchunked(mocker: MockerFixture):
    """Time filtering on unchunked MCAP (uses _read_message_non_seeking path)."""
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
        (3_000_000, b"msg3"),
        (4_000_000, b"msg4"),
        (5_000_000, b"msg5"),
    ]
    mcap_data = _create_unchunked_mcap_with_timed_messages(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, start_time_ns=2_000_000, end_time_ns=4_000_000))
    spy.assert_called_once()

    # Should get messages at 2ms and 3ms (end_time is exclusive)
    assert len(results) == 2
    assert results[0][2].log_time == 2_000_000
    assert results[1][2].log_time == 3_000_000


def test_read_message_channel_filtering_unchunked(mocker: MockerFixture):
    """Channel filtering with should_include on unchunked MCAP."""
    channel_messages = {
        1: [(1_000_000, b"ch1_msg1"), (3_000_000, b"ch1_msg2")],
        2: [(2_000_000, b"ch2_msg1"), (4_000_000, b"ch2_msg2")],
    }
    mcap_data = _create_unchunked_mcap_multi_channel(channel_messages)
    buffer = io.BytesIO(mcap_data)

    # Only include channel 1
    def should_include(channel: Channel, _schema: Schema | None) -> bool:
        return channel.id == 1

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, should_include=should_include))
    spy.assert_called_once()

    assert len(results) == 2
    assert all(r[1].id == 1 for r in results)


def test_read_message_combined_time_and_channel_filter_unchunked(mocker: MockerFixture):
    """Combined time range and channel filtering on unchunked MCAP."""
    channel_messages = {
        1: [(1_000_000, b"ch1_t1"), (2_000_000, b"ch1_t2"), (3_000_000, b"ch1_t3")],
        2: [(1_500_000, b"ch2_t1"), (2_500_000, b"ch2_t2"), (3_500_000, b"ch2_t3")],
    }
    mcap_data = _create_unchunked_mcap_multi_channel(channel_messages)
    buffer = io.BytesIO(mcap_data)

    # Only channel 1, time range 1.5ms to 3ms
    def should_include(channel: Channel, _schema: Schema | None) -> bool:
        return channel.id == 1

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(
        read_message(
            buffer,
            should_include=should_include,
            start_time_ns=1_500_000,
            end_time_ns=3_000_000,
        )
    )
    spy.assert_called_once()

    # Should only get ch1_t2 (at 2ms)
    assert len(results) == 1
    assert results[0][1].id == 1
    assert results[0][2].log_time == 2_000_000


def test_read_message_all_messages_filtered_out_unchunked(mocker: MockerFixture):
    """All messages filtered by time should return empty (unchunked)."""
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
    ]
    mcap_data = _create_unchunked_mcap_with_timed_messages(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, start_time_ns=10_000_000, end_time_ns=20_000_000))
    spy.assert_called_once()

    assert len(results) == 0


def test_read_message_unchunked_full_read(mocker: MockerFixture):
    """Reading all messages from unchunked MCAP without filters."""
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
        (3_000_000, b"msg3"),
    ]
    mcap_data = _create_unchunked_mcap_with_timed_messages(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer))
    spy.assert_called_once()

    assert len(results) == 3
    for i, (_schema, _channel, msg) in enumerate(results):
        assert msg.log_time == (i + 1) * 1_000_000


# Tests for chunked MCAP without indexes (triggers _read_message_non_seeking with chunks)


def test_chunked_no_index_time_filter(mocker: MockerFixture):
    """Time filtering on chunked MCAP without indexes in summary.

    This exercises the chunk processing code in _read_message_non_seeking.
    """
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
        (3_000_000, b"msg3"),
        (4_000_000, b"msg4"),
        (5_000_000, b"msg5"),
    ]
    mcap_data = _create_chunked_mcap_no_index(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, start_time_ns=2_000_000, end_time_ns=4_000_000))
    spy.assert_called_once()

    assert len(results) == 2
    assert results[0][2].log_time == 2_000_000
    assert results[1][2].log_time == 3_000_000


def test_chunked_no_index_channel_filter(mocker: MockerFixture):
    """Channel filtering on chunked MCAP without indexes.

    This tests the exclude_channels logic in _read_message_non_seeking.
    """
    channel_messages = {
        1: [(1_000_000, b"ch1_msg1"), (3_000_000, b"ch1_msg2")],
        2: [(2_000_000, b"ch2_msg1"), (4_000_000, b"ch2_msg2")],
    }
    mcap_data = _create_chunked_mcap_multi_channel_no_index(channel_messages)
    buffer = io.BytesIO(mcap_data)

    def should_include(channel: Channel, _schema: Schema | None) -> bool:
        return channel.id == 1

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, should_include=should_include))
    spy.assert_called_once()

    assert len(results) == 2
    assert all(r[1].id == 1 for r in results)


def test_chunked_no_index_combined_filter(mocker: MockerFixture):
    """Combined time and channel filter on chunked MCAP without indexes."""
    channel_messages = {
        1: [(1_000_000, b"ch1_t1"), (2_000_000, b"ch1_t2"), (3_000_000, b"ch1_t3")],
        2: [(1_500_000, b"ch2_t1"), (2_500_000, b"ch2_t2"), (3_500_000, b"ch2_t3")],
    }
    mcap_data = _create_chunked_mcap_multi_channel_no_index(channel_messages)
    buffer = io.BytesIO(mcap_data)

    def should_include(channel: Channel, _schema: Schema | None) -> bool:
        return channel.id == 1

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(
        read_message(
            buffer,
            should_include=should_include,
            start_time_ns=1_500_000,
            end_time_ns=3_000_000,
        )
    )
    spy.assert_called_once()

    assert len(results) == 1
    assert results[0][1].id == 1
    assert results[0][2].log_time == 2_000_000


def test_chunked_no_index_out_of_range(mocker: MockerFixture):
    """Time filter outside chunk time range should skip the chunk."""
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
    ]
    mcap_data = _create_chunked_mcap_no_index(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, start_time_ns=10_000_000, end_time_ns=20_000_000))
    spy.assert_called_once()

    assert len(results) == 0


def test_chunked_no_index_full_read(mocker: MockerFixture):
    """Reading all messages from chunked MCAP without indexes."""
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
        (3_000_000, b"msg3"),
    ]
    mcap_data = _create_chunked_mcap_no_index(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer))
    spy.assert_called_once()

    assert len(results) == 3
    for i, (_schema, _channel, msg) in enumerate(results):
        assert msg.log_time == (i + 1) * 1_000_000


# Tests for chunked MCAP with message indexes but no chunk indexes
# This triggers the message index filtering code path (lines 667-685)


def test_chunked_with_message_indexes_time_filter(mocker: MockerFixture):
    """Time filtering using message indexes in _read_message_non_seeking.

    This tests the path where pending_message_indexes is populated.
    """
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
        (3_000_000, b"msg3"),
        (4_000_000, b"msg4"),
        (5_000_000, b"msg5"),
    ]
    mcap_data = _create_chunked_mcap_with_message_indexes(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, start_time_ns=2_000_000, end_time_ns=4_000_000))
    spy.assert_called_once()

    assert len(results) == 2
    assert results[0][2].log_time == 2_000_000
    assert results[1][2].log_time == 3_000_000


def test_chunked_with_message_indexes_channel_filter(mocker: MockerFixture):
    """Channel filtering with message indexes.

    Tests the exclude_channels and seen_channels logic.
    """
    channel_messages = {
        1: [(1_000_000, b"ch1_msg1"), (3_000_000, b"ch1_msg2")],
        2: [(2_000_000, b"ch2_msg1"), (4_000_000, b"ch2_msg2")],
    }
    mcap_data = _create_chunked_mcap_multi_channel_with_message_indexes(channel_messages)
    buffer = io.BytesIO(mcap_data)

    def should_include(channel: Channel, _schema: Schema | None) -> bool:
        return channel.id == 1

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, should_include=should_include))
    spy.assert_called_once()

    assert len(results) == 2
    assert all(r[1].id == 1 for r in results)


def test_chunked_with_message_indexes_combined_filter(mocker: MockerFixture):
    """Combined time and channel filtering with message indexes."""
    channel_messages = {
        1: [(1_000_000, b"ch1_t1"), (2_000_000, b"ch1_t2"), (3_000_000, b"ch1_t3")],
        2: [(1_500_000, b"ch2_t1"), (2_500_000, b"ch2_t2"), (3_500_000, b"ch2_t3")],
    }
    mcap_data = _create_chunked_mcap_multi_channel_with_message_indexes(channel_messages)
    buffer = io.BytesIO(mcap_data)

    def should_include(channel: Channel, _schema: Schema | None) -> bool:
        return channel.id == 1

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(
        read_message(
            buffer,
            should_include=should_include,
            start_time_ns=1_500_000,
            end_time_ns=3_000_000,
        )
    )
    spy.assert_called_once()

    assert len(results) == 1
    assert results[0][1].id == 1
    assert results[0][2].log_time == 2_000_000


def test_chunked_with_message_indexes_all_excluded(mocker: MockerFixture):
    """All channels excluded should skip chunk processing.

    Tests the all_excluded path where chunk is skipped entirely.
    """
    channel_messages = {
        1: [(1_000_000, b"ch1_msg1"), (2_000_000, b"ch1_msg2")],
        2: [(1_500_000, b"ch2_msg1"), (2_500_000, b"ch2_msg2")],
    }
    mcap_data = _create_chunked_mcap_multi_channel_with_message_indexes(channel_messages)
    buffer = io.BytesIO(mcap_data)

    # Exclude all channels
    def should_include(_channel: Channel, _schema: Schema | None) -> bool:
        return False

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer, should_include=should_include))
    spy.assert_called_once()

    assert len(results) == 0


def test_chunked_with_message_indexes_full_read(mocker: MockerFixture):
    """Reading all messages using message indexes."""
    messages = [
        (1_000_000, b"msg1"),
        (2_000_000, b"msg2"),
        (3_000_000, b"msg3"),
    ]
    mcap_data = _create_chunked_mcap_with_message_indexes(messages)
    buffer = io.BytesIO(mcap_data)

    spy = mocker.spy(reader_module, "_read_message_non_seeking")
    results = list(read_message(buffer))
    spy.assert_called_once()

    assert len(results) == 3
    for i, (_schema, _channel, msg) in enumerate(results):
        assert msg.log_time == (i + 1) * 1_000_000


# =============================================================================
# Non-seeking edge cases
# =============================================================================


def test_non_seekable_reverse_raises():
    """reverse=True on non-seekable stream should raise SeekRequiredError."""
    mcap_data = _create_unchunked_mcap_with_timed_messages([(1_000_000, b"msg")])

    class NonSeekableStream(io.BytesIO):
        def seekable(self):
            return False

    stream = NonSeekableStream(mcap_data)

    with pytest.raises(SeekRequiredError):
        list(read_message(stream, reverse=True))


def test_empty_message_index_filter():
    """_filter_message_index_by_time should handle empty records."""
    mi = MessageIndex(channel_id=1, records=[])
    result = _filter_message_index_by_time(mi, 0, 1_000_000)

    assert result is mi  # Same object returned for empty
