"""Edge case tests for small-mcap - boundary conditions and unusual scenarios."""

import io

import pytest
import small_mcap.reader as reader_module
from pytest_mock import MockerFixture
from small_mcap import (
    IllegalOpcodeInChunkError,
    McapWriter,
    SeekRequiredError,
    get_header,
    get_summary,
    read_message,
)
from small_mcap.reader import stream_reader
from small_mcap.records import (
    MAGIC,
    Channel,
    Chunk,
    DataEnd,
    Footer,
    Header,
    LazyChunk,
    Message,
    MessageIndex,
    Schema,
)
from small_mcap.writer import CompressionType, IndexType


class TestEmptyFiles:
    """Test handling of empty or minimal MCAP files."""

    def test_empty_mcap_no_data(self, reference_mcap_files):
        """Test reading MCAP with no data records."""
        with open(reference_mcap_files["minimal"], "rb") as f:
            # Should not crash
            header = get_header(f)
            assert header is not None

            # No messages expected
            f.seek(0)
            messages = list(read_message(f))
            assert len(messages) == 0

    def test_write_empty_mcap(self, temp_mcap_file):
        """Test writing and reading MCAP with no messages."""
        # Write empty file
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start(profile="test", library="small-mcap")
            # Don't add any data
            writer.finish()

        # Read it back
        with open(temp_mcap_file, "rb") as f:
            header = get_header(f)
            assert header.profile == "test"

            messages = list(read_message(f))
            assert len(messages) == 0


class TestSchemalessMessages:
    """Test handling of messages without schemas (schema_id=0)."""

    def test_write_read_schemaless(self, temp_mcap_file):
        """Test writing and reading schemaless messages."""
        # Write
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            # Add channel with schema_id=0 (no schema)
            writer.add_channel(
                channel_id=1,
                topic="/raw/data",
                message_encoding="application/octet-stream",
                schema_id=0,  # No schema
            )

            writer.add_message(
                channel_id=1,
                log_time=1000,
                data=b"raw binary data",
                publish_time=1000,
            )

            writer.finish()

        # Read
        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f))
            assert len(messages) == 1

            _schema, channel, message = messages[0]
            # Verify with exact equality
            assert channel == Channel(
                id=1,
                schema_id=0,
                topic="/raw/data",
                message_encoding="application/octet-stream",
                metadata={},
            )
            assert message == Message(
                channel_id=1,
                sequence=0,
                log_time=1000,
                publish_time=1000,
                data=b"raw binary data",
            )


class TestOutOfOrderMessages:
    """Test handling of messages with out-of-order timestamps."""

    def test_write_read_out_of_order(self, temp_mcap_file):
        """Test writing and reading out-of-order messages."""
        # Write messages with decreasing timestamps
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=1024)
            writer.start()

            writer.add_schema(1, "Test", "json", b"{}")
            writer.add_channel(1, "/test", "json", 1)

            # Add messages in reverse time order
            for i in range(10, 0, -1):
                writer.add_message(1, i * 1000, f"msg{i}".encode(), i * 1000)

            writer.finish()

        # Read back
        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f))
            assert len(messages) == 10

            # Messages should be readable in the order they were written
            log_times = [msg[2].log_time for msg in messages]
            assert log_times == list(range(10000, 0, -1000))

    @pytest.mark.parametrize("is_seekable", [True, False])
    def test_narrow_time_filter_handles_nonmonotonic_message_index(self, is_seekable):
        buffer = io.BytesIO()
        writer = McapWriter(buffer, chunk_size=1024, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(1, "T", "raw", b"")
        writer.add_channel(1, "/t", "raw", 1)
        for timestamp in [1000, 10000, 2000]:
            writer.add_message(1, timestamp, str(timestamp).encode(), timestamp)
        writer.finish()

        stream = io.BytesIO(buffer.getvalue())
        if not is_seekable:
            stream.seekable = lambda: False
        messages = list(read_message(stream, start_time_ns=8000, end_time_ns=11000))

        assert [message.log_time for _schema, _channel, message in messages] == [10000]

    def test_reverse_orders_nonmonotonic_indexes_by_log_time(self):
        buffer = io.BytesIO()
        writer = McapWriter(buffer, chunk_size=10_000, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(1, "T", "raw", b"")
        writer.add_channel(1, "/one", "raw", 1)
        writer.add_channel(2, "/two", "raw", 1)
        for channel_id, timestamps in (
            (1, [1000, 10000, 2000]),
            (2, [1500, 9000, 2500]),
        ):
            for timestamp in timestamps:
                writer.add_message(channel_id, timestamp, b"x", timestamp)
        writer.finish()

        messages = list(read_message(io.BytesIO(buffer.getvalue()), reverse=True))

        assert [message.log_time for _schema, _channel, message in messages] == [
            10000,
            9000,
            2500,
            2000,
            1500,
            1000,
        ]

    def test_reverse_rejects_chunk_indexes_without_message_indexes(self):
        buffer = io.BytesIO()
        writer = McapWriter(
            buffer,
            chunk_size=10_000,
            compression=CompressionType.NONE,
            index_types=IndexType.CHUNK,
        )
        writer.start()
        writer.add_schema(1, "T", "raw", b"")
        writer.add_channel(1, "/test", "raw", 1)
        for timestamp in (1, 2, 3):
            writer.add_message(1, timestamp, b"x", timestamp)
        writer.finish()

        with pytest.raises(SeekRequiredError, match="requires MessageIndex"):
            list(read_message(io.BytesIO(buffer.getvalue()), reverse=True))


class TestLargeData:
    """Test handling of large messages and data."""

    def test_large_message_data(self, temp_mcap_file):
        """Test writing and reading a message with large data payload."""
        large_data = b"x" * 5_000_000  # 5MB message

        # Write
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=10_000_000)
            writer.start()

            writer.add_schema(1, "Large", "raw", b"")
            writer.add_channel(1, "/large", "raw", 1)
            writer.add_message(1, 1000, large_data, 1000)

            writer.finish()

        # Read
        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f))
            assert len(messages) == 1
            assert len(messages[0][2].data) == 5_000_000
            assert messages[0][2].data == large_data

    def test_many_small_messages(self, temp_mcap_file):
        """Test writing and reading many small messages."""
        num_messages = 10000

        # Write
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=1024)
            writer.start()

            writer.add_schema(1, "Small", "json", b"{}")
            writer.add_channel(1, "/many", "json", 1)

            for i in range(num_messages):
                writer.add_message(1, i, b"x", i)

            writer.finish()

        # Read
        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f))
            assert len(messages) == num_messages


class TestSpecialCharacters:
    """Test handling of special characters in strings."""

    def test_unicode_in_topic_names(self, temp_mcap_file):
        """Test topic names with Unicode characters."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_schema(1, "Test", "json", b"{}")
            writer.add_channel(1, "/topic/世界/🚀", "json", 1)
            writer.add_message(1, 1000, b"data", 1000)

            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f))
            assert len(messages) == 1
            _schema, channel, _message = messages[0]
            # Verify complete channel fields including Unicode topic
            assert channel.id == 1
            assert channel.schema_id == 1
            assert channel.topic == "/topic/世界/🚀"
            assert channel.message_encoding == "json"
            assert isinstance(channel.metadata, dict)

    def test_unicode_in_schema_names(self, temp_mcap_file):
        """Test schema names with Unicode characters."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_schema(1, "Schéma_тест_测试", "json", b"{}")
            writer.add_channel(1, "/test", "json", 1)
            writer.add_message(1, 1000, b"data", 1000)

            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f))
            assert len(messages) == 1
            schema, _channel, _message = messages[0]
            # Verify complete schema fields including Unicode name
            assert schema.id == 1
            assert schema.name == "Schéma_тест_测试"
            assert schema.encoding == "json"
            assert schema.data == b"{}"

    def test_empty_strings(self, temp_mcap_file):
        """Test handling of empty strings in various fields."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start(profile="", library="")

            writer.add_schema(1, "", "", b"")
            writer.add_channel(1, "/test", "", 1)
            writer.add_message(1, 1000, b"", 1000)

            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            header = get_header(f)
            assert header.profile == ""
            assert header.library == ""

            messages = list(read_message(f))
            assert len(messages) == 1


class TestMetadataAndAttachments:
    """Test edge cases with metadata and attachments."""

    def test_empty_attachment(self, temp_mcap_file):
        """Test attachment with empty data."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_attachment(
                log_time=1000,
                create_time=1000,
                name="empty.txt",
                media_type="text/plain",
                data=b"",  # Empty
            )

            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            stats = summary.statistics
            assert stats.message_count == 0
            assert stats.schema_count == 0
            assert stats.channel_count == 0
            assert stats.attachment_count == 1
            assert stats.metadata_count == 0
            assert stats.chunk_count >= 0
            assert stats.message_start_time == 0
            assert stats.message_end_time == 0
            assert stats.channel_message_counts == {}

    def test_empty_metadata(self, temp_mcap_file):
        """Test metadata with empty dict."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_metadata("test", {})  # Empty metadata

            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            stats = summary.statistics
            assert stats.message_count == 0
            assert stats.schema_count == 0
            assert stats.channel_count == 0
            assert stats.attachment_count == 0
            assert stats.metadata_count == 1
            assert stats.chunk_count >= 0
            assert stats.message_start_time == 0
            assert stats.message_end_time == 0
            assert stats.channel_message_counts == {}


class TestTimeRangeBoundaries:
    """Test time range filtering edge cases."""

    def test_messages_at_exact_boundaries(self, temp_mcap_file):
        """Test time filtering with messages at exact start/end times."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=1024)
            writer.start()

            writer.add_schema(1, "T", "json", b"{}")
            writer.add_channel(1, "/test", "json", 1)

            for i in range(10):
                writer.add_message(1, i * 1000, b"x", i * 1000)

            writer.finish()

        # Test filtering with inclusive boundaries
        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f, start_time_ns=3000, end_time_ns=6000))
            # The public interval is inclusive at start and exclusive at end.
            assert [message.log_time for _schema, _channel, message in messages] == [
                3000,
                4000,
                5000,
            ]

    def test_empty_time_range(self, temp_mcap_file):
        """Test time range with no messages."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=1024)
            writer.start()

            writer.add_schema(1, "T", "json", b"{}")
            writer.add_channel(1, "/test", "json", 1)

            writer.add_message(1, 1000, b"early", 1000)
            writer.add_message(1, 10000, b"late", 10000)

            writer.finish()

        # Query a range with no messages
        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f, start_time_ns=5000, end_time_ns=6000))
            assert len(messages) == 0


class TestNonSeekingChunkProcessing:
    """Test _read_message_non_seeking with chunked MCAP files.

    These tests exercise the non-seeking code path which is used when:
    1. The stream is not seekable, OR
    2. The MCAP file has no chunk indexes in the summary

    The tests verify chunk processing and time filtering work correctly.
    """

    def _create_chunked_mcap_no_index(
        self,
        messages: list[tuple[int, bytes]],
        chunk_size: int = 64,
    ) -> bytes:
        """Create a chunked MCAP without any indexes in the summary."""
        buffer = io.BytesIO()
        writer = McapWriter(
            buffer, use_chunking=True, chunk_size=chunk_size, index_types=IndexType.NONE
        )
        writer.start()
        writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")
        writer.add_channel(channel_id=1, topic="/test", message_encoding="raw", schema_id=1)

        for log_time, data in messages:
            writer.add_message(channel_id=1, log_time=log_time, publish_time=log_time, data=data)

        writer.finish()
        return buffer.getvalue()

    def test_single_chunk_at_eof(self, mocker: MockerFixture):
        """Single chunk at EOF (no MessageIndex) should be processed."""
        messages = [(1_000_000, b"msg1"), (2_000_000, b"msg2")]
        mcap_data = self._create_chunked_mcap_no_index(messages, chunk_size=10000)

        # Use non-seekable stream to force _read_message_non_seeking path
        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        stream = NonSeekableStream(mcap_data)
        spy = mocker.spy(reader_module, "_read_message_non_seeking")

        results = list(read_message(stream))
        spy.assert_called_once()

        # All messages should be returned including those in the final chunk
        assert len(results) == 2
        assert results[0][2].log_time == 1_000_000
        assert results[1][2].log_time == 2_000_000

    def test_multiple_chunks_final_processed(self, mocker: MockerFixture):
        """Multiple chunks with final chunk at EOF should all be processed."""
        # Create enough messages to span multiple chunks
        messages = [(i * 1_000_000, f"msg{i}".encode()) for i in range(1, 6)]
        mcap_data = self._create_chunked_mcap_no_index(messages, chunk_size=32)

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        stream = NonSeekableStream(mcap_data)
        spy = mocker.spy(reader_module, "_read_message_non_seeking")

        results = list(read_message(stream))
        spy.assert_called_once()

        # All messages from all chunks should be returned
        assert len(results) == 5
        for i, (_, _, msg) in enumerate(results):
            assert msg.log_time == (i + 1) * 1_000_000

    def test_final_chunk_outside_time_range(self, mocker: MockerFixture):
        """Final chunk outside time range should be skipped."""
        messages = [(10_000_000, b"msg1"), (20_000_000, b"msg2")]
        mcap_data = self._create_chunked_mcap_no_index(messages, chunk_size=10000)

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        stream = NonSeekableStream(mcap_data)
        spy = mocker.spy(reader_module, "_read_message_non_seeking")

        # Query time range before all messages
        results = list(read_message(stream, start_time_ns=0, end_time_ns=1_000_000))
        spy.assert_called_once()

        # No messages should be returned (final chunk skipped due to time range)
        assert len(results) == 0

    def test_final_chunk_partial_time_range(self, mocker: MockerFixture):
        """Final chunk with some messages in time range should return those messages."""
        messages = [
            (1_000_000, b"msg1"),
            (2_000_000, b"msg2"),
            (3_000_000, b"msg3"),
        ]
        mcap_data = self._create_chunked_mcap_no_index(messages, chunk_size=10000)

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        stream = NonSeekableStream(mcap_data)
        spy = mocker.spy(reader_module, "_read_message_non_seeking")

        # Query middle time range
        results = list(read_message(stream, start_time_ns=1_500_000, end_time_ns=2_500_000))
        spy.assert_called_once()

        # Only msg2 should be returned
        assert len(results) == 1
        assert results[0][2].log_time == 2_000_000


class TestLazyChunkPadding:
    """Test that LazyChunk correctly handles records with padding bytes.

    Per MCAP spec, records may be extended with new fields at the end.
    LazyChunk must seek past ALL bytes (including unknown padding) based on
    the record_length from the header, not just the fields it knows about.
    """

    def test_lazy_chunk_skips_padding_bytes(self):
        """LazyChunk should correctly position stream after padding bytes."""
        # Build MCAP manually with a Chunk that has padding
        buffer = io.BytesIO()
        buffer.write(MAGIC)

        Header(profile="test", library="test").write_record_to(buffer)
        Schema(id=1, name="test", encoding="raw", data=b"").write_record_to(buffer)
        Channel(
            id=1, schema_id=1, topic="/test", message_encoding="raw", metadata={}
        ).write_record_to(buffer)

        # Build chunk with message
        chunk_content = io.BytesIO()
        Message(
            channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"test"
        ).write_record_to(chunk_content)

        chunk = Chunk(
            message_start_time=1000,
            message_end_time=1000,
            uncompressed_size=len(chunk_content.getvalue()),
            uncompressed_crc=0,
            compression="",
            data=chunk_content.getvalue(),
        )

        # Write chunk with padding
        temp = io.BytesIO()
        chunk.write_record_to(temp)
        chunk_bytes = temp.getvalue()

        # Modify: add padding to the record
        opcode = chunk_bytes[0:1]
        original_length = int.from_bytes(chunk_bytes[1:9], "little")
        content = chunk_bytes[9:]
        padding = b"\x00" * 20  # Simulate future MCAP fields

        buffer.write(opcode)
        buffer.write((original_length + len(padding)).to_bytes(8, "little"))
        buffer.write(content)
        buffer.write(padding)

        # Write remaining records
        DataEnd(data_section_crc=0).write_record_to(buffer)
        Footer(summary_start=0, summary_offset_start=0, summary_crc=0).write_record_to(buffer)
        buffer.write(MAGIC)

        # Read with lazy_chunks=True
        buffer.seek(0)
        records = list(stream_reader(buffer, emit_chunks=True, lazy_chunks=True))

        # Verify LazyChunk was returned and Footer was readable
        lazy_chunks = [r for r in records if isinstance(r, LazyChunk)]
        footers = [r for r in records if isinstance(r, Footer)]

        assert len(lazy_chunks) == 1, "Should have one LazyChunk"
        assert len(footers) == 1, "Should have Footer (stream positioned correctly after padding)"


class TestSchemasInChunks:
    """Test reading MCAP files where Schema/Channel records are inside chunks.

    ROS2 rosbags (created by libmcap) store Schema and Channel records inside
    chunks rather than in the data section outside chunks. The non-seekable reader
    must handle this by fully decompressing chunks that introduce new channels.
    """

    @staticmethod
    def _build_chunk(records_data: bytes, msg_start: int, msg_end: int) -> Chunk:
        """Build an uncompressed Chunk from raw record bytes."""
        return Chunk(
            message_start_time=msg_start,
            message_end_time=msg_end,
            uncompressed_size=len(records_data),
            uncompressed_crc=0,
            compression="",
            data=records_data,
        )

    @staticmethod
    def _build_mcap_with_chunks(
        chunks: list[tuple[Chunk, list[MessageIndex]]],
    ) -> bytes:
        """Assemble a complete MCAP file from chunks and their message indexes.

        Args:
            chunks: List of (Chunk, [MessageIndex, ...]) pairs. Each chunk is
                followed by its message indexes in the data section.

        Returns:
            Complete MCAP file bytes with summary_start=0 (forces non-seeking path).
        """
        buf = io.BytesIO()
        buf.write(MAGIC)
        Header(profile="test", library="test").write_record_to(buf)

        for chunk, message_indexes in chunks:
            chunk.write_record_to(buf)
            for mi in message_indexes:
                mi.write_record_to(buf)

        DataEnd(data_section_crc=0).write_record_to(buf)
        Footer(summary_start=0, summary_offset_start=0, summary_crc=0).write_record_to(buf)
        buf.write(MAGIC)
        return buf.getvalue()

    @staticmethod
    def _build_chunk_records(
        records: list[Schema | Channel | Message],
    ) -> tuple[bytes, list[MessageIndex]]:
        """Serialize records into chunk data and build MessageIndex entries.

        Returns:
            (chunk_data_bytes, [MessageIndex, ...])
        """
        buf = io.BytesIO()
        # channel_id -> [(log_time, offset)]
        msg_offsets: dict[int, list[tuple[int, int]]] = {}

        for record in records:
            offset = buf.tell()
            record.write_record_to(buf)
            if isinstance(record, Message):
                msg_offsets.setdefault(record.channel_id, []).append((record.log_time, offset))

        message_indexes = [
            MessageIndex(
                channel_id=ch_id,
                timestamps=[t for t, _ in entries],
                offsets=[o for _, o in entries],
            )
            for ch_id, entries in msg_offsets.items()
        ]
        return buf.getvalue(), message_indexes

    @pytest.mark.parametrize("opcode", [0x10, 0x80])
    def test_chunk_skips_unknown_reserved_and_private_records(self, opcode):
        raw = io.BytesIO()
        raw.write(bytes([opcode]))
        raw.write((3).to_bytes(8, "little"))
        raw.write(b"ext")
        Schema(1, "T", "raw", b"").write_record_to(raw)
        Channel(1, 1, "/t", "raw", {}).write_record_to(raw)
        Message(1, 0, 1000, 1000, b"value").write_record_to(raw)
        chunk = self._build_chunk(raw.getvalue(), 1000, 1000)

        results = list(read_message(io.BytesIO(self._build_mcap_with_chunks([(chunk, [])]))))

        assert [(channel.topic, message.data) for _, channel, message in results] == [
            ("/t", b"value")
        ]

    def test_schemas_in_chunks_non_seekable(self):
        """Single chunk with Schema+Channel+Messages, read via non-seekable stream."""
        schema = Schema(id=1, name="TestMsg", encoding="raw", data=b"{}")
        channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="raw", metadata={})
        msg1 = Message(channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"hello")
        msg2 = Message(channel_id=1, sequence=1, log_time=2000, publish_time=2000, data=b"world")

        chunk_data, message_indexes = self._build_chunk_records([schema, channel, msg1, msg2])
        chunk = self._build_chunk(chunk_data, msg_start=1000, msg_end=2000)
        mcap_bytes = self._build_mcap_with_chunks([(chunk, message_indexes)])

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        results = list(read_message(NonSeekableStream(mcap_bytes)))
        assert len(results) == 2
        assert results[0][1].topic == "/test"
        assert results[0][2].data == b"hello"
        assert results[1][2].data == b"world"

    def test_schemas_in_chunks_multiple_channels(self):
        """Two chunks, each introducing a new channel."""
        # Chunk 1: schema1 + channel1 + messages
        s1 = Schema(id=1, name="Msg1", encoding="raw", data=b"")
        ch1 = Channel(id=1, schema_id=1, topic="/topic1", message_encoding="raw", metadata={})
        m1 = Message(channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"a")

        chunk1_data, mi1 = self._build_chunk_records([s1, ch1, m1])
        chunk1 = self._build_chunk(chunk1_data, msg_start=1000, msg_end=1000)

        # Chunk 2: schema2 + channel2 + messages
        s2 = Schema(id=2, name="Msg2", encoding="raw", data=b"")
        ch2 = Channel(id=2, schema_id=2, topic="/topic2", message_encoding="raw", metadata={})
        m2 = Message(channel_id=2, sequence=0, log_time=2000, publish_time=2000, data=b"b")

        chunk2_data, mi2 = self._build_chunk_records([s2, ch2, m2])
        chunk2 = self._build_chunk(chunk2_data, msg_start=2000, msg_end=2000)

        mcap_bytes = self._build_mcap_with_chunks([(chunk1, mi1), (chunk2, mi2)])

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        results = list(read_message(NonSeekableStream(mcap_bytes)))
        assert len(results) == 2
        assert results[0][1].topic == "/topic1"
        assert results[0][2].data == b"a"
        assert results[1][1].topic == "/topic2"
        assert results[1][2].data == b"b"

    def test_schemas_in_chunks_with_time_filtering(self):
        """Time filtering still works when schemas are inside chunks."""
        schema = Schema(id=1, name="T", encoding="raw", data=b"")
        channel = Channel(id=1, schema_id=1, topic="/t", message_encoding="raw", metadata={})
        m1 = Message(channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"early")
        m2 = Message(channel_id=1, sequence=1, log_time=5000, publish_time=5000, data=b"mid")
        m3 = Message(channel_id=1, sequence=2, log_time=9000, publish_time=9000, data=b"late")

        chunk_data, message_indexes = self._build_chunk_records([schema, channel, m1, m2, m3])
        chunk = self._build_chunk(chunk_data, msg_start=1000, msg_end=9000)
        mcap_bytes = self._build_mcap_with_chunks([(chunk, message_indexes)])

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        results = list(
            read_message(NonSeekableStream(mcap_bytes), start_time_ns=4000, end_time_ns=6000)
        )
        assert len(results) == 1
        assert results[0][2].data == b"mid"

    def test_non_seekable_filter_keeps_definitions_from_skipped_chunk(self):
        """A later indexed message can use definitions from an out-of-range chunk."""
        schema = Schema(id=1, name="T", encoding="raw", data=b"")
        channel = Channel(id=1, schema_id=1, topic="/t", message_encoding="raw", metadata={})
        early = Message(channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"early")
        late = Message(channel_id=1, sequence=1, log_time=5000, publish_time=5000, data=b"late")

        first_data, first_indexes = self._build_chunk_records([schema, channel, early])
        first_data += bytes([0x80]) + (3).to_bytes(8, "little") + b"ext"
        second_data, second_indexes = self._build_chunk_records([late])
        chunks = [
            (self._build_chunk(first_data, 1000, 1000), first_indexes),
            (self._build_chunk(second_data, 5000, 5000), second_indexes),
        ]

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        results = list(
            read_message(
                NonSeekableStream(self._build_mcap_with_chunks(chunks)), start_time_ns=4000
            )
        )

        assert [
            (channel.topic, message.log_time, message.data) for _, channel, message in results
        ] == [("/t", 5000, b"late")]

    def test_skipped_chunk_definition_scan_rejects_illegal_standard_record(self):
        schema = Schema(id=1, name="T", encoding="raw", data=b"")
        channel = Channel(id=1, schema_id=1, topic="/t", message_encoding="raw", metadata={})
        early = Message(channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"early")
        late = Message(channel_id=1, sequence=1, log_time=5000, publish_time=5000, data=b"late")
        first_data, first_indexes = self._build_chunk_records([schema, channel, early])
        illegal = io.BytesIO()
        Header(profile="test", library="test").write_record_to(illegal)
        first_data += illegal.getvalue()
        second_data, second_indexes = self._build_chunk_records([late])

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        chunk = self._build_chunk(first_data, 1000, 1000)
        chunks = [
            (chunk, first_indexes),
            (self._build_chunk(second_data, 5000, 5000), second_indexes),
        ]
        with pytest.raises(IllegalOpcodeInChunkError):
            list(
                read_message(
                    NonSeekableStream(self._build_mcap_with_chunks(chunks)),
                    start_time_ns=4000,
                )
            )

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_non_seekable_filter_flushes_skipped_definitions_before_plain_message(
        self, num_workers
    ):
        schema = Schema(id=1, name="T", encoding="raw", data=b"")
        channel = Channel(id=1, schema_id=1, topic="/t", message_encoding="raw", metadata={})
        early = Message(channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"early")
        late = Message(channel_id=1, sequence=1, log_time=5000, publish_time=5000, data=b"late")
        chunk_data, message_indexes = self._build_chunk_records([schema, channel, early])
        chunk = self._build_chunk(chunk_data, 1000, 1000)

        buffer = io.BytesIO()
        buffer.write(MAGIC)
        Header(profile="test", library="test").write_record_to(buffer)
        chunk.write_record_to(buffer)
        for message_index in message_indexes:
            message_index.write_record_to(buffer)
        late.write_record_to(buffer)
        DataEnd(data_section_crc=0).write_record_to(buffer)
        Footer(summary_start=0, summary_offset_start=0, summary_crc=0).write_record_to(buffer)
        buffer.write(MAGIC)

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        results = list(
            read_message(
                NonSeekableStream(buffer.getvalue()),
                start_time_ns=4000,
                num_workers=num_workers,
            )
        )

        assert len(results) == 1
        assert results[0][1].topic == "/t"
        assert results[0][2] == late

    def test_schemas_in_chunks_second_chunk_known_channel(self):
        """Second chunk reuses known channel — should use optimized indexed path."""
        # Chunk 1: introduces schema + channel
        schema = Schema(id=1, name="T", encoding="raw", data=b"")
        channel = Channel(id=1, schema_id=1, topic="/t", message_encoding="raw", metadata={})
        m1 = Message(channel_id=1, sequence=0, log_time=1000, publish_time=1000, data=b"first")

        chunk1_data, mi1 = self._build_chunk_records([schema, channel, m1])
        chunk1 = self._build_chunk(chunk1_data, msg_start=1000, msg_end=1000)

        # Chunk 2: only messages on already-known channel (no schema/channel records)
        m2 = Message(channel_id=1, sequence=1, log_time=2000, publish_time=2000, data=b"second")
        m3 = Message(channel_id=1, sequence=2, log_time=3000, publish_time=3000, data=b"third")

        chunk2_data, mi2 = self._build_chunk_records([m2, m3])
        chunk2 = self._build_chunk(chunk2_data, msg_start=2000, msg_end=3000)

        mcap_bytes = self._build_mcap_with_chunks([(chunk1, mi1), (chunk2, mi2)])

        class NonSeekableStream(io.BytesIO):
            def seekable(self):
                return False

        results = list(read_message(NonSeekableStream(mcap_bytes)))
        assert len(results) == 3
        assert [r[2].data for r in results] == [b"first", b"second", b"third"]
