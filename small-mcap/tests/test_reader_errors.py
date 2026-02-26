"""Tests for reader.py error handling and edge cases."""

import io
import struct

import pytest
import small_mcap.reader as reader_module
from small_mcap import (
    MAGIC,
    ChannelNotFoundError,
    CompressionType,
    CRCValidationError,
    EndOfFileError,
    IllegalOpcodeInChunkError,
    InvalidHeaderError,
    InvalidMagicError,
    McapWriter,
    RecordLengthLimitExceededError,
    SchemaNotFoundError,
    UnsupportedCompressionError,
    get_header,
    get_summary,
    read_attachment,
    read_message,
    read_metadata,
    stream_reader,
)
from small_mcap.reader import _get_chunk_data_stream, _read_summary_from_iterable, breakup_chunk
from small_mcap.records import (
    Attachment,
    Channel,
    Chunk,
    DataEnd,
    Footer,
    Header,
    Message,
    Metadata,
    Opcode,
    Schema,
)

# Test constants
WRONG_CRC_CHUNK = 12345
WRONG_CRC_DATA_END = 99999
UNKNOWN_OPCODE = 255
HUGE_RECORD_SIZE = 2**50
TRUNCATED_RECORD_SIZE = 100


@pytest.fixture
def mcap_buffer():
    """Create a BytesIO buffer with MAGIC and Header."""
    buffer = io.BytesIO()
    buffer.write(MAGIC)
    header = Header(profile="", library="")
    header.write_record_to(buffer)
    return buffer


@pytest.fixture
def mcap_writer():
    """Create a McapWriter with a BytesIO buffer."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer)
    return writer, buffer


def test_invalid_magic_at_start():
    """Test that InvalidMagicError is raised when magic bytes are invalid at start."""
    # Create a buffer with invalid magic bytes
    buffer = io.BytesIO(b"NOTMCAP0\x00\x00\x00\x00\x00\x00\x00\x00")

    with pytest.raises(InvalidMagicError) as exc_info:
        list(stream_reader(buffer))

    assert "invalid magic" in str(exc_info.value)


def test_invalid_magic_at_end():
    """Test that error is raised when magic bytes are invalid at end."""
    buffer = io.BytesIO()

    # Write valid MCAP file
    writer = McapWriter(buffer)
    writer.start()
    writer.finish()

    # Corrupt the final magic bytes
    data = buffer.getvalue()
    # The last 8 bytes should be the magic, corrupt them
    corrupted = data[:-8] + b"BADMAGIC"
    buffer = io.BytesIO(corrupted)

    # Reading should raise error when hitting corrupted footer
    with pytest.raises(InvalidMagicError):
        list(stream_reader(buffer))


def test_unsupported_compression_type(mcap_buffer):
    """Test that UnsupportedCompressionError is raised for unknown compression."""
    # Write a chunk with unsupported compression
    chunk = Chunk(
        message_start_time=0,
        message_end_time=1,
        uncompressed_size=10,
        uncompressed_crc=0,
        compression="bzip2",  # Unsupported compression type
        data=b"\x00" * 10,
    )
    chunk.write_record_to(mcap_buffer)
    mcap_buffer.seek(0)

    with pytest.raises(UnsupportedCompressionError, match="bzip2"):
        list(stream_reader(mcap_buffer))


def test_non_string_compression_type(mcap_buffer):
    """Test handling of non-string compression type."""
    # Manually create a chunk with non-UTF8 compression bytes
    # Must use struct.pack because Chunk class validates compression string
    compression = b"\x00\xff\x02"  # Invalid UTF-8 bytes
    records = b"\x00" * 10

    chunk_data = struct.pack("<QQQ", 0, 1, 10)  # times and size
    chunk_data += struct.pack("<I", 0)  # CRC
    chunk_data += struct.pack("<I", len(compression)) + compression
    chunk_data += struct.pack("<Q", len(records)) + records

    mcap_buffer.write(Opcode.CHUNK.to_bytes(1, "little"))
    mcap_buffer.write(len(chunk_data).to_bytes(8, "little"))
    mcap_buffer.write(chunk_data)
    mcap_buffer.seek(0)

    # Should handle gracefully or raise an error
    with pytest.raises((UnsupportedCompressionError, UnicodeDecodeError, ValueError)):
        list(stream_reader(mcap_buffer))


@pytest.mark.parametrize(
    ("compression_type", "expected_name"),
    [
        (CompressionType.LZ4, "lz4"),
        (CompressionType.ZSTD, "zstd"),
    ],
)
def test_compression_file(compression_type, expected_name):
    """Test reading MCAP file with different compression types."""
    buffer = io.BytesIO()

    # Create a valid MCAP file with specified compression
    writer = McapWriter(buffer, compression=compression_type)
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="", data=b"")
    writer.add_channel(channel_id=1, topic="test", message_encoding="", schema_id=1)
    writer.add_message(channel_id=1, log_time=0, publish_time=0, sequence=0, data=b"test" * 100)
    writer.finish()

    buffer.seek(0)

    # Verify compression type in chunk records
    records = list(stream_reader(buffer, emit_chunks=True))
    chunk_records = [r for r in records if isinstance(r, Chunk)]

    assert len(chunk_records) == 1, "Expected exactly one chunk record"
    assert chunk_records[0].compression == expected_name


def test_record_size_limit_exceeded(mcap_buffer):
    """Test that RecordLengthLimitExceededError is raised for huge records."""
    # Write a record with an impossibly large size (exceeds _RECORD_SIZE_LIMIT)
    # Must manually write opcode and length to create invalid size
    mcap_buffer.write(Opcode.SCHEMA.to_bytes(1, "little"))
    mcap_buffer.write(HUGE_RECORD_SIZE.to_bytes(8, "little"))
    mcap_buffer.seek(0)

    # The default _RECORD_SIZE_LIMIT should trigger the error
    with pytest.raises(RecordLengthLimitExceededError):
        list(stream_reader(mcap_buffer))


def test_chunk_crc_validation_failure(mcap_buffer):
    """Test that CRCValidationError is raised when chunk CRC doesn't match."""
    # Create uncompressed chunk with WRONG CRC
    chunk = Chunk(
        message_start_time=0,
        message_end_time=1,
        uncompressed_size=10,
        uncompressed_crc=WRONG_CRC_CHUNK,
        compression="",
        data=b"\x00" * 10,
    )
    chunk.write_record_to(mcap_buffer)
    mcap_buffer.seek(0)

    # Should raise CRCValidationError when validate_crc=True
    with pytest.raises(CRCValidationError):
        list(stream_reader(mcap_buffer, validate_crc=True))


def test_unknown_record_type(mcap_buffer):
    """Test that unknown record types are skipped gracefully."""
    # Write an unknown opcode (255 is not defined)
    # Must manually write opcode because it's not in the Opcode enum
    unknown_data = b"some unknown data"
    mcap_buffer.write(UNKNOWN_OPCODE.to_bytes(1, "little"))
    mcap_buffer.write(len(unknown_data).to_bytes(8, "little"))
    mcap_buffer.write(unknown_data)

    # Write footer using record class
    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(mcap_buffer)
    mcap_buffer.write(MAGIC)
    mcap_buffer.seek(0)

    # Should not raise an error, just skip the unknown record
    records = list(stream_reader(mcap_buffer))

    # Should have header and footer records
    record_types = [type(r).__name__ for r in records]
    assert "Header" in record_types
    assert "Footer" in record_types


def test_end_of_file_error(mcap_buffer):
    """Test that EndOfFileError is raised when file is truncated."""
    # Write a record with length but truncate the data
    # Must manually write to create truncation
    mcap_buffer.write(Opcode.SCHEMA.to_bytes(1, "little"))
    mcap_buffer.write(TRUNCATED_RECORD_SIZE.to_bytes(8, "little"))  # Says 100 bytes
    mcap_buffer.write(b"only 10 bytes!")  # But only write 14 bytes
    mcap_buffer.seek(0)

    # Should raise EndOfFileError
    with pytest.raises(EndOfFileError):
        list(stream_reader(mcap_buffer))


def test_data_end_crc_validation_failure(mcap_buffer):
    """Test that CRCValidationError is raised when DataEnd CRC doesn't match."""
    # Write DataEnd with WRONG CRC
    data_end = DataEnd(data_section_crc=WRONG_CRC_DATA_END)
    data_end.write_record_to(mcap_buffer)

    # Write footer using record class
    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(mcap_buffer)
    mcap_buffer.write(MAGIC)
    mcap_buffer.seek(0)

    # Should raise CRCValidationError when validate_crc=True
    with pytest.raises(CRCValidationError):
        list(stream_reader(mcap_buffer, validate_crc=True))


def test_record_with_padding(mcap_buffer):
    """Test reading a record with padding bytes."""
    # Write a schema with extra padding at the end
    # Use Schema class to get the data, then manually add padding
    schema = Schema(id=1, name="test", encoding="proto", data=b"schema_data")

    # Get schema content without opcode/length by writing to buffer and extracting
    temp_buf = io.BytesIO()
    schema.write_record_to(temp_buf)
    schema_record = temp_buf.getvalue()
    # Extract content (skip opcode 1 byte + length 8 bytes)
    schema_data = schema_record[9:]

    # Add padding bytes
    padding_bytes = b"\x00" * 10

    # Manually write opcode and length to include padding
    mcap_buffer.write(Opcode.SCHEMA.to_bytes(1, "little"))
    mcap_buffer.write((len(schema_data) + len(padding_bytes)).to_bytes(8, "little"))
    mcap_buffer.write(schema_data)
    mcap_buffer.write(padding_bytes)  # Padding at the end

    # Write footer using record class
    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(mcap_buffer)
    mcap_buffer.write(MAGIC)
    mcap_buffer.seek(0)

    # Should read successfully, handling the padding
    records = list(stream_reader(mcap_buffer))

    # Should have header, schema, and footer records
    record_types = [type(r).__name__ for r in records]
    assert "Header" in record_types
    assert "Schema" in record_types
    assert "Footer" in record_types


def test_get_summary_invalid_magic(mcap_buffer):
    """Test get_summary returns None with invalid ending magic bytes."""
    # Write footer using record class
    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(mcap_buffer)

    # Write WRONG magic at end
    mcap_buffer.write(b"BADMAGIC")
    mcap_buffer.seek(0)

    # Should return None (falls back to non-seeking path)
    assert get_summary(mcap_buffer) is None


def test_get_summary_non_seekable():
    """Test get_summary returns None for non-seekable streams."""

    class NonSeekableStream:
        def seekable(self) -> bool:
            return False

    stream = NonSeekableStream()
    summary = get_summary(stream)
    assert summary is None


def test_get_summary_oserror():
    """Test get_summary returns None when OSError occurs during seek."""

    class BadSeekStream(io.BytesIO):
        def __init__(self):
            super().__init__(b"x" * 100)

        def seek(self, offset, whence=0):
            if whence == io.SEEK_END:
                raise OSError("Simulated seek error")
            return super().seek(offset, whence)

    stream = BadSeekStream()
    result = get_summary(stream)
    assert result is None


def test_zstd_not_installed(monkeypatch):
    """Test UnsupportedCompressionError when zstd module not installed."""
    monkeypatch.setattr(reader_module, "ZstdDecompressor", None)

    chunk = Chunk(
        message_start_time=0,
        message_end_time=0,
        uncompressed_size=0,
        uncompressed_crc=0,
        compression="zstd",
        data=b"",
    )

    with pytest.raises(UnsupportedCompressionError, match=r"zstd.*not installed"):
        _get_chunk_data_stream(chunk)


def test_lz4_not_installed(monkeypatch):
    """Test UnsupportedCompressionError when lz4 module not installed."""
    monkeypatch.setattr(reader_module, "lz4_decompress", None)

    chunk = Chunk(
        message_start_time=0,
        message_end_time=0,
        uncompressed_size=0,
        uncompressed_crc=0,
        compression="lz4",
        data=b"",
    )

    with pytest.raises(UnsupportedCompressionError, match=r"lz4.*not installed"):
        _get_chunk_data_stream(chunk)


def test_get_header_not_header():
    """Test get_header raises when first record isn't Header."""
    # Write a Footer instead of Header after the magic
    buffer = io.BytesIO()
    buffer.write(MAGIC)  # Valid magic
    # Write Footer record instead of Header
    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(buffer)
    buffer.write(MAGIC)
    buffer.seek(0)

    with pytest.raises(InvalidHeaderError):
        get_header(buffer)


def test_read_summary_from_iterable_no_footer():
    """_read_summary_from_iterable returns summary when no Footer found."""
    # Pass an iterable with just a Schema (no Footer)
    schema = Schema(id=1, name="test", encoding="raw", data=b"")
    result = _read_summary_from_iterable([schema])

    # Should return the summary (with schema added)
    assert result is not None
    assert 1 in result.schemas


def test_read_summary_from_iterable_footer_no_summary():
    """_read_summary_from_iterable returns None when footer has summary_start=0."""
    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    result = _read_summary_from_iterable([footer])

    assert result is None


def test_message_references_unknown_channel():
    """_read_inner should raise when message references unknown channel."""
    # Create MCAP with message but no channel
    buffer = io.BytesIO()
    buffer.write(MAGIC)
    header = Header(profile="", library="")
    header.write_record_to(buffer)

    # Write a message for a channel that doesn't exist
    msg = Message(channel_id=99, sequence=0, log_time=0, publish_time=0, data=b"data")
    msg.write_record_to(buffer)

    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(buffer)
    buffer.write(MAGIC)
    buffer.seek(0)

    with pytest.raises(ChannelNotFoundError, match="99"):
        list(read_message(buffer))


def test_channel_references_unknown_schema():
    """_read_inner should raise when channel references unknown schema."""
    buffer = io.BytesIO()
    buffer.write(MAGIC)
    header = Header(profile="", library="")
    header.write_record_to(buffer)

    # Write a channel referencing schema_id=99 that doesn't exist
    channel = Channel(id=1, schema_id=99, topic="/test", message_encoding="raw", metadata={})
    channel.write_record_to(buffer)

    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(buffer)
    buffer.write(MAGIC)
    buffer.seek(0)

    with pytest.raises(SchemaNotFoundError, match="99"):
        list(read_message(buffer))


# --- Tests for incomplete MCAP files (still being written) ---


def _build_incomplete_mcap() -> bytes:
    """Build an MCAP file without footer or trailing magic (simulates a file still being written)."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer)
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")
    writer.add_channel(channel_id=1, topic="/test", message_encoding="raw", schema_id=1)
    writer.add_message(channel_id=1, log_time=1000, publish_time=1000, sequence=0, data=b"hello")
    writer.add_message(channel_id=1, log_time=2000, publish_time=2000, sequence=1, data=b"world")
    writer.finish()

    # Strip footer and trailing magic to simulate incomplete file
    full_data = buffer.getvalue()
    # Find the footer opcode (0x02) by scanning from end
    # Footer record: opcode(1) + length(8) + summary_start(8) + summary_offset_start(8) + summary_crc(4)
    # Trailing magic: 8 bytes
    # Strip from DataEnd onward by finding DataEnd opcode
    # Simpler: just strip everything after data section
    # The data section ends before the DataEnd record. Let's find it.
    from small_mcap.records import Opcode

    # Scan for DataEnd opcode to find where to truncate
    pos = len(MAGIC)  # skip leading magic
    view = memoryview(full_data)
    while pos < len(view):
        opcode = view[pos]
        length = int.from_bytes(view[pos + 1 : pos + 9], "little")
        if opcode == Opcode.DATA_END:
            # Truncate here — no DataEnd, no footer, no trailing magic
            return bytes(full_data[:pos])
        pos += 9 + length

    # Fallback: strip last 50 bytes (footer + magic)
    return full_data[:-50]


def test_read_messages_from_incomplete_mcap():
    """Reading messages from an incomplete MCAP (no footer/trailing magic) should work."""
    data = _build_incomplete_mcap()
    stream = io.BytesIO(data)

    messages = list(read_message(stream))
    assert len(messages) == 2
    assert messages[0][2].log_time == 1000
    assert messages[1][2].log_time == 2000


def test_get_summary_returns_none_for_incomplete_mcap():
    """get_summary should return None for an incomplete MCAP file."""
    data = _build_incomplete_mcap()
    stream = io.BytesIO(data)

    summary = get_summary(stream)
    assert summary is None


def test_stream_reader_allow_incomplete_true():
    """stream_reader(allow_incomplete=True) should yield complete records and stop at truncation."""
    data = _build_incomplete_mcap()
    stream = io.BytesIO(data)

    records = list(stream_reader(stream, allow_incomplete=True))
    # Should have at least Header + records from chunks, no crash
    record_types = [type(r).__name__ for r in records]
    assert "Header" in record_types
    # Should NOT have Footer (file is incomplete)
    assert "Footer" not in record_types


def test_stream_reader_allow_incomplete_false_raises():
    """stream_reader(allow_incomplete=False) should raise EndOfFileError on truncated file."""
    data = _build_incomplete_mcap()
    stream = io.BytesIO(data)

    with pytest.raises(EndOfFileError):
        list(stream_reader(stream, allow_incomplete=False))


def test_stream_reader_mid_record_truncation():
    """stream_reader(allow_incomplete=True) handles truncation mid-record."""
    # Build a valid MCAP, then truncate in the middle of a record
    buffer = io.BytesIO()
    buffer.write(MAGIC)
    header = Header(profile="", library="")
    header.write_record_to(buffer)

    # Write a schema record header claiming 100 bytes, but only write 10
    buffer.write(Opcode.SCHEMA.to_bytes(1, "little"))
    buffer.write((100).to_bytes(8, "little"))
    buffer.write(b"0123456789")  # Only 10 bytes of the 100

    data = buffer.getvalue()
    stream = io.BytesIO(data)

    records = list(stream_reader(stream, allow_incomplete=True))
    # Should have Header only, truncated record is skipped
    assert len(records) == 1
    assert isinstance(records[0], Header)


# --- Tests for IllegalOpcodeInChunkError ---


def test_illegal_opcode_in_chunk_breakup():
    """breakup_chunk raises IllegalOpcodeInChunkError for non-Schema/Channel/Message opcodes."""
    # Build chunk data containing an Attachment record (illegal inside chunks)
    inner_buf = io.BytesIO()
    # Write a valid message first
    Message(channel_id=1, sequence=0, log_time=0, publish_time=0, data=b"ok").write_record_to(
        inner_buf
    )
    # Write an illegal record (DataEnd opcode inside chunk)
    DataEnd(data_section_crc=0).write_record_to(inner_buf)

    chunk_data = inner_buf.getvalue()
    chunk = Chunk(
        message_start_time=0,
        message_end_time=0,
        uncompressed_size=len(chunk_data),
        uncompressed_crc=0,
        compression="",
        data=chunk_data,
    )

    records = breakup_chunk(chunk)
    # First record is the valid message
    first = next(iter(records))
    assert isinstance(first, Message)

    # Second record has illegal opcode, should raise
    with pytest.raises(IllegalOpcodeInChunkError, match="illegal opcode"):
        list(records)


def test_illegal_opcode_in_chunk_via_stream_reader(mcap_buffer):
    """stream_reader with emit_chunks=False raises IllegalOpcodeInChunkError for illegal opcodes."""
    # Build chunk data with a Statistics record (illegal inside chunk)
    from small_mcap.records import Statistics

    inner_buf = io.BytesIO()
    Statistics(
        message_count=0,
        schema_count=0,
        channel_count=0,
        attachment_count=0,
        metadata_count=0,
        chunk_count=0,
        message_start_time=0,
        message_end_time=0,
        channel_message_counts={},
    ).write_record_to(inner_buf)

    chunk_data = inner_buf.getvalue()
    chunk = Chunk(
        message_start_time=0,
        message_end_time=0,
        uncompressed_size=len(chunk_data),
        uncompressed_crc=0,
        compression="",
        data=chunk_data,
    )
    chunk.write_record_to(mcap_buffer)
    mcap_buffer.seek(0)

    with pytest.raises(IllegalOpcodeInChunkError):
        list(stream_reader(mcap_buffer))


# --- Tests for read_attachment and read_metadata ---


def test_read_attachment_from_index():
    """read_attachment reads a full Attachment record given its AttachmentIndex."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer)
    writer.start()
    writer.add_attachment(
        log_time=1000,
        create_time=2000,
        name="test.txt",
        media_type="text/plain",
        data=b"hello world",
    )
    writer.finish()

    buffer.seek(0)
    summary = get_summary(buffer)
    assert summary is not None
    assert len(summary.attachment_indexes) == 1

    attachment = read_attachment(buffer, summary.attachment_indexes[0])
    assert isinstance(attachment, Attachment)
    assert attachment.name == "test.txt"
    assert attachment.media_type == "text/plain"
    assert attachment.data == b"hello world"
    assert attachment.log_time == 1000
    assert attachment.create_time == 2000


def test_read_metadata_from_index():
    """read_metadata reads a full Metadata record given its MetadataIndex."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer)
    writer.start()
    writer.add_metadata("hardware_info", {"serial": "ABC123", "board": "v2"})
    writer.finish()

    buffer.seek(0)
    summary = get_summary(buffer)
    assert summary is not None
    assert len(summary.metadata_indexes) == 1

    metadata = read_metadata(buffer, summary.metadata_indexes[0])
    assert isinstance(metadata, Metadata)
    assert metadata.name == "hardware_info"
    assert metadata.metadata == {"serial": "ABC123", "board": "v2"}
