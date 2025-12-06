"""Tests for reader.py error handling and edge cases."""

import io
import struct

import pytest
from small_mcap import (
    MAGIC,
    CompressionType,
    CRCValidationError,
    EndOfFileError,
    InvalidMagicError,
    McapWriter,
    RecordLengthLimitExceededError,
    UnsupportedCompressionError,
    get_summary,
    stream_reader,
)
from small_mcap.records import Chunk, DataEnd, Footer, Header, Opcode, Schema

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
    """Test get_summary with invalid ending magic bytes."""
    # Write footer using record class
    footer = Footer(summary_start=0, summary_offset_start=0, summary_crc=0)
    footer.write_record_to(mcap_buffer)

    # Write WRONG magic at end
    mcap_buffer.write(b"BADMAGIC")
    mcap_buffer.seek(0)

    # Should raise InvalidMagicError
    with pytest.raises(InvalidMagicError):
        get_summary(mcap_buffer)


def test_get_summary_non_seekable():
    """Test get_summary returns None for non-seekable streams."""

    class NonSeekableStream:
        def seekable(self) -> bool:
            return False

    stream = NonSeekableStream()
    summary = get_summary(stream)
    assert summary is None
