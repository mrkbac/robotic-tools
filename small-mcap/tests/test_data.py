"""Tests for small_mcap.data module - record serialization and deserialization."""

import io

from small_mcap import (
    MAGIC,
    MAGIC_SIZE,
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
    Message,
    MessageIndex,
    Metadata,
    MetadataIndex,
    Opcode,
    Schema,
    Statistics,
    SummaryOffset,
)
from small_mcap.records import McapRecord, _read_map, _read_string, _write_map, _write_string


def write_content(record: McapRecord) -> bytes:
    """Write record to buffer and extract content (without opcode and length prefix)."""
    buf = io.BytesIO()
    record.write_record_to(buf)
    # Skip opcode (1 byte) and length (8 bytes) to get content only
    return buf.getvalue()[9:]


class TestHelperFunctions:
    """Test helper serialization functions."""

    def test_write_read_string_roundtrip(self):
        """Test writing and reading a string."""
        original = "hello world"
        serialized = _write_string(original)
        deserialized, offset = _read_string(serialized, 0)

        assert deserialized == original
        assert offset == len(serialized)

    def test_write_read_empty_string(self):
        """Test writing and reading an empty string."""
        original = ""
        serialized = _write_string(original)
        deserialized, offset = _read_string(serialized, 0)

        assert deserialized == original
        assert offset == 4  # Just the length prefix
        assert len(serialized) == 4

    def test_write_read_unicode_string(self):
        """Test writing and reading a Unicode string."""
        original = "Hello 世界 🚀"
        serialized = _write_string(original)
        deserialized, offset = _read_string(serialized, 0)

        assert deserialized == original
        assert offset == len(serialized)

    def test_write_read_string_with_offset(self):
        """Test reading a string at a specific offset."""
        prefix = b"\x00\x00\x00\x00"
        string_data = _write_string("test")
        combined = prefix + string_data

        deserialized, offset = _read_string(combined, 4)
        assert deserialized == "test"
        assert offset == len(combined)

    def test_write_read_map_roundtrip(self):
        """Test writing and reading a map."""
        original = {"key1": "value1", "key2": "value2"}
        serialized = _write_map(original)
        deserialized, offset = _read_map(serialized, 0)

        assert deserialized == original
        assert offset == len(serialized)

    def test_write_read_empty_map(self):
        """Test writing and reading an empty map."""
        original: dict[str, str] = {}
        serialized = _write_map(original)
        deserialized, offset = _read_map(serialized, 0)

        assert deserialized == original
        assert offset == 4  # Just the length prefix
        assert len(serialized) == 4

    def test_write_read_map_with_unicode(self):
        """Test writing and reading a map with Unicode values."""
        original = {"greeting": "Hello 世界", "emoji": "🚀"}
        serialized = _write_map(original)
        deserialized, _offset = _read_map(serialized, 0)

        assert deserialized == original

    def test_write_read_map_with_empty_values(self):
        """Test writing and reading a map with empty string values."""
        original = {"key1": "", "key2": "value", "key3": ""}
        serialized = _write_map(original)
        deserialized, _offset = _read_map(serialized, 0)

        assert deserialized == original


class TestHeader:
    """Test Header record serialization."""

    def test_header_write_read_roundtrip(self):
        """Test Header write and read roundtrip."""
        original = Header(profile="ros2", library="small-mcap-test")
        serialized = write_content(original)
        deserialized = Header.read(serialized)

        assert deserialized == original
        assert deserialized.profile == "ros2"
        assert deserialized.library == "small-mcap-test"

    def test_header_with_empty_profile(self):
        """Test Header with empty profile."""
        original = Header(profile="", library="test")
        serialized = write_content(original)
        deserialized = Header.read(serialized)

        assert deserialized == original

    def test_header_opcode(self):
        """Test Header has correct opcode."""
        assert Header.OPCODE == Opcode.HEADER


class TestFooter:
    """Test Footer record serialization."""

    def test_footer_write_read_roundtrip(self):
        """Test Footer write and read roundtrip."""
        original = Footer(
            summary_start=1234,
            summary_offset_start=5678,
            summary_crc=0xDEADBEEF,
        )
        serialized = write_content(original)
        deserialized = Footer.read(serialized)

        assert deserialized == original
        assert deserialized.summary_start == 1234
        assert deserialized.summary_offset_start == 5678
        assert deserialized.summary_crc == 0xDEADBEEF

    def test_footer_with_zero_values(self):
        """Test Footer with zero values."""
        original = Footer(
            summary_start=0,
            summary_offset_start=0,
            summary_crc=0,
        )
        serialized = write_content(original)
        deserialized = Footer.read(serialized)

        assert deserialized == original

    def test_footer_opcode(self):
        """Test Footer has correct opcode."""
        assert Footer.OPCODE == Opcode.FOOTER


class TestSchema:
    """Test Schema record serialization."""

    def test_schema_write_read_roundtrip(self):
        """Test Schema write and read roundtrip."""
        original = Schema(
            id=42,
            name="TestSchema",
            encoding="protobuf",
            data=b"message Test { int32 value = 1; }",
        )
        serialized = write_content(original)
        deserialized = Schema.read(serialized)

        assert deserialized == original
        assert deserialized.id == 42
        assert deserialized.name == "TestSchema"
        assert deserialized.encoding == "protobuf"
        assert deserialized.data == b"message Test { int32 value = 1; }"

    def test_schema_with_empty_data(self):
        """Test Schema with empty data."""
        original = Schema(id=1, name="Empty", encoding="json", data=b"")
        serialized = write_content(original)
        deserialized = Schema.read(serialized)

        assert deserialized == original

    def test_schema_with_large_data(self):
        """Test Schema with large data."""
        large_data = b"x" * 100000
        original = Schema(id=1, name="Large", encoding="flatbuffer", data=large_data)
        serialized = write_content(original)
        deserialized = Schema.read(serialized)

        assert deserialized == original
        assert len(deserialized.data) == 100000

    def test_schema_opcode(self):
        """Test Schema has correct opcode."""
        assert Schema.OPCODE == Opcode.SCHEMA


class TestChannel:
    """Test Channel record serialization."""

    def test_channel_write_read_roundtrip(self):
        """Test Channel write and read roundtrip."""
        original = Channel(
            id=1,
            schema_id=42,
            topic="/test/topic",
            message_encoding="protobuf",
            metadata={"key": "value", "foo": "bar"},
        )
        serialized = write_content(original)
        deserialized = Channel.read(serialized)

        assert deserialized == original
        assert deserialized.id == 1
        assert deserialized.schema_id == 42
        assert deserialized.topic == "/test/topic"
        assert deserialized.message_encoding == "protobuf"
        assert deserialized.metadata == {"key": "value", "foo": "bar"}

    def test_channel_with_empty_metadata(self):
        """Test Channel with empty metadata."""
        original = Channel(
            id=1,
            schema_id=0,
            topic="/test",
            message_encoding="json",
            metadata={},
        )
        serialized = write_content(original)
        deserialized = Channel.read(serialized)

        assert deserialized == original
        assert deserialized.metadata == {}

    def test_channel_with_no_schema(self):
        """Test Channel with schema_id=0 (no schema)."""
        original = Channel(
            id=1,
            schema_id=0,
            topic="/raw",
            message_encoding="application/octet-stream",
            metadata={},
        )
        serialized = write_content(original)
        deserialized = Channel.read(serialized)

        assert deserialized == original
        assert deserialized.schema_id == 0

    def test_channel_opcode(self):
        """Test Channel has correct opcode."""
        assert Channel.OPCODE == Opcode.CHANNEL


class TestMessage:
    """Test Message record serialization."""

    def test_message_write_read_roundtrip(self):
        """Test Message write and read roundtrip."""
        original = Message(
            channel_id=1,
            sequence=42,
            log_time=1000000000,
            publish_time=1000000001,
            data=b"\x08\x2a",
        )
        serialized = write_content(original)
        deserialized = Message.read(serialized)

        assert deserialized == original
        assert deserialized.channel_id == 1
        assert deserialized.sequence == 42
        assert deserialized.log_time == 1000000000
        assert deserialized.publish_time == 1000000001
        assert deserialized.data == b"\x08\x2a"

    def test_message_with_empty_data(self):
        """Test Message with empty data."""
        original = Message(
            channel_id=1,
            sequence=0,
            log_time=0,
            publish_time=0,
            data=b"",
        )
        serialized = write_content(original)
        deserialized = Message.read(serialized)

        assert deserialized == original

    def test_message_with_large_data(self):
        """Test Message with large data."""
        large_data = b"x" * 1000000
        original = Message(
            channel_id=1,
            sequence=1,
            log_time=1000,
            publish_time=1000,
            data=large_data,
        )
        serialized = write_content(original)
        deserialized = Message.read(serialized)

        assert deserialized == original
        assert len(deserialized.data) == 1000000

    def test_message_opcode(self):
        """Test Message has correct opcode."""
        assert Message.OPCODE == Opcode.MESSAGE


class TestChunk:
    """Test Chunk record serialization."""

    def test_chunk_write_read_roundtrip(self):
        """Test Chunk write and read roundtrip."""
        original = Chunk(
            message_start_time=1000000000,
            message_end_time=2000000000,
            uncompressed_size=1024,
            uncompressed_crc=0x12345678,
            compression="zstd",
            data=b"\x00\x01\x02\x03",
        )
        serialized = write_content(original)
        deserialized = Chunk.read(serialized)

        assert deserialized == original
        assert deserialized.message_start_time == 1000000000
        assert deserialized.message_end_time == 2000000000
        assert deserialized.uncompressed_size == 1024
        assert deserialized.uncompressed_crc == 0x12345678
        assert deserialized.compression == "zstd"
        assert deserialized.data == b"\x00\x01\x02\x03"

    def test_chunk_with_no_compression(self):
        """Test Chunk with no compression."""
        original = Chunk(
            message_start_time=1000,
            message_end_time=2000,
            uncompressed_size=100,
            uncompressed_crc=0,
            compression="",
            data=b"test data",
        )
        serialized = write_content(original)
        deserialized = Chunk.read(serialized)

        assert deserialized == original
        assert deserialized.compression == ""

    def test_chunk_opcode(self):
        """Test Chunk has correct opcode."""
        assert Chunk.OPCODE == Opcode.CHUNK


class TestMessageIndex:
    """Test MessageIndex record serialization."""

    def test_message_index_write_read_roundtrip(self):
        """Test MessageIndex write and read roundtrip."""
        original = MessageIndex(
            channel_id=1,
            records=[
                (1000, 100),
                (1000, 200),
                (2000, 300),
                (3000, 400),
                (3000, 500),
                (3000, 600),
            ],
        )
        serialized = write_content(original)
        deserialized = MessageIndex.read(serialized)

        assert deserialized == original
        assert deserialized.channel_id == 1
        assert len(deserialized.records) == 6

    def test_message_index_with_empty_records(self):
        """Test MessageIndex with empty records."""
        original = MessageIndex(channel_id=1, records=[])
        serialized = write_content(original)
        deserialized = MessageIndex.read(serialized)

        assert deserialized == original
        assert deserialized.records == []

    def test_message_index_opcode(self):
        """Test MessageIndex has correct opcode."""
        assert MessageIndex.OPCODE == Opcode.MESSAGE_INDEX


class TestChunkIndex:
    """Test ChunkIndex record serialization."""

    def test_chunk_index_write_read_roundtrip(self):
        """Test ChunkIndex write and read roundtrip."""
        original = ChunkIndex(
            message_start_time=1000000000,
            message_end_time=2000000000,
            chunk_start_offset=1024,
            chunk_length=4096,
            message_index_offsets={1: 5120, 2: 6144},
            message_index_length=1024,
            compression="lz4",
            compressed_size=2048,
            uncompressed_size=4096,
        )
        serialized = write_content(original)
        deserialized = ChunkIndex.read(serialized)

        assert deserialized == original
        assert deserialized.message_start_time == 1000000000
        assert deserialized.message_end_time == 2000000000
        assert deserialized.chunk_start_offset == 1024
        assert deserialized.chunk_length == 4096
        assert deserialized.message_index_offsets == {1: 5120, 2: 6144}
        assert deserialized.message_index_length == 1024
        assert deserialized.compression == "lz4"
        assert deserialized.compressed_size == 2048
        assert deserialized.uncompressed_size == 4096

    def test_chunk_index_with_empty_message_index_offsets(self):
        """Test ChunkIndex with empty message_index_offsets."""
        original = ChunkIndex(
            message_start_time=1000,
            message_end_time=2000,
            chunk_start_offset=100,
            chunk_length=200,
            message_index_offsets={},
            message_index_length=0,
            compression="",
            compressed_size=200,
            uncompressed_size=200,
        )
        serialized = write_content(original)
        deserialized = ChunkIndex.read(serialized)

        assert deserialized == original
        assert deserialized.message_index_offsets == {}

    def test_chunk_index_opcode(self):
        """Test ChunkIndex has correct opcode."""
        assert ChunkIndex.OPCODE == Opcode.CHUNK_INDEX


class TestAttachment:
    """Test Attachment record serialization."""

    def test_attachment_write_read_roundtrip(self):
        """Test Attachment write and read roundtrip."""
        original = Attachment(
            log_time=1000000000,
            create_time=1000000001,
            name="config.yaml",
            media_type="application/yaml",
            data=b"key: value\n",
        )
        serialized = write_content(original)
        deserialized = Attachment.read(serialized)

        assert deserialized == original
        assert deserialized.log_time == 1000000000
        assert deserialized.create_time == 1000000001
        assert deserialized.name == "config.yaml"
        assert deserialized.media_type == "application/yaml"
        assert deserialized.data == b"key: value\n"

    def test_attachment_with_empty_data(self):
        """Test Attachment with empty data."""
        original = Attachment(
            log_time=1000,
            create_time=1000,
            name="empty.txt",
            media_type="text/plain",
            data=b"",
        )
        serialized = write_content(original)
        deserialized = Attachment.read(serialized)

        assert deserialized == original

    def test_attachment_opcode(self):
        """Test Attachment has correct opcode."""
        assert Attachment.OPCODE == Opcode.ATTACHMENT


class TestAttachmentIndex:
    """Test AttachmentIndex record serialization."""

    def test_attachment_index_write_read_roundtrip(self):
        """Test AttachmentIndex write and read roundtrip."""
        original = AttachmentIndex(
            offset=1024,
            length=2048,
            log_time=1000000000,
            create_time=1000000001,
            data_size=2000,
            name="file.bin",
            media_type="application/octet-stream",
        )
        serialized = write_content(original)
        deserialized = AttachmentIndex.read(serialized)

        assert deserialized == original
        assert deserialized.offset == 1024
        assert deserialized.length == 2048
        assert deserialized.log_time == 1000000000
        assert deserialized.create_time == 1000000001
        assert deserialized.data_size == 2000
        assert deserialized.name == "file.bin"
        assert deserialized.media_type == "application/octet-stream"

    def test_attachment_index_opcode(self):
        """Test AttachmentIndex has correct opcode."""
        assert AttachmentIndex.OPCODE == Opcode.ATTACHMENT_INDEX


class TestMetadata:
    """Test Metadata record serialization."""

    def test_metadata_write_read_roundtrip(self):
        """Test Metadata write and read roundtrip."""
        original = Metadata(
            name="config",
            metadata={"version": "1.0", "author": "test"},
        )
        serialized = write_content(original)
        deserialized = Metadata.read(serialized)

        assert deserialized == original
        assert deserialized.name == "config"
        assert deserialized.metadata == {"version": "1.0", "author": "test"}

    def test_metadata_with_empty_metadata(self):
        """Test Metadata with empty metadata dict."""
        original = Metadata(name="empty", metadata={})
        serialized = write_content(original)
        deserialized = Metadata.read(serialized)

        assert deserialized == original
        assert deserialized.metadata == {}

    def test_metadata_opcode(self):
        """Test Metadata has correct opcode."""
        assert Metadata.OPCODE == Opcode.METADATA


class TestMetadataIndex:
    """Test MetadataIndex record serialization."""

    def test_metadata_index_write_read_roundtrip(self):
        """Test MetadataIndex write and read roundtrip."""
        original = MetadataIndex(
            offset=1024,
            length=256,
            name="config",
        )
        serialized = write_content(original)
        deserialized = MetadataIndex.read(serialized)

        assert deserialized == original
        assert deserialized.offset == 1024
        assert deserialized.length == 256
        assert deserialized.name == "config"

    def test_metadata_index_opcode(self):
        """Test MetadataIndex has correct opcode."""
        assert MetadataIndex.OPCODE == Opcode.METADATA_INDEX


class TestStatistics:
    """Test Statistics record serialization."""

    def test_statistics_write_read_roundtrip(self):
        """Test Statistics write and read roundtrip."""
        original = Statistics(
            message_count=1000,
            schema_count=5,
            channel_count=10,
            attachment_count=2,
            metadata_count=3,
            chunk_count=50,
            message_start_time=1000000000,
            message_end_time=2000000000,
            channel_message_counts={1: 500, 2: 300, 3: 200},
        )
        serialized = write_content(original)
        deserialized = Statistics.read(serialized)

        assert deserialized == original
        assert deserialized.message_count == 1000
        assert deserialized.schema_count == 5
        assert deserialized.channel_count == 10
        assert deserialized.attachment_count == 2
        assert deserialized.metadata_count == 3
        assert deserialized.chunk_count == 50
        assert deserialized.message_start_time == 1000000000
        assert deserialized.message_end_time == 2000000000
        assert deserialized.channel_message_counts == {1: 500, 2: 300, 3: 200}

    def test_statistics_with_zero_counts(self):
        """Test Statistics with zero counts."""
        original = Statistics(
            message_count=0,
            schema_count=0,
            channel_count=0,
            attachment_count=0,
            metadata_count=0,
            chunk_count=0,
            message_start_time=0,
            message_end_time=0,
            channel_message_counts={},
        )
        serialized = write_content(original)
        deserialized = Statistics.read(serialized)

        assert deserialized == original

    def test_statistics_opcode(self):
        """Test Statistics has correct opcode."""
        assert Statistics.OPCODE == Opcode.STATISTICS


class TestSummaryOffset:
    """Test SummaryOffset record serialization."""

    def test_summary_offset_write_read_roundtrip(self):
        """Test SummaryOffset write and read roundtrip."""
        original = SummaryOffset(
            group_opcode=Opcode.SCHEMA,
            group_start=1024,
            group_length=2048,
        )
        serialized = write_content(original)
        deserialized = SummaryOffset.read(serialized)

        assert deserialized == original
        assert deserialized.group_opcode == Opcode.SCHEMA
        assert deserialized.group_start == 1024
        assert deserialized.group_length == 2048

    def test_summary_offset_with_different_opcodes(self):
        """Test SummaryOffset with different group opcodes."""
        for opcode in [
            Opcode.SCHEMA,
            Opcode.CHANNEL,
            Opcode.CHUNK_INDEX,
            Opcode.ATTACHMENT_INDEX,
            Opcode.METADATA_INDEX,
        ]:
            original = SummaryOffset(
                group_opcode=opcode,
                group_start=100,
                group_length=200,
            )
            serialized = write_content(original)
            deserialized = SummaryOffset.read(serialized)

            assert deserialized == original
            assert deserialized.group_opcode == opcode

    def test_summary_offset_opcode(self):
        """Test SummaryOffset has correct opcode."""
        assert SummaryOffset.OPCODE == Opcode.SUMMARY_OFFSET


class TestDataEnd:
    """Test DataEnd record serialization."""

    def test_data_end_write_read_roundtrip(self):
        """Test DataEnd write and read roundtrip."""
        original = DataEnd(data_section_crc=0xDEADBEEF)
        serialized = write_content(original)
        deserialized = DataEnd.read(serialized)

        assert deserialized == original
        assert deserialized.data_section_crc == 0xDEADBEEF

    def test_data_end_with_zero_crc(self):
        """Test DataEnd with zero CRC (CRC validation disabled)."""
        original = DataEnd(data_section_crc=0)
        serialized = write_content(original)
        deserialized = DataEnd.read(serialized)

        assert deserialized == original
        assert deserialized.data_section_crc == 0

    def test_data_end_opcode(self):
        """Test DataEnd has correct opcode."""
        assert DataEnd.OPCODE == Opcode.DATA_END


class TestMagic:
    """Test MCAP magic bytes."""

    def test_magic_value(self):
        """Test MCAP magic bytes value."""
        assert MAGIC == b"\x89MCAP0\r\n"

    def test_magic_size(self):
        """Test MCAP magic bytes size."""
        assert MAGIC_SIZE == 8
        assert len(MAGIC) == MAGIC_SIZE
