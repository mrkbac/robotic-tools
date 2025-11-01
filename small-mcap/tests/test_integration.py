"""Integration tests for small-mcap writer and reader."""

from pathlib import Path

from small_mcap import CompressionType, McapWriter, get_header, get_summary, read_message


class TestIntegration:
    """Test writer and reader integration."""

    def test_minimal_file(self, temp_mcap_file: Path):
        """Test creating and reading minimal MCAP file."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start(profile="test", library="small-mcap")
            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            header = get_header(f)
            assert header.profile == "test"
            assert header.library == "small-mcap"

    def test_write_read_messages(self, temp_mcap_file: Path):
        """Test writing and reading messages."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=1024)
            writer.start()
            writer.add_schema(1, "Test", "json", b"{}")
            writer.add_channel(1, "/test", "json", 1)

            for i in range(10):
                writer.add_message(1, i * 1000, f"msg{i}".encode(), i * 1000)

            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            msgs = list(read_message(f))
            assert len(msgs) == 10
            # Verify at least the first message fields
            schema, channel, message = msgs[0]
            assert schema.id == 1
            assert schema.name == "Test"
            assert channel.id == 1
            assert channel.topic == "/test"
            assert message.channel_id == 1
            assert message.log_time == 0
            assert message.data == b"msg0"

    def test_compression_zstd(self, temp_mcap_file: Path):
        """Test ZSTD compression."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, compression=CompressionType.ZSTD, chunk_size=1024)
            writer.start()
            writer.add_schema(1, "T", "json", b"{}")
            writer.add_channel(1, "/t", "json", 1)
            for i in range(10):
                writer.add_message(1, i, b"x" * 100, i)
            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            msgs = list(read_message(f))
            assert len(msgs) == 10
            # Verify at least the first message fields
            schema, channel, message = msgs[0]
            assert schema.id == 1
            assert schema.name == "T"
            assert channel.id == 1
            assert channel.topic == "/t"
            assert message.channel_id == 1
            assert message.log_time == 0
            assert message.data == b"x" * 100

    def test_attachments(self, temp_mcap_file: Path):
        """Test attachments."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()
            writer.add_attachment(1000, 1000, "file.txt", "text/plain", b"hello")
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

    def test_metadata(self, temp_mcap_file: Path):
        """Test metadata."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()
            writer.add_metadata("cfg", {"v": "1"})
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

    def test_statistics(self, temp_mcap_file: Path):
        """Test statistics tracking."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()
            writer.add_schema(1, "T", "json", b"{}")
            writer.add_channel(1, "/t", "json", 1)
            for i in range(42):
                writer.add_message(1, i, b"x", i)
            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            stats = summary.statistics
            assert stats.message_count == 42
            assert stats.schema_count == 1
            assert stats.channel_count == 1
            assert stats.attachment_count == 0
            assert stats.metadata_count == 0
            assert stats.chunk_count >= 0
            assert stats.message_start_time == 0
            assert stats.message_end_time == 41
            assert stats.channel_message_counts == {1: 42}
