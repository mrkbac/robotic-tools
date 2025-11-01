"""Tests for compatibility with the official mcap library."""

from pathlib import Path

import pytest
from mcap.reader import make_reader as mcap_make_reader
from mcap.writer import Writer as McapWriter
from small_mcap import CompressionType
from small_mcap import McapWriter as SmallMcapWriter
from small_mcap import read_message as small_read_message


@pytest.mark.compat
class TestSmallMcapWriterOfficialReader:
    """Test that files written by small-mcap can be read by official mcap."""

    def test_write_small_read_official(self, temp_mcap_file: Path):
        """Write with small-mcap, read with official mcap."""
        # Write with small-mcap
        with open(temp_mcap_file, "wb") as f:
            writer = SmallMcapWriter(f, chunk_size=1024)
            writer.start(profile="test", library="small-mcap")
            writer.add_schema(1, "TestSchema", "json", b'{"type": "object"}')
            writer.add_channel(1, "/test/topic", "json", 1)

            for i in range(10):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1000000,
                    data=f'{{"value": {i}}}'.encode(),
                    publish_time=i * 1000000,
                )

            writer.finish()

        # Read with official mcap
        with open(temp_mcap_file, "rb") as f:
            reader = mcap_make_reader(f)

            # Verify header
            header = reader.get_header()
            assert header.profile == "test"
            assert header.library == "small-mcap"

            # Verify messages
            messages = list(reader.iter_messages())
            assert len(messages) == 10

            for i, (schema, channel, message) in enumerate(messages):
                # Verify complete schema fields
                assert schema.id == 1
                assert schema.name == "TestSchema"
                assert schema.encoding == "json"
                assert schema.data == b'{"type": "object"}'
                # Verify complete channel fields
                assert channel.id == 1
                assert channel.schema_id == 1
                assert channel.topic == "/test/topic"
                assert channel.message_encoding == "json"
                # Verify complete message fields
                assert message.channel_id == 1
                assert message.log_time == i * 1000000
                assert message.publish_time == i * 1000000
                assert message.data == f'{{"value": {i}}}'.encode()


@pytest.mark.compat
class TestOfficialWriterSmallMcapReader:
    """Test that files written by official mcap can be read by small-mcap."""

    def test_write_official_read_small(self, temp_mcap_file: Path):
        """Write with official mcap, read with small-mcap."""
        # Write with official mcap
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start(profile="test", library="official-mcap")

            schema_id = writer.register_schema(
                name="TestSchema",
                encoding="json",
                data=b'{"type": "object"}',
            )

            channel_id = writer.register_channel(
                schema_id=schema_id,
                topic="/official/topic",
                message_encoding="json",
            )

            for i in range(10):
                writer.add_message(
                    channel_id=channel_id,
                    log_time=i * 1000000,
                    data=f'{{"num": {i}}}'.encode(),
                    publish_time=i * 1000000,
                )

            writer.finish()

        # Read with small-mcap
        with open(temp_mcap_file, "rb") as f:
            messages = list(small_read_message(f))
            assert len(messages) == 10

            for i, (schema, channel, message) in enumerate(messages):
                # Verify complete schema fields
                assert schema.name == "TestSchema"
                assert schema.encoding == "json"
                assert schema.data == b'{"type": "object"}'
                # Verify complete channel fields
                assert channel.topic == "/official/topic"
                assert channel.message_encoding == "json"
                assert channel.schema_id == schema.id
                # Verify complete message fields
                assert message.channel_id == channel.id
                assert message.log_time == i * 1000000
                assert message.publish_time == i * 1000000
                assert message.data == f'{{"num": {i}}}'.encode()


@pytest.mark.compat
class TestCompression:
    """Test compression compatibility."""

    def test_zstd_small_to_official(self, temp_mcap_file: Path):
        """Test ZSTD compression from small-mcap to official."""
        # Write with small-mcap using ZSTD
        with open(temp_mcap_file, "wb") as f:
            writer = SmallMcapWriter(f, compression=CompressionType.ZSTD, chunk_size=512)
            writer.start()
            writer.add_schema(1, "Schema", "json", b"{}")
            writer.add_channel(1, "/topic", "json", 1)

            for i in range(20):
                writer.add_message(1, i * 1000, b"x" * 100, i * 1000)

            writer.finish()

        # Read with official mcap
        with open(temp_mcap_file, "rb") as f:
            reader = mcap_make_reader(f)
            messages = list(reader.iter_messages())
            assert len(messages) == 20
            # Verify at least one message has all fields
            schema, channel, message = messages[0]
            assert schema.name == "Schema"
            assert channel.topic == "/topic"
            assert message.data == b"x" * 100

    def test_lz4_small_to_official(self, temp_mcap_file: Path):
        """Test LZ4 compression from small-mcap to official."""
        # Write with small-mcap using LZ4
        with open(temp_mcap_file, "wb") as f:
            writer = SmallMcapWriter(f, compression=CompressionType.LZ4, chunk_size=512)
            writer.start()
            writer.add_schema(1, "Schema", "json", b"{}")
            writer.add_channel(1, "/topic", "json", 1)

            for i in range(20):
                writer.add_message(1, i * 1000, b"y" * 100, i * 1000)

            writer.finish()

        # Read with official mcap
        with open(temp_mcap_file, "rb") as f:
            reader = mcap_make_reader(f)
            messages = list(reader.iter_messages())
            assert len(messages) == 20
            # Verify at least one message has all fields
            schema, channel, message = messages[0]
            assert schema.name == "Schema"
            assert channel.topic == "/topic"
            assert message.data == b"y" * 100


@pytest.mark.compat
class TestMultipleChannels:
    """Test multiple channels compatibility."""

    def test_multiple_schemas_channels(self, temp_mcap_file: Path):
        """Test multiple schemas and channels."""
        # Write with small-mcap
        with open(temp_mcap_file, "wb") as f:
            writer = SmallMcapWriter(f)
            writer.start()

            writer.add_schema(1, "TypeA", "json", b'{"type": "A"}')
            writer.add_schema(2, "TypeB", "json", b'{"type": "B"}')

            writer.add_channel(1, "/topic/a", "json", 1)
            writer.add_channel(2, "/topic/b", "json", 2)

            for i in range(5):
                writer.add_message(1, i * 1000, b"a", i * 1000)
                writer.add_message(2, i * 1000 + 500, b"b", i * 1000 + 500)

            writer.finish()

        # Read with official mcap
        with open(temp_mcap_file, "rb") as f:
            reader = mcap_make_reader(f)
            messages = list(reader.iter_messages())
            assert len(messages) == 10

            # Verify we have both topics
            # iter_messages returns tuple (schema, channel, message)
            topics = {channel.topic for schema, channel, message in messages}
            assert topics == {"/topic/a", "/topic/b"}

            # Verify at least one message from each channel
            msg_a = next(m for m in messages if m[1].topic == "/topic/a")
            schema_a, channel_a, message_a = msg_a
            assert schema_a.name == "TypeA"
            assert channel_a.message_encoding == "json"
            assert message_a.data == b"a"

            msg_b = next(m for m in messages if m[1].topic == "/topic/b")
            schema_b, channel_b, message_b = msg_b
            assert schema_b.name == "TypeB"
            assert channel_b.message_encoding == "json"
            assert message_b.data == b"b"


@pytest.mark.compat
class TestAttachmentsMetadata:
    """Test attachments and metadata compatibility."""

    def test_attachments_compat(self, temp_mcap_file: Path):
        """Test attachments are compatible."""
        # Write with small-mcap
        with open(temp_mcap_file, "wb") as f:
            writer = SmallMcapWriter(f)
            writer.start()

            writer.add_attachment(
                log_time=1000000,
                create_time=1000000,
                name="config.yaml",
                media_type="application/yaml",
                data=b"version: 1.0\n",
            )

            writer.finish()

        # Read with official mcap
        with open(temp_mcap_file, "rb") as f:
            reader = mcap_make_reader(f)
            summary = reader.get_summary()

            stats = summary.statistics
            assert stats.message_count == 0
            assert stats.schema_count == 0
            assert stats.channel_count == 0
            assert stats.attachment_count == 1
            assert stats.metadata_count == 0
            assert stats.chunk_count >= 0

    def test_metadata_compat(self, temp_mcap_file: Path):
        """Test metadata is compatible."""
        # Write with small-mcap
        with open(temp_mcap_file, "wb") as f:
            writer = SmallMcapWriter(f)
            writer.start()

            writer.add_metadata("config", {"version": "1.0", "author": "test"})

            writer.finish()

        # Read with official mcap
        with open(temp_mcap_file, "rb") as f:
            reader = mcap_make_reader(f)
            summary = reader.get_summary()

            stats = summary.statistics
            assert stats.message_count == 0
            assert stats.schema_count == 0
            assert stats.channel_count == 0
            assert stats.attachment_count == 0
            assert stats.metadata_count == 1
            assert stats.chunk_count >= 0
