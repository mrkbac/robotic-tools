"""Integration tests for small-mcap writer and reader."""

import io
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

    def test_reverse_read_single_channel(self, temp_mcap_file: Path):
        """Test reading messages in reverse order with a single channel (optimized path)."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=1024)
            writer.start()
            writer.add_schema(1, "Test", "json", b"{}")
            writer.add_channel(1, "/test", "json", 1)

            for i in range(10):
                writer.add_message(1, i * 1000, f"msg{i}".encode(), i * 1000)

            writer.finish()

        # Read forward
        with open(temp_mcap_file, "rb") as f:
            forward_msgs = list(read_message(f))

        # Read reverse
        with open(temp_mcap_file, "rb") as f:
            reverse_msgs = list(read_message(f, reverse=True))

        assert len(forward_msgs) == 10
        assert len(reverse_msgs) == 10

        # Verify reverse order
        for i, (_schema, _channel, message) in enumerate(reverse_msgs):
            expected_idx = 9 - i
            assert message.log_time == expected_idx * 1000
            assert message.data == f"msg{expected_idx}".encode()

    def test_reverse_read_multiple_channels(self, temp_mcap_file: Path):
        """Test reading messages in reverse order with multiple channels (heap path)."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=4096)
            writer.start()
            writer.add_schema(1, "Test", "json", b"{}")
            writer.add_channel(1, "/a", "json", 1)
            writer.add_channel(2, "/b", "json", 1)

            # Interleave messages from both channels
            for i in range(10):
                writer.add_message(1, i * 1000, f"a{i}".encode(), i * 1000)
                writer.add_message(2, i * 1000 + 500, f"b{i}".encode(), i * 1000 + 500)

            writer.finish()

        # Read forward
        with open(temp_mcap_file, "rb") as f:
            forward_msgs = list(read_message(f))

        # Read reverse
        with open(temp_mcap_file, "rb") as f:
            reverse_msgs = list(read_message(f, reverse=True))

        assert len(forward_msgs) == 20
        assert len(reverse_msgs) == 20

        # Verify messages are in descending log_time order
        prev_time = float("inf")
        for _schema, _channel, message in reverse_msgs:
            assert message.log_time <= prev_time
            prev_time = message.log_time

        # Verify reverse order matches reversed forward order
        forward_times = [msg.log_time for _, _, msg in forward_msgs]
        reverse_times = [msg.log_time for _, _, msg in reverse_msgs]
        assert reverse_times == list(reversed(forward_times))


class TestMultiStreamMerging:
    """Test reading from multiple MCAP streams."""

    def _create_mcap(self, messages: list[tuple[int, bytes]], topic: str = "/test") -> bytes:
        """Create an MCAP file with given messages."""
        buffer = io.BytesIO()
        writer = McapWriter(buffer)
        writer.start()
        writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")
        writer.add_channel(channel_id=1, topic=topic, message_encoding="raw", schema_id=1)
        for log_time, data in messages:
            writer.add_message(channel_id=1, log_time=log_time, publish_time=log_time, data=data)
        writer.finish()
        return buffer.getvalue()

    def test_merge_two_streams(self):
        """read_message should merge multiple streams in time order."""
        mcap1 = self._create_mcap([(1000, b"a"), (3000, b"c")], topic="/t1")
        mcap2 = self._create_mcap([(2000, b"b"), (4000, b"d")], topic="/t2")

        streams = [io.BytesIO(mcap1), io.BytesIO(mcap2)]
        results = list(read_message(streams))

        assert len(results) == 4
        times = [r[2].log_time for r in results]
        assert times == [1000, 2000, 3000, 4000]

    def test_merge_streams_reverse(self):
        """Merged streams should support reverse order."""
        mcap1 = self._create_mcap([(1000, b"a"), (3000, b"c")])
        mcap2 = self._create_mcap([(2000, b"b"), (4000, b"d")])

        streams = [io.BytesIO(mcap1), io.BytesIO(mcap2)]
        results = list(read_message(streams, reverse=True))

        assert len(results) == 4
        times = [r[2].log_time for r in results]
        assert times == [4000, 3000, 2000, 1000]

    def test_merge_with_id_conflicts(self):
        """Streams with conflicting channel IDs should be remapped."""
        # Both use channel_id=1 but different topics
        mcap1 = self._create_mcap([(1000, b"a")], topic="/topic1")
        mcap2 = self._create_mcap([(2000, b"b")], topic="/topic2")

        streams = [io.BytesIO(mcap1), io.BytesIO(mcap2)]
        results = list(read_message(streams))

        assert len(results) == 2
        topics = {r[1].topic for r in results}
        assert topics == {"/topic1", "/topic2"}
