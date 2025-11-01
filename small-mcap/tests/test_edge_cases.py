"""Edge case tests for small-mcap - boundary conditions and unusual scenarios."""

from small_mcap import Channel, McapWriter, Message, get_header, get_summary, read_message


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
            # Should include messages at times: 3000, 4000, 5000, 6000
            assert len(messages) >= 3

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
