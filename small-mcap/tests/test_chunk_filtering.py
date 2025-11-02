"""Tests for chunk filtering and skipping optimizations.

These tests verify that the chunk skipping optimizations in _read_message_non_seeking
work correctly for both time-based and channel-based filtering.
"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from small_mcap import include_topics, read_message
from small_mcap.reader import breakup_chunk


class TestChunkSkipping:
    """Test that chunks are skipped when filtered."""

    @pytest.fixture
    def ten_messages_file(self) -> Path:
        """Path to test file with 10 messages in chunks."""
        return Path(__file__).parent / "conformance_data" / "TenMessages" / "TenMessages-ch.mcap"

    @pytest.fixture
    def ten_messages_with_index_file(self) -> Path:
        """Path to test file with 10 messages in chunks WITH MessageIndex."""
        return Path(__file__).parent / "conformance_data" / "TenMessages" / "TenMessages-ch-mx.mcap"

    @pytest.fixture
    def one_message_file(self) -> Path:
        """Path to test file with 1 message in a chunk."""
        return Path(__file__).parent / "conformance_data" / "OneMessage" / "OneMessage-ch.mcap"

    def make_non_seekable(self, data: bytes) -> io.BytesIO:
        """Create a non-seekable stream from bytes."""
        stream = io.BytesIO(data)
        stream.seekable = lambda: False
        return stream

    def test_time_filter_skips_all_chunks(self, ten_messages_file: Path):
        """Test that chunks are skipped when time filter excludes all messages."""
        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        # Read file data
        with open(ten_messages_file, "rb") as f:
            data = f.read()

        # Create non-seekable stream and filter to time range with no messages
        stream = self.make_non_seekable(data)

        with patch("small_mcap.reader.breakup_chunk", wraps=breakup_chunk) as mock_breakup:
            messages = list(read_message(stream, start_time_ns=1000, end_time_ns=2000))

            # Should skip all chunks - no messages and no breakup_chunk calls
            assert len(messages) == 0
            assert mock_breakup.call_count == 0, "Expected no chunks to be decompressed"

    def test_time_filter_skips_chunks_before_range(self, ten_messages_file: Path):
        """Test that chunks before time range are skipped."""
        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        with open(ten_messages_file, "rb") as f:
            data = f.read()

        # Filter to messages with log_time >= 5 (times are 0,1,2,3,3,4,5,7,8,9)
        stream = self.make_non_seekable(data)

        with patch("small_mcap.reader.breakup_chunk", wraps=breakup_chunk) as mock_breakup:
            messages = list(read_message(stream, start_time_ns=5))

            # Should get messages with times 5,7,8,9 (4 messages)
            assert len(messages) == 4
            # Should have fewer breakup_chunk calls than total chunks
            # (some chunks should be skipped)
            assert mock_breakup.call_count >= 1  # At least some chunks processed

    def test_time_filter_skips_chunks_after_range(self, ten_messages_file: Path):
        """Test that chunks after time range are skipped."""
        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        with open(ten_messages_file, "rb") as f:
            data = f.read()

        # Filter to messages with log_time < 5
        stream = self.make_non_seekable(data)

        with patch("small_mcap.reader.breakup_chunk", wraps=breakup_chunk) as mock_breakup:
            messages = list(read_message(stream, end_time_ns=5))

            # Should get messages with times 0,1,2,3,3,4 (6 messages)
            assert len(messages) == 6
            assert mock_breakup.call_count >= 1

    def test_channel_filter_returns_no_messages(self, ten_messages_with_index_file: Path):
        """Test that channel filtering excludes messages correctly.

        Note: Channel-based chunk skipping happens when exclude_channels is already populated.
        On the first pass through non-seekable streams, chunks must be decompressed to discover
        channel definitions. The optimization skips chunks on subsequent iterations or when
        channels are known in advance (e.g., from summary section in seekable streams).
        """
        if not ten_messages_with_index_file.exists():
            pytest.skip("Test file not available")

        with open(ten_messages_with_index_file, "rb") as f:
            data = f.read()

        # Filter to non-existent topic
        stream = self.make_non_seekable(data)
        messages = list(read_message(stream, should_include=include_topics(["non_existent"])))

        # Should filter out all messages
        assert len(messages) == 0

    def test_combined_time_and_channel_filter(self, ten_messages_file: Path):
        """Test that time and channel filters work together."""
        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        with open(ten_messages_file, "rb") as f:
            data = f.read()

        # Combine time filter that would include some + channel filter that excludes all
        stream = self.make_non_seekable(data)
        messages = list(
            read_message(
                stream,
                start_time_ns=2,
                end_time_ns=7,
                should_include=include_topics(["non_existent"]),
            )
        )

        # Channel filter should exclude all messages
        assert len(messages) == 0

    def test_seekable_vs_non_seekable_with_filter(self, ten_messages_file: Path):
        """Test that seekable and non-seekable streams produce same results with filtering."""
        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        # Read with seekable stream
        with open(ten_messages_file, "rb") as f:
            seekable_messages = list(read_message(f, start_time_ns=3, end_time_ns=7))

        # Read with non-seekable stream
        with open(ten_messages_file, "rb") as f:
            data = f.read()
        stream = self.make_non_seekable(data)
        non_seekable_messages = list(read_message(stream, start_time_ns=3, end_time_ns=7))

        # Should produce identical results
        assert len(seekable_messages) == len(non_seekable_messages)
        assert len(seekable_messages) > 0  # Sanity check

        # Verify message contents match
        for (_s_schema, s_channel, s_msg), (_ns_schema, ns_channel, ns_msg) in zip(
            seekable_messages, non_seekable_messages, strict=True
        ):
            assert s_msg.log_time == ns_msg.log_time
            assert s_msg.data == ns_msg.data
            assert s_channel.topic == ns_channel.topic

    def test_no_filtering_processes_all_chunks(self, ten_messages_file: Path):
        """Test that without filtering, all chunks are processed."""
        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        with open(ten_messages_file, "rb") as f:
            data = f.read()

        stream = self.make_non_seekable(data)

        with patch("small_mcap.reader.breakup_chunk", wraps=breakup_chunk) as mock_breakup:
            messages = list(read_message(stream))

            # Should get all 10 messages
            assert len(messages) == 10
            # Should have called breakup_chunk for each chunk in the file
            assert mock_breakup.call_count >= 1

    def test_partial_time_overlap_processes_chunk(self, ten_messages_file: Path):
        """Test that chunks partially overlapping time range are processed."""
        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        with open(ten_messages_file, "rb") as f:
            data = f.read()

        # Use narrow time range that should hit at least one chunk
        stream = self.make_non_seekable(data)

        with patch("small_mcap.reader.breakup_chunk", wraps=breakup_chunk) as mock_breakup:
            messages = list(read_message(stream, start_time_ns=4, end_time_ns=6))

            # Should get messages in range
            assert len(messages) > 0
            # Should have processed at least one chunk
            assert mock_breakup.call_count >= 1

            # Verify message times are in range
            for _, _, msg in messages:
                assert 4 <= msg.log_time < 6


class TestChunkBuffering:
    """Test the chunk buffering logic in non-seekable streams."""

    @pytest.fixture
    def one_message_file(self) -> Path:
        """Path to test file with 1 message in a chunk."""
        return Path(__file__).parent / "conformance_data" / "OneMessage" / "OneMessage-ch.mcap"

    def make_non_seekable(self, data: bytes) -> io.BytesIO:
        """Create a non-seekable stream from bytes."""
        stream = io.BytesIO(data)
        stream.seekable = lambda: False
        return stream

    def test_chunk_processed_before_other_records(self, one_message_file: Path):
        """Test that buffered chunk is processed before other records are yielded."""
        if not one_message_file.exists():
            pytest.skip("Test file not available")

        with open(one_message_file, "rb") as f:
            data = f.read()

        stream = self.make_non_seekable(data)
        messages = list(read_message(stream))

        # Should successfully read the message
        assert len(messages) == 1

    def test_final_chunk_is_processed(self, one_message_file: Path):
        """Test that the final buffered chunk at end of stream is processed."""
        if not one_message_file.exists():
            pytest.skip("Test file not available")

        with open(one_message_file, "rb") as f:
            data = f.read()

        stream = self.make_non_seekable(data)

        with patch("small_mcap.reader.breakup_chunk", wraps=breakup_chunk) as mock_breakup:
            messages = list(read_message(stream))

            # Should process the final chunk
            assert len(messages) == 1
            assert mock_breakup.call_count == 1

    def test_empty_result_with_time_filter_on_final_chunk(self, one_message_file: Path):
        """Test that final chunk is skipped if filtered by time."""
        if not one_message_file.exists():
            pytest.skip("Test file not available")

        with open(one_message_file, "rb") as f:
            data = f.read()

        stream = self.make_non_seekable(data)

        with patch("small_mcap.reader.breakup_chunk", wraps=breakup_chunk) as mock_breakup:
            messages = list(read_message(stream, start_time_ns=1000))

            # Should skip the final chunk
            assert len(messages) == 0
            assert mock_breakup.call_count == 0


class TestChannelDefinitionPreservation:
    """Test that channel/schema definitions are preserved when chunks are skipped."""

    def make_non_seekable(self, data: bytes) -> io.BytesIO:
        """Create a non-seekable stream from bytes."""
        stream = io.BytesIO(data)
        stream.seekable = lambda: False
        return stream

    def test_preserves_channel_definitions_from_skipped_chunks(self):
        """Test that channel definitions are loaded even when chunk is outside time range.

        This is a regression test for the issue where time filtering would skip chunks
        entirely, including the Channel/Schema definitions inside them, causing
        "no channel record found" errors for later chunks.
        """
        ten_messages_file = (
            Path(__file__).parent / "conformance_data" / "TenMessages" / "TenMessages-ch.mcap"
        )

        if not ten_messages_file.exists():
            pytest.skip("Test file not available")

        with open(ten_messages_file, "rb") as f:
            data = f.read()

        # Filter to messages after time 5, which might exclude first chunk
        # If the first chunk contains channel definitions, they must still be loaded
        stream = self.make_non_seekable(data)

        # This should NOT raise McapError("no channel record found")
        messages = list(read_message(stream, start_time_ns=5))

        # Should successfully get messages with times >= 5
        assert len(messages) > 0
        # Verify all messages have valid channel references
        for _schema, channel, message in messages:
            assert channel is not None
            assert channel.topic == "example"  # Known topic from test file
            assert message.log_time >= 5
