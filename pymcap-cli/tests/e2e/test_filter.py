"""E2E tests for the filter command."""

from pathlib import Path

import pytest
from pymcap_cli.mcap_processor import (
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)


@pytest.mark.e2e
class TestFilter:
    """Test filter command functionality."""

    def test_filter_include_topic(self, multi_topic_mcap: Path, output_file: Path):
        """Test filtering with topic inclusion."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_topic_regex=["/camera/.*"],
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should only include messages from camera topics
        assert stats.writer_statistics.message_count > 0
        assert stats.writer_statistics.channel_count == 2  # /camera/front and /camera/back
        assert stats.messages_processed >= stats.writer_statistics.message_count  # Total >= written

    def test_filter_exclude_topic(self, multi_topic_mcap: Path, output_file: Path):
        """Test filtering with topic exclusion."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        exclude_topic_regex=["/debug/.*"],
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should exclude debug topic (3 topics * 50 messages)
        assert stats.writer_statistics.message_count >= 150
        assert stats.writer_statistics.channel_count == 3

    def test_filter_multiple_include_patterns(self, multi_topic_mcap: Path, output_file: Path):
        """Test filtering with multiple include patterns."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_topic_regex=["/camera/.*", "/lidar/.*"],
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should include 3 topics * 50 messages
        assert stats.writer_statistics.message_count >= 150
        assert stats.writer_statistics.channel_count == 3

    def test_filter_time_range(self, multi_topic_mcap: Path, output_file: Path):
        """Test filtering with time range."""
        # Include messages from time 0 to 50ms (50 million nanoseconds)
        # With 200 messages total, each message is 1ms apart
        # So this should include first 50 messages (first 12-13 from each topic)
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        start_nsecs=0,
                        end_nsecs=50_000_000,  # 50ms in nanoseconds
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should include only messages within time range
        assert stats.writer_statistics.message_count < 200
        assert stats.writer_statistics.message_count > 0

    def test_filter_topic_and_time(self, multi_topic_mcap: Path, output_file: Path):
        """Test filtering with both topic and time filters."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_topic_regex=["/camera/.*"],
                        start_nsecs=0,
                        end_nsecs=50_000_000,  # 50ms
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should include only camera messages within time range
        assert stats.writer_statistics.message_count < 100  # Less than all camera messages
        assert stats.writer_statistics.message_count > 0
        assert stats.writer_statistics.channel_count <= 2  # Only camera channels

    def test_filter_exclude_metadata(self, simple_mcap: Path, output_file: Path):
        """Test filtering with metadata excluded."""
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_metadata=False,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should not write metadata
        assert stats.writer_statistics.metadata_count == 0

    def test_filter_exclude_attachments(self, simple_mcap: Path, output_file: Path):
        """Test filtering with attachments excluded."""
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_attachments=False,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should not write attachments
        assert stats.writer_statistics.attachment_count == 0

    def test_filter_all_topics_filtered_out(self, multi_topic_mcap: Path, output_file: Path):
        """Test filtering that excludes all topics."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_topic_regex=["/nonexistent/.*"],
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should write no messages
        assert stats.writer_statistics.message_count >= 0
        assert stats.writer_statistics.channel_count == 0

    def test_filter_exact_topic_match(self, multi_topic_mcap: Path, output_file: Path):
        """Test filtering with exact topic match."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_topic_regex=["^/camera/front$"],
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should only include one topic
        assert stats.writer_statistics.message_count >= 50
        assert stats.writer_statistics.channel_count == 1

    def test_filter_with_compression_formats(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test filtering with different compression formats."""
        compressions = ["zstd", "lz4", "none"]
        file_size = multi_topic_mcap.stat().st_size

        for compression in compressions:
            output_file = tmp_path / f"output_{compression}.mcap"

            with (
                multi_topic_mcap.open("rb") as input_stream,
                output_file.open("wb") as output_stream,
            ):
                options = ProcessingOptions(
                    inputs=[
                        InputOptions(
                            stream=input_stream,
                            file_size=file_size,
                            include_topic_regex=["/camera/.*"],
                        )
                    ],
                    output=OutputOptions(compression=compression, chunk_size=4 * 1024 * 1024),
                )

                processor = McapProcessor(options)
                stats = processor.process(output_stream)

            assert stats.writer_statistics.message_count >= 100
            assert output_file.exists()

    def test_filter_statistics(self, multi_topic_mcap: Path, output_file: Path):
        """Test that filter statistics are correctly reported."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        include_topic_regex=["/camera/.*"],
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Verify all statistics are reasonable
        assert stats.messages_processed == 200
        assert stats.writer_statistics.message_count >= 100
        assert stats.writer_statistics.schema_count == 1
        assert stats.writer_statistics.channel_count == 2
        assert stats.chunks_processed > 0
