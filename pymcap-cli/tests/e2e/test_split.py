"""E2E tests for the split command."""

# ruff: noqa: ARG001

from pathlib import Path

import pytest
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from pymcap_cli.core.processors import (
    DurationSplitProcessor,
    ExpressionSplitProcessor,
    TimestampSplitProcessor,
)
from small_mcap import stream_reader


def _write_mcap(data: bytes, path: Path) -> None:
    path.write_bytes(data)


def _read_mcap(path: Path) -> dict:
    """Read an MCAP file and return message counts per channel."""
    channel_topics = {}
    message_counts = {}
    with path.open("rb") as f:
        for record in stream_reader(f):
            if hasattr(record, "topic"):
                channel_topics[record.id] = record.topic
                message_counts[record.id] = 0
            elif hasattr(record, "channel_id"):
                message_counts[record.channel_id] = message_counts.get(record.channel_id, 0) + 1
    return {"channel_topics": channel_topics, "message_counts": message_counts}


@pytest.mark.e2e
class TestDurationSplit:
    """Test duration-based splitting."""

    def test_split_into_multiple_files(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that duration split creates multiple output files."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "output_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Verify multiple segments were created
        assert len(processor.output_manager.segments) > 1

        # Verify total messages match
        assert stats.writer_statistics.message_count > 0

    def test_split_preserves_all_messages(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that all messages are preserved across split files."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[DurationSplitProcessor(duration_ns=50_000_000)],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Count messages in each output file
        total_messages = 0
        for segment in processor.output_manager.segments.values():
            total_messages += segment.writer.statistics.message_count

        # All messages should be preserved
        assert total_messages == stats.writer_statistics.message_count

    def test_split_file_naming(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that output files are named correctly."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "output_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            processor.process(output_stream=None)

        # Verify files exist with expected naming
        for segment in processor.output_manager.segments.values():
            assert Path(segment.path).exists()


@pytest.mark.e2e
class TestTimestampSplit:
    """Test timestamp-based splitting."""

    def test_split_at_timestamps(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test splitting at specific timestamps."""
        file_size = multi_topic_mcap.stat().st_size

        # Split at 50ms and 100ms (50_000_000 and 100_000_000 ns)
        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[TimestampSplitProcessor(split_points=[50_000_000, 100_000_000])],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Should create 3 segments (before 50ms, 50-100ms, after 100ms)
        assert len(processor.output_manager.segments) == 3
        assert stats.writer_statistics.message_count > 0

    def test_combined_duration_and_timestamp_split(self, multi_topic_mcap: Path, tmp_path: Path):
        """Combined split processors should route by the intersection of both boundary sets."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[
                        DurationSplitProcessor(duration_ns=100_000_000),
                        TimestampSplitProcessor(split_points=[125_000_000]),
                    ],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        assert len(processor.output_manager.segments) == 3
        assert (
            sum(
                segment.writer.statistics.message_count
                for segment in processor.output_manager.segments.values()
            )
            == stats.writer_statistics.message_count
        )


@pytest.mark.e2e
class TestExpressionSplit:
    """Test expression-based splitting."""

    def test_split_by_channel(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test splitting messages by channel ID."""
        file_size = multi_topic_mcap.stat().st_size

        def split_by_channel(message, channels):
            return message.channel_id

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[ExpressionSplitProcessor(split_by_channel)],
                    output_template=str(tmp_path / "channel_{key}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Should create one segment per channel
        assert len(processor.output_manager.segments) > 1
        assert stats.writer_statistics.message_count > 0

    def test_split_by_channel_preserves_messages(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that expression split preserves all messages."""
        file_size = multi_topic_mcap.stat().st_size

        def split_by_channel(message, channels):
            return message.channel_id

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[ExpressionSplitProcessor(split_by_channel)],
                    output_template=str(tmp_path / "ch_{key}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Count messages across all segments
        total = sum(
            s.writer.statistics.message_count for s in processor.output_manager.segments.values()
        )
        assert total == stats.writer_statistics.message_count


@pytest.mark.e2e
class TestSplitWithRechunking:
    """Test splitting combined with rechunking."""

    def test_split_and_rechunk(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that splitting works with rechunking active."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                    rechunk_strategy="all",
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Should create multiple segments with rechunking
        assert len(processor.output_manager.segments) > 1
        assert stats.writer_statistics.message_count > 0


@pytest.mark.e2e
class TestSplitWithFiltering:
    """Test splitting combined with topic/time filtering."""

    def test_split_with_topic_filter(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that splitting works with topic filtering."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(include_topic_regex=["/camera/.*"]),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    processors=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "camera_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=4 * 1024 * 1024,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Should only include camera topics
        assert stats.writer_statistics.channel_count <= 2
        assert stats.writer_statistics.message_count > 0
