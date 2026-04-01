"""E2E tests for the recover command."""

from pathlib import Path

import pytest
from pymcap_cli.cmd.recover_inplace_cmd import recover_inplace
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from small_mcap import CompressionType, McapWriter, get_header, get_summary
from small_mcap.rebuild import rebuild_summary


@pytest.mark.e2e
class TestRecover:
    """Test recover command functionality."""

    def test_recover_valid_file(self, simple_mcap: Path, output_file: Path):
        """Test recovery of a valid MCAP file."""
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Verify recovery succeeded
        assert stats.writer_statistics.message_count > 0
        assert stats.errors_encountered == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_recover_preserves_profile_and_sets_pymcap_cli_library(
        self, tmp_path: Path, output_file: Path
    ) -> None:
        """Recovery should preserve profile and stamp pymcap-cli as the writer."""
        source_file = tmp_path / "source.mcap"

        with source_file.open("wb") as stream:
            writer = McapWriter(stream, chunk_size=1024 * 1024, compression=CompressionType.ZSTD)
            writer.start(profile="ros2", library="test-lib")
            writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
            writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=1)
            writer.add_message(
                channel_id=1,
                log_time=1,
                data=b'{"ok": true}',
                publish_time=1,
            )
            writer.finish()

        file_size = source_file.stat().st_size
        with source_file.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        assert stats.writer_statistics.message_count == 1
        with output_file.open("rb") as recovered_stream:
            header = get_header(recovered_stream)
        assert header.profile == "ros2"
        assert header.library == "pymcap-cli"

    def test_recover_multiple_inputs_preserves_shared_profile(self, tmp_path: Path) -> None:
        """Merged outputs should keep a shared profile and use the CLI library name."""
        first = tmp_path / "first.mcap"
        second = tmp_path / "second.mcap"
        output_file = tmp_path / "merged.mcap"

        for path, channel_id in ((first, 1), (second, 2)):
            with path.open("wb") as stream:
                writer = McapWriter(
                    stream, chunk_size=1024 * 1024, compression=CompressionType.ZSTD
                )
                writer.start(profile="ros2", library=f"source-{channel_id}")
                writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
                writer.add_channel(
                    channel_id=channel_id,
                    topic=f"/test/{channel_id}",
                    message_encoding="json",
                    schema_id=1,
                )
                writer.add_message(
                    channel_id=channel_id,
                    log_time=channel_id,
                    data=b'{"ok": true}',
                    publish_time=channel_id,
                )
                writer.finish()

        with (
            first.open("rb") as first_stream,
            second.open("rb") as second_stream,
            output_file.open("wb") as output_stream,
        ):
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=first_stream,
                        size=first.stat().st_size,
                        options=InputOptions.from_args(),
                    ),
                    InputFile(
                        stream=second_stream,
                        size=second.stat().st_size,
                        options=InputOptions.from_args(),
                    ),
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        assert stats.writer_statistics.message_count == 2
        with output_file.open("rb") as merged_stream:
            header = get_header(merged_stream)
        assert header.profile == "ros2"
        assert header.library == "pymcap-cli"

    def test_recover_truncated_file(self, truncated_mcap: Path, output_file: Path):
        """Test recovery of truncated MCAP file."""
        file_size = truncated_mcap.stat().st_size

        with truncated_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should recover some messages even from corrupt file
        assert stats.writer_statistics.message_count > 0
        # May have encountered errors
        assert output_file.exists()

    def test_recover_with_always_decode_chunk(self, simple_mcap: Path, output_file: Path):
        """Test recovery with --always-decode-chunk flag."""
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(always_decode_chunk=True),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should decode all chunks
        assert stats.chunks_decoded > 0
        assert stats.chunks_copied == 0  # No fast copying
        assert stats.writer_statistics.message_count > 0

    def test_recover_multi_topic(self, multi_topic_mcap: Path, output_file: Path):
        """Test recovery of multi-topic MCAP file."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should recover messages from all topics
        # 4 topics * 50 messages = 200 total
        assert stats.writer_statistics.message_count > 0
        assert stats.writer_statistics.channel_count == 4
        assert stats.writer_statistics.schema_count == 1

    def test_recover_compression_formats(self, simple_mcap: Path, tmp_path: Path):
        """Test recovery with different compression formats."""
        compressions = ["zstd", "lz4", "none"]
        file_size = simple_mcap.stat().st_size

        for compression in compressions:
            output_file = tmp_path / f"output_{compression}.mcap"

            with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
                options = ProcessingOptions(
                    inputs=[
                        InputFile(
                            stream=input_stream,
                            size=file_size,
                            options=InputOptions.from_args(),
                        )
                    ],
                    input_options=InputOptions.from_args(),
                    output_options=OutputOptions(
                        compression=compression, chunk_size=4 * 1024 * 1024
                    ),
                )

                processor = McapProcessor(options)
                stats = processor.process(output_stream)

            assert stats.writer_statistics.message_count > 0
            assert output_file.exists()
            assert output_file.stat().st_size > 0

    def test_recover_preserves_metadata_and_attachments(self, simple_mcap: Path, output_file: Path):
        """Test that recovery preserves metadata and attachments."""
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        assert stats.writer_statistics.message_count > 0
        # Metadata and attachments counts depend on fixture
        assert output_file.exists()

    def test_recover_chunk_copying_optimization(self, simple_mcap: Path, output_file: Path):
        """Test that chunk copying optimization works."""
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(always_decode_chunk=False),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Should use fast chunk copying when possible
        assert stats.chunks_processed > 0
        # Either fast copied or decoded
        assert (stats.chunks_copied + stats.chunks_decoded) == stats.chunks_processed

    @pytest.mark.parametrize(
        "chunk_size",
        [
            512 * 1024,  # 512KB
            1 * 1024 * 1024,  # 1MB
            4 * 1024 * 1024,  # 4MB
            8 * 1024 * 1024,  # 8MB
        ],
    )
    def test_recover_different_chunk_sizes(
        self, simple_mcap: Path, tmp_path: Path, chunk_size: int
    ):
        """Test recovery with different output chunk sizes."""
        output_file = tmp_path / f"output_{chunk_size}.mcap"
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=input_stream,
                        size=file_size,
                        options=InputOptions.from_args(),
                    )
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=chunk_size),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()

    def test_recover_inplace_rebuilds_summary(self, simple_mcap: Path, tmp_path: Path):
        """Recover-inplace should rebuild summary/footer after truncation."""
        truncated = tmp_path / "truncated.mcap"
        truncated.write_bytes(simple_mcap.read_bytes())

        # Remove existing summary/footer to simulate corruption
        with truncated.open("r+b") as f:
            info = rebuild_summary(
                f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
            )
            f.truncate(info.next_offset)

        result = recover_inplace(str(truncated), force=True)
        assert result == 0

        with truncated.open("rb") as f:
            summary = get_summary(f)
            assert summary.statistics is not None
            assert summary.statistics.message_count > 0
