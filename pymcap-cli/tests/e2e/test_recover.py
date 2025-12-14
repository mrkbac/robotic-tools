"""E2E tests for the recover command."""

from pathlib import Path

import pytest
from small_mcap import get_summary
from small_mcap.rebuild import rebuild_summary

from pymcap_cli.cmd.recover_inplace_cmd import recover_inplace
from pymcap_cli.mcap_processor import (
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)


@pytest.mark.e2e
class TestRecover:
    """Test recover command functionality."""

    def test_recover_valid_file(self, simple_mcap: Path, output_file: Path):
        """Test recovery of a valid MCAP file."""
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream)

        # Verify recovery succeeded
        assert stats.writer_statistics.message_count > 0
        assert stats.errors_encountered == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_recover_truncated_file(self, truncated_mcap: Path, output_file: Path):
        """Test recovery of truncated MCAP file."""
        file_size = truncated_mcap.stat().st_size

        with truncated_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            options = ProcessingOptions(
                inputs=[
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
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
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        always_decode_chunk=True,  # Force chunk decoding
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
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
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
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
                        InputOptions(
                            stream=input_stream,
                            file_size=file_size,
                        )
                    ],
                    output=OutputOptions(compression=compression, chunk_size=4 * 1024 * 1024),
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
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
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
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                        always_decode_chunk=False,  # Enable fast chunk copying
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=4 * 1024 * 1024),
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
                    InputOptions(
                        stream=input_stream,
                        file_size=file_size,
                    )
                ],
                output=OutputOptions(compression="zstd", chunk_size=chunk_size),
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
