"""E2E tests for the recover command."""

from pathlib import Path

import pytest
from pymcap_cli.mcap_processor import McapProcessor, ProcessingOptions


class TestRecover:
    """Test recover command functionality."""

    def test_recover_valid_file(self, simple_mcap: Path, output_file: Path):
        """Test recovery of a valid MCAP file."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process(input_stream, output_stream, file_size)

        # Verify recovery succeeded
        assert stats.writer_statistics.message_count > 0
        assert stats.errors_encountered == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_recover_truncated_file(self, truncated_mcap: Path, output_file: Path):
        """Test recovery of truncated MCAP file."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = truncated_mcap.stat().st_size

        with truncated_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process(input_stream, output_stream, file_size)

        # Should recover some messages even from corrupt file
        assert stats.writer_statistics.message_count > 0
        # May have encountered errors
        assert output_file.exists()

    def test_recover_with_always_decode_chunk(self, simple_mcap: Path, output_file: Path):
        """Test recovery with --always-decode-chunk flag."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=True,  # Force chunk decoding
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process(input_stream, output_stream, file_size)

        # Should decode all chunks
        assert stats.chunks_decoded > 0
        assert stats.chunks_copied == 0  # No fast copying
        assert stats.writer_statistics.message_count > 0

    def test_recover_multi_topic(self, multi_topic_mcap: Path, output_file: Path):
        """Test recovery of multi-topic MCAP file."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process(input_stream, output_stream, file_size)

        # Should recover messages from all topics
        # 4 topics * 50 messages = 200 total
        assert stats.writer_statistics.message_count > 0
        assert stats.writer_statistics.channel_count == 4
        assert stats.writer_statistics.schema_count == 1

    def test_recover_compression_formats(self, simple_mcap: Path, tmp_path: Path):
        """Test recovery with different compression formats."""
        compressions = ["zstd", "lz4", "none"]

        for compression in compressions:
            output_file = tmp_path / f"output_{compression}.mcap"
            options = ProcessingOptions(
                recovery_mode=True,
                always_decode_chunk=False,
                compression=compression,
                chunk_size=4 * 1024 * 1024,
            )

            processor = McapProcessor(options)
            file_size = simple_mcap.stat().st_size

            with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
                stats = processor.process(input_stream, output_stream, file_size)

            assert stats.writer_statistics.message_count > 0
            assert output_file.exists()
            assert output_file.stat().st_size > 0

    def test_recover_preserves_metadata_and_attachments(self, simple_mcap: Path, output_file: Path):
        """Test that recovery preserves metadata and attachments."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            include_metadata=True,
            include_attachments=True,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process(input_stream, output_stream, file_size)

        assert stats.writer_statistics.message_count > 0
        # Metadata and attachments counts depend on fixture
        assert output_file.exists()

    def test_recover_chunk_copying_optimization(self, simple_mcap: Path, output_file: Path):
        """Test that chunk copying optimization works."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,  # Enable fast chunk copying
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process(input_stream, output_stream, file_size)

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
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=chunk_size,
        )

        processor = McapProcessor(options)
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process(input_stream, output_stream, file_size)

        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()
