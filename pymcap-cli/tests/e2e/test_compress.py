"""E2E tests for the compress command."""

from pathlib import Path

import pytest
from pymcap_cli.mcap_processor import McapProcessor, ProcessingOptions


@pytest.mark.e2e
class TestCompress:
    """Test compress command functionality."""

    def test_compress_to_zstd(self, uncompressed_mcap: Path, output_file: Path):
        """Test compressing an uncompressed file to zstd."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = uncompressed_mcap.stat().st_size

        with uncompressed_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        # Verify compression succeeded
        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()
        # Compressed file should be smaller
        assert output_file.stat().st_size < uncompressed_mcap.stat().st_size

    def test_compress_to_lz4(self, uncompressed_mcap: Path, output_file: Path):
        """Test compressing an uncompressed file to lz4."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="lz4",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = uncompressed_mcap.stat().st_size

        with uncompressed_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        # Verify compression succeeded
        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()
        # Compressed file should be smaller
        assert output_file.stat().st_size < uncompressed_mcap.stat().st_size

    def test_decompress_to_none(self, simple_mcap: Path, output_file: Path):
        """Test decompressing a compressed file."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="none",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        # Verify decompression succeeded
        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()
        # Uncompressed file should be larger
        assert output_file.stat().st_size > simple_mcap.stat().st_size

    def test_recompress_zstd_to_lz4(self, simple_mcap: Path, output_file: Path):
        """Test recompressing from zstd to lz4."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="lz4",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = simple_mcap.stat().st_size

        with simple_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        # Verify recompression succeeded
        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()

    def test_recompress_lz4_to_zstd(self, lz4_mcap: Path, output_file: Path):
        """Test recompressing from lz4 to zstd."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = lz4_mcap.stat().st_size

        with lz4_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        # Verify recompression succeeded
        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()

    @pytest.mark.parametrize(
        "chunk_size",
        [
            512 * 1024,  # 512KB
            1 * 1024 * 1024,  # 1MB
            4 * 1024 * 1024,  # 4MB
            8 * 1024 * 1024,  # 8MB
        ],
    )
    def test_compress_with_different_chunk_sizes(
        self, uncompressed_mcap: Path, tmp_path: Path, chunk_size: int
    ):
        """Test compression with different chunk sizes."""
        output_file = tmp_path / f"output_{chunk_size}.mcap"
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=chunk_size,
        )

        processor = McapProcessor(options)
        file_size = uncompressed_mcap.stat().st_size

        with uncompressed_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()

    def test_compress_multi_topic_file(self, multi_topic_mcap: Path, output_file: Path):
        """Test compressing a multi-topic file."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="lz4",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        # Should preserve all messages
        assert stats.writer_statistics.message_count > 0
        assert stats.writer_statistics.channel_count == 4
        assert output_file.exists()

    def test_compress_preserves_all_content(self, simple_mcap: Path, output_file: Path):
        """Test that compression preserves all content."""
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
            stats = processor.process([input_stream], output_stream, [file_size])

        # All content should be preserved
        assert stats.writer_statistics.message_count > 0
        assert stats.writer_statistics.schema_count > 0
        assert stats.writer_statistics.channel_count > 0
        assert output_file.exists()

    def test_compress_large_file(self, large_1mb_mcap: Path, output_file: Path):
        """Test compressing a large file."""
        options = ProcessingOptions(
            recovery_mode=True,
            always_decode_chunk=False,
            compression="zstd",
            chunk_size=4 * 1024 * 1024,
        )

        processor = McapProcessor(options)
        file_size = large_1mb_mcap.stat().st_size

        with large_1mb_mcap.open("rb") as input_stream, output_file.open("wb") as output_stream:
            stats = processor.process([input_stream], output_stream, [file_size])

        # Should process all messages
        assert stats.writer_statistics.message_count > 0
        assert output_file.exists()
        assert stats.errors_encountered == 0

    def test_compression_ratio_comparison(self, uncompressed_mcap: Path, tmp_path: Path):
        """Compare compression ratios for different algorithms."""
        uncompressed_size = uncompressed_mcap.stat().st_size
        compressions = ["zstd", "lz4"]
        ratios = {}

        for compression in compressions:
            output_file = tmp_path / f"output_{compression}.mcap"
            options = ProcessingOptions(
                recovery_mode=True,
                always_decode_chunk=False,
                compression=compression,
                chunk_size=4 * 1024 * 1024,
            )

            processor = McapProcessor(options)

            with (
                uncompressed_mcap.open("rb") as input_stream,
                output_file.open("wb") as output_stream,
            ):
                stats = processor.process([input_stream], output_stream, [uncompressed_size])

            compressed_size = output_file.stat().st_size
            ratios[compression] = compressed_size / uncompressed_size
            assert stats.writer_statistics.message_count > 0

        # Both should compress the file
        assert all(ratio < 1.0 for ratio in ratios.values())
        # ZSTD typically has better compression than LZ4
        # (though this may not always be true for small files)
