"""Tests for small_mcap.writer module."""

from pathlib import Path

import pytest
from small_mcap import (
    CompressionType,
    IndexType,
    McapWriter,
    MessageEncoding,
    SchemaEncoding,
    WriterNotStartedError,
    get_summary,
    read_message,
)


class TestMcapWriterBasics:
    """Test basic McapWriter functionality."""

    def test_write_minimal_file(self, temp_mcap_file: Path):
        """Test writing a minimal MCAP file."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()
            writer.finish()

        # Verify file was created and has basic structure
        assert temp_mcap_file.exists()
        assert temp_mcap_file.stat().st_size > 0

    def test_write_with_schema_and_channel(self, temp_mcap_file: Path):
        """Test writing a file with schema and channel."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start(profile="test", library="small-mcap-test")

            writer.add_schema(
                schema_id=1,
                name="TestSchema",
                encoding=SchemaEncoding.Protobuf,
                data=b"message Test { int32 value = 1; }",
            )

            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding=MessageEncoding.Protobuf,
                schema_id=1,
            )

            writer.finish()

    def test_write_messages(self, temp_mcap_file: Path):
        """Test writing messages to a file."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_schema(
                schema_id=1,
                name="Test",
                encoding=SchemaEncoding.Protobuf,
                data=b"message Test {}",
            )
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding=MessageEncoding.Protobuf,
                schema_id=1,
            )

            # Write multiple messages
            for i in range(10):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1000000000,
                    data=f"message_{i}".encode(),
                    publish_time=i * 1000000000,
                )

            writer.finish()

        # Verify we can read the messages back
        with open(temp_mcap_file, "rb") as f:
            messages = list(read_message(f))
            assert len(messages) == 10

    def test_write_without_start_raises_error(self, temp_mcap_file: Path):
        """Test that writing without calling start() raises WriterNotStartedError."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            with pytest.raises(WriterNotStartedError, match="Writer not started"):
                writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")

    def test_write_after_finish_raises_error(self, temp_mcap_file: Path):
        """Test that writing after finish() raises an error."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()
            writer.finish()

            with pytest.raises(RuntimeError, match="Writer already finished"):
                writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")


class TestMcapWriterChunking:
    """Test chunking functionality."""

    def test_chunked_mode(self, temp_mcap_file: Path):
        """Test writing in chunked mode."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=1024)  # Small chunk size
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            # Write enough messages to create multiple chunks
            for i in range(100):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1000000,
                    data=b"x" * 100,
                    publish_time=i * 1000000,
                )

            writer.finish()

        # Verify chunks were created
        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert summary.statistics
            stats = summary.statistics
            assert stats.message_count == 100
            assert stats.schema_count == 1
            assert stats.channel_count == 1
            assert stats.attachment_count == 0
            assert stats.metadata_count == 0
            assert stats.chunk_count > 0
            assert stats.message_start_time == 0
            assert stats.message_end_time == 99 * 1000000
            assert stats.channel_message_counts == {1: 100}

    def test_unchunked_mode(self, temp_mcap_file: Path):
        """Test writing in unchunked mode."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, chunk_size=0)  # Disable chunking
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            writer.add_message(
                channel_id=1,
                log_time=1000,
                data=b"test",
                publish_time=1000,
            )

            writer.finish()

        # Verify file was created successfully
        # Note: small-mcap always creates at least one chunk
        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert summary.statistics
            stats = summary.statistics
            assert stats.message_count == 1
            assert stats.schema_count == 1
            assert stats.channel_count == 1
            assert stats.attachment_count == 0
            assert stats.metadata_count == 0
            assert stats.chunk_count >= 0  # Unchunked mode may have 0 or 1 chunk
            assert stats.message_start_time == 1000
            assert stats.message_end_time == 1000
            assert stats.channel_message_counts == {1: 1}

    def test_unchunked_mode_statistics(self, temp_mcap_file: Path):
        """Test that statistics are correct in true unchunked mode (use_chunking=False)."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, use_chunking=False)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )
            writer.add_channel(
                channel_id=2,
                topic="/test2",
                message_encoding="json",
                schema_id=1,
            )

            writer.add_message(channel_id=1, log_time=1000, data=b"a", publish_time=1000)
            writer.add_message(channel_id=1, log_time=3000, data=b"b", publish_time=3000)
            writer.add_message(channel_id=2, log_time=2000, data=b"c", publish_time=2000)

            writer.finish()

        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert summary.statistics
            stats = summary.statistics
            assert stats.message_count == 3
            assert stats.schema_count == 1
            assert stats.channel_count == 2
            assert stats.chunk_count == 0
            assert stats.message_start_time == 1000
            assert stats.message_end_time == 3000
            assert stats.channel_message_counts == {1: 2, 2: 1}


class TestMcapWriterCompression:
    """Test compression functionality."""

    def test_no_compression(self, temp_mcap_file: Path):
        """Test writing without compression."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, compression=CompressionType.NONE, chunk_size=1024)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            writer.add_message(
                channel_id=1,
                log_time=1000,
                data=b"x" * 1000,
                publish_time=1000,
            )

            writer.finish()

    def test_lz4_compression(self, temp_mcap_file: Path):
        """Test writing with LZ4 compression."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, compression=CompressionType.LZ4, chunk_size=1024)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            # Write compressible data
            for i in range(10):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1000,
                    data=b"x" * 1000,
                    publish_time=i * 1000,
                )

            writer.finish()

    def test_zstd_compression(self, temp_mcap_file: Path):
        """Test writing with ZSTD compression."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, compression=CompressionType.ZSTD, chunk_size=1024)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            # Write compressible data
            for i in range(10):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1000,
                    data=b"x" * 1000,
                    publish_time=i * 1000,
                )

            writer.finish()


class TestMcapWriterIndexing:
    """Test index generation."""

    def test_with_all_indexes(self, temp_mcap_file: Path):
        """Test writing with all indexes enabled."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, index_types=IndexType.ALL, chunk_size=1024)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            for i in range(10):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1000,
                    data=b"test",
                    publish_time=i * 1000,
                )

            writer.finish()

        # Verify indexes were created
        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert len(summary.chunk_indexes) > 0

    def test_without_indexes(self, temp_mcap_file: Path):
        """Test writing without indexes."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f, index_types=IndexType.NONE, chunk_size=1024)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            writer.add_message(
                channel_id=1,
                log_time=1000,
                data=b"test",
                publish_time=1000,
            )

            writer.finish()


class TestMcapWriterAttachmentsMetadata:
    """Test attachment and metadata functionality."""

    def test_add_attachment(self, temp_mcap_file: Path):
        """Test adding attachments."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_attachment(
                log_time=1000000000,
                create_time=1000000000,
                name="config.yaml",
                media_type="application/yaml",
                data=b"key: value\n",
            )

            writer.finish()

        # Verify attachment was added
        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert summary.statistics
            stats = summary.statistics
            assert stats.message_count == 0
            assert stats.schema_count == 0
            assert stats.channel_count == 0
            assert stats.attachment_count == 1
            assert stats.metadata_count == 0
            assert stats.chunk_count >= 0
            assert stats.message_start_time == 0  # No messages
            assert stats.message_end_time == 0  # No messages
            assert stats.channel_message_counts == {}

    def test_add_metadata(self, temp_mcap_file: Path):
        """Test adding metadata."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_metadata(
                name="config",
                metadata={"version": "1.0", "author": "test"},
            )

            writer.finish()

        # Verify metadata was added
        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert summary.statistics
            stats = summary.statistics
            assert stats.message_count == 0
            assert stats.schema_count == 0
            assert stats.channel_count == 0
            assert stats.attachment_count == 0
            assert stats.metadata_count == 1
            assert stats.chunk_count >= 0
            assert stats.message_start_time == 0  # No messages
            assert stats.message_end_time == 0  # No messages
            assert stats.channel_message_counts == {}


class TestMcapWriterStatistics:
    """Test statistics tracking."""

    def test_statistics_message_count(self, temp_mcap_file: Path):
        """Test that statistics correctly track message count."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            num_messages = 42
            for i in range(num_messages):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1000,
                    data=b"test",
                    publish_time=i * 1000,
                )

            writer.finish()

        # Verify statistics
        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert summary.statistics
            stats = summary.statistics
            assert stats.message_count == num_messages
            assert stats.schema_count == 1
            assert stats.channel_count == 1
            assert stats.attachment_count == 0
            assert stats.metadata_count == 0
            assert stats.chunk_count >= 0
            assert stats.message_start_time == 0
            assert stats.message_end_time == (num_messages - 1) * 1000
            assert stats.channel_message_counts == {1: num_messages}

    def test_statistics_time_range(self, temp_mcap_file: Path):
        """Test that statistics correctly track time range."""
        with open(temp_mcap_file, "wb") as f:
            writer = McapWriter(f)
            writer.start()

            writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
            writer.add_channel(
                channel_id=1,
                topic="/test",
                message_encoding="json",
                schema_id=1,
            )

            start_time = 1000000000
            end_time = 2000000000

            writer.add_message(
                channel_id=1,
                log_time=start_time,
                data=b"first",
                publish_time=start_time,
            )
            writer.add_message(
                channel_id=1,
                log_time=end_time,
                data=b"last",
                publish_time=end_time,
            )

            writer.finish()

        # Verify time range
        with open(temp_mcap_file, "rb") as f:
            summary = get_summary(f)
            assert summary.statistics
            stats = summary.statistics
            assert stats.message_count == 2
            assert stats.schema_count == 1
            assert stats.channel_count == 1
            assert stats.attachment_count == 0
            assert stats.metadata_count == 0
            assert stats.chunk_count >= 0
            assert stats.message_start_time == start_time
            assert stats.message_end_time == end_time
            assert stats.channel_message_counts == {1: 2}


def _write_test_mcap(path, num_workers: int, compression=CompressionType.ZSTD, chunk_size=512):
    """Helper to write a test MCAP with given num_workers."""
    with open(path, "wb") as f:
        writer = McapWriter(
            f,
            compression=compression,
            chunk_size=chunk_size,
            num_workers=num_workers,
        )
        writer.start()
        writer.add_schema(schema_id=1, name="Test", encoding="json", data=b"{}")
        writer.add_channel(channel_id=1, topic="/a", message_encoding="json", schema_id=1)
        writer.add_channel(channel_id=2, topic="/b", message_encoding="json", schema_id=1)
        for i in range(200):
            ch = 1 if i % 2 == 0 else 2
            writer.add_message(
                channel_id=ch,
                log_time=i * 1_000_000,
                data=f"msg-{i}".encode() + b"\x00" * 50,
                publish_time=i * 1_000_000,
            )
        writer.finish()


class TestMcapWriterParallel:
    """Test parallel chunk compression."""

    def test_parallel_matches_sequential(self, tmp_path):
        """Parallel and sequential writes produce identical messages on read-back."""
        seq_path = tmp_path / "seq.mcap"
        par_path = tmp_path / "par.mcap"

        _write_test_mcap(seq_path, num_workers=0)
        _write_test_mcap(par_path, num_workers=2)

        with open(seq_path, "rb") as f:
            seq_msgs = list(read_message(f))
        with open(par_path, "rb") as f:
            par_msgs = list(read_message(f))

        assert len(seq_msgs) == len(par_msgs) == 200
        for (_, s_ch, s_msg), (_, p_ch, p_msg) in zip(seq_msgs, par_msgs, strict=True):
            assert s_ch.id == p_ch.id
            assert s_msg.log_time == p_msg.log_time
            assert s_msg.data == p_msg.data

    def test_statistics_match(self, tmp_path):
        """Statistics are identical between parallel and sequential modes."""
        seq_path = tmp_path / "seq.mcap"
        par_path = tmp_path / "par.mcap"

        _write_test_mcap(seq_path, num_workers=0)
        _write_test_mcap(par_path, num_workers=2)

        with open(seq_path, "rb") as f:
            seq_stats = get_summary(f).statistics
        with open(par_path, "rb") as f:
            par_stats = get_summary(f).statistics

        assert seq_stats.message_count == par_stats.message_count
        assert seq_stats.chunk_count == par_stats.chunk_count
        assert seq_stats.channel_message_counts == par_stats.channel_message_counts
        assert seq_stats.message_start_time == par_stats.message_start_time
        assert seq_stats.message_end_time == par_stats.message_end_time

    @pytest.mark.parametrize(
        "compression",
        [CompressionType.ZSTD, CompressionType.LZ4, CompressionType.NONE],
    )
    def test_compression_types(self, tmp_path, compression):
        """Parallel mode works with all compression types."""
        path = tmp_path / f"par_{compression.value or 'none'}.mcap"
        _write_test_mcap(path, num_workers=2, compression=compression)

        with open(path, "rb") as f:
            msgs = list(read_message(f))
        assert len(msgs) == 200

        with open(path, "rb") as f:
            stats = get_summary(f).statistics
        assert stats.message_count == 200
        assert stats.chunk_count > 1

    def test_interleaved_attachment_metadata(self, tmp_path):
        """Attachments and metadata interleaved with messages drain pending chunks."""
        path = tmp_path / "interleaved.mcap"
        with open(path, "wb") as f:
            writer = McapWriter(f, chunk_size=256, num_workers=2)
            writer.start()
            writer.add_schema(schema_id=1, name="T", encoding="json", data=b"{}")
            writer.add_channel(channel_id=1, topic="/t", message_encoding="json", schema_id=1)

            for i in range(50):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1_000_000,
                    data=b"x" * 100,
                    publish_time=i * 1_000_000,
                )

            writer.add_attachment(
                log_time=0,
                create_time=0,
                name="f.bin",
                media_type="application/octet-stream",
                data=b"att",
            )

            for i in range(50, 100):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1_000_000,
                    data=b"x" * 100,
                    publish_time=i * 1_000_000,
                )

            writer.add_metadata(name="info", metadata={"k": "v"})

            for i in range(100, 150):
                writer.add_message(
                    channel_id=1,
                    log_time=i * 1_000_000,
                    data=b"x" * 100,
                    publish_time=i * 1_000_000,
                )

            writer.finish()

        with open(path, "rb") as f:
            stats = get_summary(f).statistics
        assert stats.message_count == 150
        assert stats.attachment_count == 1
        assert stats.metadata_count == 1

        with open(path, "rb") as f:
            msgs = list(read_message(f))
        assert len(msgs) == 150

    def test_use_chunking_false_ignores_num_workers(self, tmp_path):
        """When use_chunking=False, num_workers is ignored."""
        path = tmp_path / "unchunked.mcap"
        with open(path, "wb") as f:
            writer = McapWriter(f, use_chunking=False, num_workers=2)
            writer.start()
            writer.add_schema(schema_id=1, name="T", encoding="json", data=b"{}")
            writer.add_channel(channel_id=1, topic="/t", message_encoding="json", schema_id=1)
            writer.add_message(channel_id=1, log_time=1000, data=b"hi", publish_time=1000)
            writer.finish()

        assert writer._pool is None

        with open(path, "rb") as f:
            stats = get_summary(f).statistics
        assert stats.message_count == 1
        assert stats.chunk_count == 0
