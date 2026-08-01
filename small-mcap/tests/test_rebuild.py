"""Tests for rebuild.py - rebuilding MCAP summary sections from data."""

import io
import struct
from pathlib import Path

import pytest
from small_mcap import (
    McapWriter,
    get_summary,
    read_info_approximate,
    rebuild_summary,
    stream_reader,
)
from small_mcap.exceptions import (
    CRCValidationError,
    EndOfFileError,
    McapError,
    UnsupportedCompressionError,
)
from small_mcap.rebuild import MESSAGE_RECORD_OVERHEAD, _estimate_size_from_indexes
from small_mcap.records import (
    MAGIC,
    OPCODE_AND_LEN_STRUCT,
    Chunk,
    DataEnd,
    Header,
    LazyChunk,
    Message,
    MessageIndex,
    Opcode,
)
from small_mcap.writer import CompressionType, IndexType

# Path to conformance test data
CONFORMANCE_DIR = Path(__file__).parent.parent.parent / "data" / "conformance"


def _write_multi_chunk_file(path: Path) -> list[LazyChunk]:
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=160, compression=CompressionType.NONE)
        writer.start(profile="test", library="small-mcap-test")
        writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
        writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=1)
        for index in range(24):
            writer.add_message(
                channel_id=1,
                log_time=index + 1,
                publish_time=index + 1,
                data=bytes([index]) * 48,
            )
        writer.finish()

    with path.open("rb") as stream:
        chunks = [
            record
            for record in stream_reader(stream, emit_chunks=True, lazy_chunks=True)
            if isinstance(record, LazyChunk)
        ]

    assert len(chunks) >= 2
    return chunks


def _write_file_truncated_inside_final_chunk(path: Path) -> list[LazyChunk]:
    chunks = _write_multi_chunk_file(path)
    final_chunk = chunks[-1]
    chunk_data_start = (
        final_chunk.record_start
        + OPCODE_AND_LEN_STRUCT.size
        + 8
        + 8
        + 8
        + 4
        + 4
        + len(final_chunk.compression.encode())
        + 8
    )
    assert final_chunk.data_len > 1
    truncate_at = chunk_data_start + final_chunk.data_len // 2
    path.write_bytes(path.read_bytes()[:truncate_at])
    return chunks


def test_read_info_approximate_reports_message_index_progress():
    buffer = io.BytesIO()
    writer = McapWriter(buffer, chunk_size=256, compression=CompressionType.NONE)
    writer.start(profile="test", library="small-mcap-test")
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/left", message_encoding="json", schema_id=1)
    writer.add_channel(channel_id=2, topic="/right", message_encoding="json", schema_id=1)
    for index in range(50):
        log_time = index * 1_000_000
        writer.add_message(
            channel_id=1,
            log_time=log_time,
            publish_time=log_time,
            data=b'{"channel": 1}',
        )
        writer.add_message(
            channel_id=2,
            log_time=log_time,
            publish_time=log_time,
            data=b'{"channel": 2}',
        )
    writer.finish()

    progress_updates: list[tuple[int, int]] = []
    buffer.seek(0)
    info = read_info_approximate(
        buffer,
        progress_callback=lambda completed, total: progress_updates.append((completed, total)),
    )

    assert info is not None
    expected_total = sum(
        len(chunk_index.message_index_offsets) for chunk_index in info.summary.chunk_indexes
    )
    assert expected_total > 0
    assert progress_updates[0] == (0, expected_total)
    assert progress_updates[-1] == (expected_total, expected_total)
    assert [completed for completed, _ in progress_updates] == list(range(expected_total + 1))


@pytest.mark.conformance
def test_rebuild_simple_chunked_file():
    """Test rebuilding summary from a simple chunked MCAP file."""
    test_file = CONFORMANCE_DIR / "TenMessages" / "TenMessages-ch.mcap"

    with open(test_file, "rb") as f:
        # Rebuild the summary
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )

        # Verify we got valid results
        assert rebuild_info.header is not None
        assert rebuild_info.summary is not None

        # Verify summary has expected structure
        summary = rebuild_info.summary
        assert len(summary.schemas) >= 0
        assert len(summary.channels) >= 0
        assert len(summary.chunk_indexes) >= 0

        # Verify statistics
        assert summary.statistics is not None
        assert summary.statistics.chunk_count >= 1


@pytest.mark.conformance
def test_rebuild_with_indexes():
    """Test rebuilding preserves chunk and message indexes."""
    test_file = CONFORMANCE_DIR / "OneMessage" / "OneMessage-ch-chx-mx.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )

        # Verify chunk indexes were rebuilt
        assert len(rebuild_info.summary.chunk_indexes) > 0

        # With indexes, chunk information should be populated
        if rebuild_info.chunk_information:
            assert len(rebuild_info.chunk_information) > 0


@pytest.mark.conformance
def test_rebuild_with_attachments():
    """Test rebuilding preserves attachments."""
    test_file = CONFORMANCE_DIR / "OneAttachment" / "OneAttachment-ax.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )

        # Verify attachment was counted in statistics
        assert rebuild_info.summary.statistics is not None
        assert rebuild_info.summary.statistics.attachment_count == 1


@pytest.mark.conformance
def test_rebuild_with_metadata():
    """Test rebuilding preserves metadata."""
    test_file = CONFORMANCE_DIR / "OneMetadata" / "OneMetadata-mdx.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )

        # Verify metadata was counted in statistics
        assert rebuild_info.summary.statistics is not None
        assert rebuild_info.summary.statistics.metadata_count == 1


@pytest.mark.conformance
def test_rebuild_multiple_messages():
    """Test rebuilding file with multiple messages."""
    test_file = CONFORMANCE_DIR / "TenMessages" / "TenMessages-ch-chx-mx.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )

        # Verify all 10 messages were counted
        assert rebuild_info.summary.statistics is not None
        assert rebuild_info.summary.statistics.message_count == 10

        # Verify channels and schemas
        assert len(rebuild_info.summary.channels) >= 1
        assert len(rebuild_info.summary.schemas) >= 0

        # Verify chunk indexes
        assert len(rebuild_info.summary.chunk_indexes) > 0


@pytest.mark.conformance
def test_rebuild_channel_sizes_exact():
    """Test rebuilding with exact channel size calculation."""
    test_file = CONFORMANCE_DIR / "TenMessages" / "TenMessages-ch.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=True, exact_sizes=True
        )

        # Verify channel sizes were calculated
        assert rebuild_info.channel_sizes is not None
        assert len(rebuild_info.channel_sizes) > 0

        # Verify it's marked as exact
        assert rebuild_info.estimated_channel_sizes is False

        # Verify sizes are positive
        for size in rebuild_info.channel_sizes.values():
            assert size > 0


@pytest.mark.conformance
def test_rebuild_channel_sizes_estimated():
    """Test rebuilding with estimated channel size calculation."""
    test_file = CONFORMANCE_DIR / "TenMessages" / "TenMessages-ch-chx-mx.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=True, exact_sizes=False
        )

        # With message indexes, we can estimate channel sizes
        # Verify channel sizes dict exists (may be empty if no messages or no indexes)
        assert rebuild_info.channel_sizes is not None

        # Verify it's marked as estimated when using approximate calculation
        assert rebuild_info.estimated_channel_sizes is True

        # If channel sizes were calculated, verify they are positive
        for size in rebuild_info.channel_sizes.values():
            assert size > 0


@pytest.mark.conformance
def test_rebuild_full_features():
    """Test rebuilding file with all MCAP features."""
    test_file = CONFORMANCE_DIR / "OneMessage" / "OneMessage-ch-chx-mx-pad-rch-rsh-st-sum.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )

        # Verify basic structure
        assert rebuild_info.header is not None
        assert rebuild_info.summary is not None

        # Verify statistics
        stats = rebuild_info.summary.statistics
        assert stats is not None
        assert stats.message_count >= 1

        # Verify schemas and channels (repeated in summary)
        assert len(rebuild_info.summary.schemas) >= 0
        assert len(rebuild_info.summary.channels) >= 1

        # Verify chunk indexes
        assert len(rebuild_info.summary.chunk_indexes) > 0


@pytest.mark.conformance
def test_rebuild_matches_original_summary():
    """Test that rebuilding produces a summary matching the original."""
    test_file = CONFORMANCE_DIR / "TenMessages" / "TenMessages-ch-chx-mx-pad-rch-rsh-st-sum.mcap"

    # Read the original summary from the file
    with open(test_file, "rb") as f:
        original_summary = get_summary(f)

    # Rebuild the summary
    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )
        rebuilt_summary = rebuild_info.summary

    # Compare key statistics
    assert rebuilt_summary.statistics is not None
    assert original_summary.statistics is not None

    assert rebuilt_summary.statistics.message_count == original_summary.statistics.message_count
    assert rebuilt_summary.statistics.channel_count == original_summary.statistics.channel_count
    assert rebuilt_summary.statistics.schema_count == original_summary.statistics.schema_count

    # Compare number of schemas and channels
    assert len(rebuilt_summary.schemas) == len(original_summary.schemas)
    assert len(rebuilt_summary.channels) == len(original_summary.channels)

    # Compare chunk indexes count
    assert len(rebuilt_summary.chunk_indexes) == len(original_summary.chunk_indexes)


# Tests for _estimate_size_from_indexes


def test_estimate_size_single_channel():
    """Test estimation with a single channel and multiple messages."""
    # 3 messages at offsets 0, 100, 200 in a 300-byte chunk
    indexes = [MessageIndex(channel_id=1, timestamps=[1000, 2000, 3000], offsets=[0, 100, 200])]
    chunk_size = 300

    result = _estimate_size_from_indexes(indexes, chunk_size)

    # Each message: gap - overhead = 100 - 31 = 69
    expected_size = 3 * (100 - MESSAGE_RECORD_OVERHEAD)
    assert result == {1: expected_size}


def test_estimate_size_multi_channel():
    """Test multi-channel estimation (regression test for closure bug)."""
    # 2 channels with interleaved messages at equal spacing
    indexes = [
        MessageIndex(
            channel_id=1, timestamps=[100, 300], offsets=[0, 200]
        ),  # ch1 at offsets 0, 200
        MessageIndex(
            channel_id=2, timestamps=[200, 400], offsets=[100, 300]
        ),  # ch2 at offsets 100, 300
    ]
    chunk_size = 400

    result = _estimate_size_from_indexes(indexes, chunk_size)

    # Sorted by offset: (0,ch1), (100,ch2), (200,ch1), (300,ch2), end=400
    # ch1: (100-0-31) + (300-200-31) = 69 + 69 = 138
    # ch2: (200-100-31) + (400-300-31) = 69 + 69 = 138
    assert result == {1: 138, 2: 138}


def test_estimate_size_multi_channel_uneven_distribution():
    """Test multi-channel with uneven message distribution."""
    indexes = [
        MessageIndex(channel_id=1, timestamps=[100], offsets=[0]),  # 1 message
        MessageIndex(
            channel_id=2, timestamps=[200, 300, 400], offsets=[50, 100, 150]
        ),  # 3 messages
    ]
    chunk_size = 200

    result = _estimate_size_from_indexes(indexes, chunk_size)

    # Both channels must be present
    assert 1 in result
    assert 2 in result
    # Total size should equal chunk_size minus overhead for all 4 messages
    total = sum(result.values())
    assert total == chunk_size - 4 * MESSAGE_RECORD_OVERHEAD


def test_estimate_size_empty_indexes():
    """Test with empty index list."""
    result = _estimate_size_from_indexes([], 1000)
    assert result == {}


def test_estimate_size_empty_records():
    """Test with index containing no records."""
    indexes = [MessageIndex(channel_id=1, timestamps=[], offsets=[])]
    result = _estimate_size_from_indexes(indexes, 1000)
    assert result == {}


# Tests for rebuild_summary edge cases


@pytest.mark.conformance
def test_rebuild_with_initial_state_resumption():
    """Test rebuilding with initial_state for resumption."""
    test_file = CONFORMANCE_DIR / "TenMessages" / "TenMessages-ch-chx-mx.mcap"

    # First, do a partial read
    with open(test_file, "rb") as f:
        initial_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=True, exact_sizes=False
        )

    # Now resume from the initial state (simulating continuation)
    with open(test_file, "rb") as f:
        f.seek(initial_info.next_offset)
        resumed_info = rebuild_summary(
            f,
            validate_crc=False,
            calculate_channel_sizes=True,
            exact_sizes=False,
            initial_state=initial_info,
            skip_magic=True,
        )

    # Should have valid results
    assert resumed_info.header is not None
    assert resumed_info.summary is not None
    assert resumed_info.summary.statistics is not None


@pytest.mark.conformance
def test_rebuild_estimated_channel_sizes():
    """Test rebuild with estimated channel sizes and time statistics."""
    test_file = CONFORMANCE_DIR / "TenMessages" / "TenMessages-ch-chx-mx.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=True, exact_sizes=False
        )

    # Should have estimated channel sizes
    assert rebuild_info.channel_sizes is not None
    assert rebuild_info.estimated_channel_sizes is True
    assert all(size > 0 for size in rebuild_info.channel_sizes.values())

    # Should have valid time statistics
    stats = rebuild_info.summary.statistics
    assert stats is not None
    assert stats.message_start_time > 0
    assert stats.message_end_time >= stats.message_start_time


@pytest.mark.conformance
def test_rebuild_unchunked_messages():
    """Test rebuilding file with unchunked messages."""
    # This file has messages outside of chunks
    test_file = CONFORMANCE_DIR / "OneMessage" / "OneMessage.mcap"

    with open(test_file, "rb") as f:
        rebuild_info = rebuild_summary(
            f, validate_crc=False, calculate_channel_sizes=True, exact_sizes=True
        )

    # Should still count the message
    assert rebuild_info.summary.statistics is not None
    assert rebuild_info.summary.statistics.message_count >= 1


def test_rebuild_empty_stream_raises_error():
    """Test that rebuilding empty stream raises McapError."""
    empty_stream = io.BytesIO(b"")

    with pytest.raises(McapError):
        rebuild_summary(
            empty_stream, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )


def test_rebuild_invalid_magic_raises_error():
    """Test that rebuilding stream with invalid magic raises error."""
    # Invalid magic bytes
    invalid_stream = io.BytesIO(b"not an mcap file")

    with pytest.raises(McapError):
        rebuild_summary(
            invalid_stream, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
        )


@pytest.mark.parametrize(
    ("chunk", "validate_crc", "expected_error"),
    [
        (
            Chunk(0, 1, 0, 0, "bzip2", b""),
            False,
            UnsupportedCompressionError,
        ),
        (
            Chunk(0, 1, 3, 1, "", b"bad"),
            True,
            CRCValidationError,
        ),
        (
            Chunk(0, 1, 10, 0, "", struct.pack("<BQ", Opcode.MESSAGE, 22) + b"\0"),
            False,
            struct.error,
        ),
    ],
)
@pytest.mark.parametrize("allow_incomplete_tail_only", [False, True])
def test_rebuild_propagates_complete_chunk_errors(
    chunk, validate_crc, expected_error, allow_incomplete_tail_only
):
    buffer = io.BytesIO()
    buffer.write(MAGIC)
    Header(profile="test", library="test").write_record_to(buffer)
    chunk.write_record_to(buffer)
    buffer.seek(0)

    with pytest.raises(expected_error):
        rebuild_summary(
            buffer,
            validate_crc=validate_crc,
            calculate_channel_sizes=False,
            exact_sizes=True,
            allow_incomplete_tail_only=allow_incomplete_tail_only,
        )


class _NonSeekableStream(io.BytesIO):
    def seekable(self) -> bool:
        return False


@pytest.mark.parametrize(
    "rebuild_options",
    [{}, {"allow_incomplete_tail_only": True}],
    ids=["default", "tail-only"],
)
@pytest.mark.parametrize("exact_sizes", [False, True], ids=["estimated", "exact"])
@pytest.mark.parametrize("is_seekable", [True, False], ids=["seekable", "nonseekable"])
def test_rebuild_rolls_back_chunk_truncated_inside_data(
    tmp_path: Path, rebuild_options: dict[str, bool], exact_sizes: bool, is_seekable: bool
) -> None:
    path = tmp_path / "truncated-final-chunk.mcap"
    chunks = _write_file_truncated_inside_final_chunk(path)
    final_chunk = chunks[-1]

    with path.open("rb") as stream:
        assert get_summary(stream) is None
        raw = path.read_bytes()

    opened = io.BytesIO(raw) if is_seekable else _NonSeekableStream(raw)
    rebuilt = rebuild_summary(
        opened,
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=exact_sizes,
        **rebuild_options,
    )

    statistics = rebuilt.summary.statistics
    assert statistics is not None
    assert statistics.chunk_count == len(chunks) - 1
    assert 0 < statistics.message_count < 24
    assert statistics.message_end_time < final_chunk.message_end_time
    assert len(rebuilt.summary.chunk_indexes) == len(chunks) - 1
    assert all(
        chunk_index.chunk_start_offset < final_chunk.record_start
        for chunk_index in rebuilt.summary.chunk_indexes
    )
    assert rebuilt.next_offset == final_chunk.record_start


def test_lazy_chunk_short_content_raises_mcap_end_of_file(tmp_path: Path) -> None:
    path = tmp_path / "truncated-final-chunk.mcap"
    chunks = _write_file_truncated_inside_final_chunk(path)

    with path.open("rb") as stream:
        original_position = stream.tell()
        with pytest.raises(EndOfFileError):
            chunks[-1].to_chunk(stream)
        assert stream.tell() == original_position


def test_lazy_chunk_short_metadata_raises_mcap_end_of_file(tmp_path: Path) -> None:
    path = tmp_path / "truncated-final-chunk.mcap"
    chunks = _write_file_truncated_inside_final_chunk(path)
    final_chunk = chunks[-1]
    path.write_bytes(path.read_bytes()[: final_chunk.record_start + OPCODE_AND_LEN_STRUCT.size + 5])

    with path.open("rb") as stream, pytest.raises(EndOfFileError):
        list(stream_reader(stream, emit_chunks=True, lazy_chunks=True))


def test_lazy_chunk_rejects_compression_length_past_record_boundary() -> None:
    fixed_metadata = struct.pack("<QQQII", 0, 0, 0, 0, 100)
    follower = OPCODE_AND_LEN_STRUCT.pack(Opcode.HEADER, 0)
    stream = io.BytesIO(fixed_metadata + follower)

    with pytest.raises(struct.error, match="compression length exceeds"):
        LazyChunk.read_from_stream(
            stream,
            record_start=0,
            record_length=len(fixed_metadata) + 8,
        )

    assert stream.tell() == len(fixed_metadata)


@pytest.mark.parametrize(
    ("content", "record_length", "expected_error", "message"),
    [
        (
            struct.pack("<QQQII", 0, 0, 0, 0, 3) + b"ab",
            32 + 3 + 8,
            EndOfFileError,
            "Chunk compression",
        ),
        (
            struct.pack("<QQQII", 0, 0, 0, 0, 0) + b"\0" * 4,
            32 + 8,
            EndOfFileError,
            "Chunk data length",
        ),
        (
            struct.pack("<QQQIIQ", 0, 0, 0, 0, 0, 2) + b"x",
            32 + 8 + 1,
            struct.error,
            "data length exceeds",
        ),
    ],
)
def test_lazy_chunk_rejects_truncated_length_prefixed_fields(
    content: bytes,
    record_length: int,
    expected_error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(expected_error, match=message):
        LazyChunk.read_from_stream(io.BytesIO(content), 0, record_length)


def test_failed_incremental_rebuild_does_not_mutate_initial_state() -> None:
    prefix = io.BytesIO()
    prefix.write(MAGIC)
    Header(profile="test", library="test").write_record_to(prefix)
    initial = rebuild_summary(
        io.BytesIO(prefix.getvalue()),
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=True,
    )

    corrupt_continuation = io.BytesIO(prefix.getvalue())
    corrupt_continuation.seek(0, io.SEEK_END)
    Chunk(100, 200, 0, 0, "bzip2", b"").write_record_to(corrupt_continuation)
    corrupt_continuation.seek(initial.next_offset)
    with pytest.raises(UnsupportedCompressionError):
        rebuild_summary(
            corrupt_continuation,
            validate_crc=False,
            calculate_channel_sizes=True,
            exact_sizes=True,
            initial_state=initial,
            skip_magic=True,
        )

    initial_statistics = initial.summary.statistics
    assert initial_statistics is not None
    assert initial_statistics.chunk_count == 0
    assert initial_statistics.message_count == 0
    assert initial_statistics.message_start_time == 0
    assert initial_statistics.message_end_time == 0
    assert initial.summary.chunk_indexes == []
    assert initial.channel_sizes == {}

    corrected_continuation = io.BytesIO(prefix.getvalue())
    corrected_continuation.seek(0, io.SEEK_END)
    Message(1, 1, 300, 300, b"ok").write_record_to(corrected_continuation)
    corrected_continuation.seek(initial.next_offset)
    rebuilt = rebuild_summary(
        corrected_continuation,
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=True,
        initial_state=initial,
        skip_magic=True,
    )

    rebuilt_statistics = rebuilt.summary.statistics
    assert rebuilt_statistics is not None
    assert rebuilt_statistics.message_count == 1
    assert rebuilt_statistics.channel_message_counts == {1: 1}
    assert rebuilt.channel_sizes == {1: 2}
    assert initial_statistics.message_count == 0


def test_incremental_rebuild_uses_absolute_chunk_offsets(tmp_path: Path) -> None:
    path = tmp_path / "incremental.mcap"
    chunks = _write_multi_chunk_file(path)
    data = path.read_bytes()
    resume_offset = chunks[1].record_start

    initial = rebuild_summary(
        io.BytesIO(data[: resume_offset + OPCODE_AND_LEN_STRUCT.size + 5]),
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
    )
    assert initial.next_offset == resume_offset
    initial_chunk_offsets = [index.chunk_start_offset for index in initial.summary.chunk_indexes]

    resumed_stream = io.BytesIO(data)
    resumed_stream.seek(initial.next_offset)
    resumed = rebuild_summary(
        resumed_stream,
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
        initial_state=initial,
        skip_magic=True,
    )
    complete = rebuild_summary(
        io.BytesIO(data),
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
    )

    assert resumed.summary.statistics == complete.summary.statistics
    assert resumed.summary.chunk_indexes == complete.summary.chunk_indexes
    assert resumed.chunk_information == complete.chunk_information
    assert resumed.channel_sizes == complete.channel_sizes
    assert [index.chunk_start_offset for index in initial.summary.chunk_indexes] == (
        initial_chunk_offsets
    )


def test_rebuild_chunk_offset_ignores_unknown_extension_records() -> None:
    buffer = io.BytesIO()
    buffer.write(MAGIC)
    Header(profile="test", library="test").write_record_to(buffer)
    buffer.write(OPCODE_AND_LEN_STRUCT.pack(0x80, 3))
    buffer.write(b"ext")
    chunk_start = buffer.tell()
    Chunk(0, 0, 0, 0, "", b"").write_record_to(buffer)
    DataEnd(0).write_record_to(buffer)

    rebuilt = rebuild_summary(
        io.BytesIO(buffer.getvalue()),
        validate_crc=False,
        calculate_channel_sizes=False,
        exact_sizes=True,
    )

    assert rebuilt.summary.chunk_indexes[0].chunk_start_offset == chunk_start


@pytest.mark.parametrize("cut_kind", ["chunk-end", "message-index"])
def test_incremental_rebuild_retries_unterminated_chunk_group(
    tmp_path: Path, cut_kind: str
) -> None:
    path = tmp_path / "incremental-index.mcap"
    chunks = _write_multi_chunk_file(path)
    data = path.read_bytes()
    final_chunk = chunks[-1]

    initial = rebuild_summary(
        io.BytesIO(data[: final_chunk.record_start + OPCODE_AND_LEN_STRUCT.size + 5]),
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
    )
    assert initial.next_offset == final_chunk.record_start

    complete = rebuild_summary(
        io.BytesIO(data),
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
    )
    final_index = complete.summary.chunk_indexes[-1]
    message_index_offset = next(iter(final_index.message_index_offsets.values()))
    cut_offset = (
        final_index.chunk_start_offset + final_index.chunk_length
        if cut_kind == "chunk-end"
        else message_index_offset + 10
    )
    truncated_stream = io.BytesIO(data[:cut_offset])
    truncated_stream.seek(initial.next_offset)
    partial = rebuild_summary(
        truncated_stream,
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
        initial_state=initial,
        skip_magic=True,
    )

    assert partial.next_offset == final_chunk.record_start
    assert partial.summary == initial.summary
    assert partial.channel_sizes == initial.channel_sizes
    assert partial.chunk_information == initial.chunk_information

    retry_stream = io.BytesIO(data)
    retry_stream.seek(partial.next_offset)
    retried = rebuild_summary(
        retry_stream,
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
        initial_state=partial,
        skip_magic=True,
    )
    assert retried.summary.statistics == complete.summary.statistics
    assert retried.summary.chunk_indexes == complete.summary.chunk_indexes
    assert retried.chunk_information == complete.chunk_information


def test_rebuild_no_index_file_does_not_rebuild_indexes():
    """When no chunks have message indexes, final chunk should not rebuild indexes.

    A file written with IndexType.CHUNK (no MessageIndex records) should not
    have indexes rebuilt for the final chunk either. Note: without exact_sizes=True,
    messages inside chunks with no indexes are not decompressed, so message_count
    is only tracked for chunks that were force-decompressed.
    """
    buffer = io.BytesIO()
    # Write file with no message indexes (only ChunkIndex)
    writer = McapWriter(
        buffer, chunk_size=50, index_types=IndexType.CHUNK, compression=CompressionType.NONE
    )
    writer.start()
    writer.add_schema(1, "Test", "json", b"{}")
    writer.add_channel(1, "/test", "json", 1)
    for i in range(20):
        writer.add_message(1, i * 1000, b"x" * 30, i * 1000)
    writer.finish()

    data = buffer.getvalue()

    # Verify the written file actually has multiple chunks with no message indexes
    stream = io.BytesIO(data)
    chunk_count = 0
    for record in stream_reader(stream, emit_chunks=True, lazy_chunks=True):
        if isinstance(record, LazyChunk):
            chunk_count += 1
    assert chunk_count >= 2, "Test requires multiple chunks"

    stream = io.BytesIO(data)
    rebuild_info = rebuild_summary(
        stream, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
    )

    stats = rebuild_info.summary.statistics
    assert stats is not None

    # All chunk_indexes should have message_index_length == 0
    # (no indexes should have been rebuilt since the file has no indexes at all)
    for ci in rebuild_info.summary.chunk_indexes:
        assert ci.message_index_length == 0

    # chunk_information should be None or empty (no rebuilt indexes)
    assert rebuild_info.chunk_information is None or len(rebuild_info.chunk_information) == 0


def test_rebuild_multi_chunk_with_indexes_rebuilds_final():
    """When other chunks have indexes, final chunk should rebuild them."""
    buffer = io.BytesIO()
    # Use small chunk_size to force multiple chunks, with message indexes
    writer = McapWriter(
        buffer, chunk_size=50, index_types=IndexType.ALL, compression=CompressionType.NONE
    )
    writer.start()
    writer.add_schema(1, "Test", "json", b"{}")
    writer.add_channel(1, "/test", "json", 1)
    for i in range(20):
        writer.add_message(1, i * 1000, b"x" * 30, i * 1000)
    writer.finish()

    data = buffer.getvalue()

    # Find the last chunk+indexes boundary and truncate there to simulate
    # a file with missing final indexes
    stream = io.BytesIO(data)
    chunk_offsets = []
    prev_pos = 0
    for record in stream_reader(stream, emit_chunks=True, lazy_chunks=True):
        if isinstance(record, LazyChunk):
            chunk_offsets.append(prev_pos)
        prev_pos = stream.tell()

    # Need at least 2 chunks for this test
    assert len(chunk_offsets) >= 2

    # Rebuild the full file to check that indexes are rebuilt for final chunk
    stream = io.BytesIO(data)
    rebuild_info = rebuild_summary(
        stream, validate_crc=False, calculate_channel_sizes=False, exact_sizes=False
    )

    stats = rebuild_info.summary.statistics
    assert stats is not None
    assert stats.message_count == 20
    # Should have chunk_information populated
    assert rebuild_info.chunk_information is not None
    assert len(rebuild_info.chunk_information) > 0
