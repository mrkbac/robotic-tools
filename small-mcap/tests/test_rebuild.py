"""Tests for rebuild.py - rebuilding MCAP summary sections from data."""

import io
from pathlib import Path

import pytest
from small_mcap import get_summary, rebuild_summary
from small_mcap.exceptions import McapError
from small_mcap.rebuild import MESSAGE_RECORD_OVERHEAD, _estimate_size_from_indexes
from small_mcap.records import MessageIndex

# Path to conformance test data
CONFORMANCE_DIR = Path(__file__).parent.parent.parent / "data" / "conformance"


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
    indexes = [MessageIndex(channel_id=1, records=[(1000, 0), (2000, 100), (3000, 200)])]
    chunk_size = 300

    result = _estimate_size_from_indexes(indexes, chunk_size)

    # Each message: gap - overhead = 100 - 31 = 69
    expected_size = 3 * (100 - MESSAGE_RECORD_OVERHEAD)
    assert result == {1: expected_size}


def test_estimate_size_multi_channel():
    """Test multi-channel estimation (regression test for closure bug)."""
    # 2 channels with interleaved messages at equal spacing
    indexes = [
        MessageIndex(channel_id=1, records=[(100, 0), (300, 200)]),  # ch1 at offsets 0, 200
        MessageIndex(channel_id=2, records=[(200, 100), (400, 300)]),  # ch2 at offsets 100, 300
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
        MessageIndex(channel_id=1, records=[(100, 0)]),  # 1 message
        MessageIndex(channel_id=2, records=[(200, 50), (300, 100), (400, 150)]),  # 3 messages
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
    indexes = [MessageIndex(channel_id=1, records=[])]
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
