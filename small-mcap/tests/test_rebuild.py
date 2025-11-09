"""Tests for rebuild.py - rebuilding MCAP summary sections from data."""

from pathlib import Path

import pytest
from small_mcap import get_summary, rebuild_summary

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
