"""Tests for the Remapper class."""

import pytest
from small_mcap import Channel, McapError, Message, Schema
from small_mcap.remapper import Remapper


def test_schema_preserves_original_id():
    """Schema ID should be preserved when no conflict."""
    remapper = Remapper()
    schema = Schema(id=5, name="test", encoding="proto", data=b"data")

    mapped = remapper.remap_schema(0, schema)

    assert mapped.id == 5
    assert not remapper.was_schema_remapped(0, 5)


def test_schema_remaps_on_conflict():
    """Schema ID should be remapped when there's a conflict."""
    remapper = Remapper()
    schema1 = Schema(id=1, name="first", encoding="proto", data=b"data1")
    schema2 = Schema(id=1, name="second", encoding="proto", data=b"data2")

    mapped1 = remapper.remap_schema(0, schema1)
    mapped2 = remapper.remap_schema(1, schema2)

    assert mapped1.id == 1
    assert mapped2.id == 2  # Remapped due to conflict
    assert remapper.was_schema_remapped(1, 1)


def test_schema_deduplication():
    """Identical schemas should be deduplicated."""
    remapper = Remapper()
    schema1 = Schema(id=1, name="test", encoding="proto", data=b"data")
    schema2 = Schema(id=2, name="test", encoding="proto", data=b"data")

    mapped1 = remapper.remap_schema(0, schema1)
    mapped2 = remapper.remap_schema(1, schema2)

    assert mapped1.id == mapped2.id  # Same content = same ID
    assert remapper.was_schema_remapped(1, 2)  # ID changed from 2 to 1


def test_schema_none_returns_none():
    """remap_schema(None) should return None."""
    remapper = Remapper()
    assert remapper.remap_schema(0, None) is None


def test_channel_preserves_original_id():
    """Channel ID should be preserved when no conflict."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)

    channel = Channel(id=5, schema_id=1, topic="/test", message_encoding="proto", metadata={})
    mapped = remapper.remap_channel(0, channel)

    assert mapped.id == 5
    assert not remapper.was_channel_remapped(0, 5)


def test_channel_remaps_on_conflict():
    """Channel ID should be remapped when there's a conflict."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)
    remapper.remap_schema(1, schema)

    channel1 = Channel(id=1, schema_id=1, topic="/topic1", message_encoding="proto", metadata={})
    channel2 = Channel(id=1, schema_id=1, topic="/topic2", message_encoding="proto", metadata={})

    mapped1 = remapper.remap_channel(0, channel1)
    mapped2 = remapper.remap_channel(1, channel2)

    assert mapped1.id == 1
    assert mapped2.id == 2  # Remapped due to conflict
    assert remapper.was_channel_remapped(1, 1)


def test_channel_schema_id_is_remapped():
    """Channel's schema_id should be remapped to match the remapped schema ID.

    This test catches the bug where schema_id was set to 0 instead of the
    correct remapped value.
    """
    remapper = Remapper()

    # First file: schema with id=1
    schema1 = Schema(id=1, name="first", encoding="proto", data=b"data1")
    # Second file: different schema also with id=1 (conflict)
    schema2 = Schema(id=1, name="second", encoding="proto", data=b"data2")

    remapper.remap_schema(0, schema1)
    mapped_schema2 = remapper.remap_schema(1, schema2)

    # schema2 gets remapped to id=2
    assert mapped_schema2.id == 2

    # Channel from second file references schema_id=1 (original)
    channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="proto", metadata={})
    mapped_channel = remapper.remap_channel(1, channel)

    # The channel's schema_id should be remapped to 2, not 0 or 1
    assert mapped_channel.schema_id == 2


def test_channel_schemaless_preserved():
    """Schemaless channels (schema_id=0) should remain schemaless."""
    remapper = Remapper()

    channel = Channel(id=1, schema_id=0, topic="/test", message_encoding="json", metadata={})
    mapped = remapper.remap_channel(0, channel)

    assert mapped.schema_id == 0


def test_channel_with_unseen_schema_raises_error():
    """Remapping a channel that references an unseen schema should raise an error."""
    remapper = Remapper()

    # Channel references schema_id=5 but we never remapped that schema
    channel = Channel(id=1, schema_id=5, topic="/test", message_encoding="proto", metadata={})

    with pytest.raises(McapError, match=r"schema_id=5.*not been seen"):
        remapper.remap_channel(0, channel)


def test_has_channel():
    """has_channel should return True only for seen channels."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)

    channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="proto", metadata={})

    assert not remapper.has_channel(0, 1)
    remapper.remap_channel(0, channel)
    assert remapper.has_channel(0, 1)


def test_schema_fast_path_same_stream():
    """Calling remap_schema twice with same schema returns cached result."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")

    mapped1 = remapper.remap_schema(0, schema)
    mapped2 = remapper.remap_schema(0, schema)

    assert mapped1 is mapped2  # Same object from cache


def test_channel_fast_path_same_stream():
    """Calling remap_channel twice with same channel returns cached result."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)

    channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="proto", metadata={})

    mapped1 = remapper.remap_channel(0, channel)
    mapped2 = remapper.remap_channel(0, channel)

    assert mapped1 is mapped2  # Same object from cache


def test_channel_deduplication():
    """Identical channels (same topic/encoding) should be deduplicated."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)
    remapper.remap_schema(1, schema)

    channel1 = Channel(id=1, schema_id=1, topic="/test", message_encoding="proto", metadata={})
    channel2 = Channel(id=2, schema_id=1, topic="/test", message_encoding="proto", metadata={})

    mapped1 = remapper.remap_channel(0, channel1)
    mapped2 = remapper.remap_channel(1, channel2)

    assert mapped1.id == mapped2.id  # Same content = same ID
    assert remapper.was_channel_remapped(1, 2)  # ID changed from 2 to 1


def test_get_remapped_channel():
    """get_remapped_channel should return Channel or None."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)

    channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="proto", metadata={})

    # Not yet mapped
    assert remapper.get_remapped_channel(0, 1) is None

    remapper.remap_channel(0, channel)

    # Now mapped
    result = remapper.get_remapped_channel(0, 1)
    assert result is not None
    assert result.id == 1
    assert result.topic == "/test"


def test_message_returns_same_object_when_no_remap():
    """Message should be returned unchanged when channel ID wasn't remapped."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)

    channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="proto", metadata={})
    remapper.remap_channel(0, channel)

    message = Message(channel_id=1, sequence=42, log_time=1000, publish_time=2000, data=b"test")
    mapped = remapper.remap_message(0, message)

    assert mapped is message  # Same object, no copy made


def test_message_remaps_channel_id():
    """Message channel_id should be remapped when channel ID was remapped."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)
    remapper.remap_schema(1, schema)

    channel1 = Channel(id=1, schema_id=1, topic="/topic1", message_encoding="proto", metadata={})
    channel2 = Channel(id=1, schema_id=1, topic="/topic2", message_encoding="proto", metadata={})

    remapper.remap_channel(0, channel1)
    remapper.remap_channel(1, channel2)  # Gets remapped to id=2

    message = Message(channel_id=1, sequence=42, log_time=1000, publish_time=2000, data=b"test")
    mapped = remapper.remap_message(1, message)

    assert mapped is not message  # New object created
    assert mapped.channel_id == 2  # Remapped channel ID


def test_message_preserves_other_fields():
    """All message fields except channel_id should be preserved after remapping."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)
    remapper.remap_schema(1, schema)

    channel1 = Channel(id=1, schema_id=1, topic="/topic1", message_encoding="proto", metadata={})
    channel2 = Channel(id=1, schema_id=1, topic="/topic2", message_encoding="proto", metadata={})

    remapper.remap_channel(0, channel1)
    remapper.remap_channel(1, channel2)  # Gets remapped to id=2

    original_data = b"important payload data"
    message = Message(
        channel_id=1, sequence=123, log_time=999999, publish_time=888888, data=original_data
    )
    mapped = remapper.remap_message(1, message)

    assert mapped.sequence == 123
    assert mapped.log_time == 999999
    assert mapped.publish_time == 888888
    assert mapped.data == original_data


# ============================================================================
# Micro Benchmarks
# ============================================================================


@pytest.fixture
def remapper_with_data():
    """Pre-populated remapper for benchmarking cache hits."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"schema_data")
    channel = Channel(id=1, schema_id=1, topic="/test", message_encoding="proto", metadata={})
    remapper.remap_schema(0, schema)
    remapper.remap_channel(0, channel)
    return remapper, schema, channel


def test_benchmark_remap_schema_fast_path(benchmark, remapper_with_data):
    """Benchmark remap_schema cache hit (fast path)."""
    remapper, schema, _ = remapper_with_data
    benchmark(remapper.remap_schema, 0, schema)


def test_benchmark_remap_schema_slow_path(benchmark):
    """Benchmark remap_schema with new schema (slow path)."""
    schemas = [
        Schema(id=i, name=f"schema_{i}", encoding="proto", data=f"data_{i}".encode())
        for i in range(1000)
    ]

    def run():
        remapper = Remapper()
        for i, schema in enumerate(schemas):
            remapper.remap_schema(i, schema)

    benchmark(run)


def test_benchmark_remap_channel_fast_path(benchmark, remapper_with_data):
    """Benchmark remap_channel cache hit (fast path)."""
    remapper, _, channel = remapper_with_data
    benchmark(remapper.remap_channel, 0, channel)


def test_benchmark_remap_channel_slow_path(benchmark):
    """Benchmark remap_channel with new channels (slow path)."""
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    channels = [
        Channel(id=i, schema_id=1, topic=f"/topic_{i}", message_encoding="proto", metadata={})
        for i in range(1000)
    ]

    def run():
        remapper = Remapper()
        # Register schema for stream 0 (all channels use same stream)
        remapper.remap_schema(0, schema)
        for channel in channels:
            remapper.remap_channel(0, channel)

    benchmark(run)


def test_benchmark_remap_message_no_remap(benchmark, remapper_with_data):
    """Benchmark remap_message when channel ID unchanged."""
    remapper, _, _ = remapper_with_data
    message = Message(channel_id=1, sequence=1, log_time=1000, publish_time=1000, data=b"data")
    benchmark(remapper.remap_message, 0, message)


def test_benchmark_remap_message_with_remap(benchmark):
    """Benchmark remap_message when channel ID needs remapping."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)
    remapper.remap_schema(1, schema)

    channel1 = Channel(id=1, schema_id=1, topic="/topic1", message_encoding="proto", metadata={})
    channel2 = Channel(id=1, schema_id=1, topic="/topic2", message_encoding="proto", metadata={})
    remapper.remap_channel(0, channel1)
    remapper.remap_channel(1, channel2)  # Gets id=2

    message = Message(channel_id=1, sequence=1, log_time=1000, publish_time=1000, data=b"data")
    benchmark(remapper.remap_message, 1, message)


def test_benchmark_was_channel_remapped(benchmark, remapper_with_data):
    """Benchmark was_channel_remapped check."""
    remapper, _, _ = remapper_with_data
    benchmark(remapper.was_channel_remapped, 0, 1)
