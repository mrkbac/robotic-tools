"""Tests for the Remapper class."""

import pytest
from small_mcap import Channel, McapError, Remapper, Schema, Summary


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


def test_get_remapped_channel_id():
    """get_remapped_channel_id should return the mapped ID."""
    remapper = Remapper()
    schema = Schema(id=1, name="test", encoding="proto", data=b"data")
    remapper.remap_schema(0, schema)
    remapper.remap_schema(1, schema)

    channel1 = Channel(id=1, schema_id=1, topic="/topic1", message_encoding="proto", metadata={})
    channel2 = Channel(id=1, schema_id=1, topic="/topic2", message_encoding="proto", metadata={})

    remapper.remap_channel(0, channel1)
    remapper.remap_channel(1, channel2)

    assert remapper.get_remapped_channel_id(0, 1) == 1
    assert remapper.get_remapped_channel_id(1, 1) == 2


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


def test_get_remapped_schema():
    """get_remapped_schema should return schema ID or 0 if not found."""
    remapper = Remapper()
    schema = Schema(id=5, name="test", encoding="proto", data=b"data")

    # Not yet mapped - returns 0
    assert remapper.get_remapped_schema(0, 5) == 0

    remapper.remap_schema(0, schema)

    # Now mapped - returns the ID
    assert remapper.get_remapped_schema(0, 5) == 5
