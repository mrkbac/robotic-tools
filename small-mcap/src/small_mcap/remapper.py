from typing import overload

from small_mcap.exceptions import ChannelNotFoundError, SchemaNotFoundError
from small_mcap.records import (
    Channel,
    Message,
    Schema,
)


def _allocate_id(used_ids: set[int], preferred_id: int) -> int:
    """Allocate an ID, preferring the given ID if available."""
    if preferred_id not in used_ids:
        used_ids.add(preferred_id)
        return preferred_id

    new_id = preferred_id
    while new_id in used_ids:
        new_id += 1
    used_ids.add(new_id)
    return new_id


class Remapper:
    """Smart ID remapper that minimizes remapping by preserving original IDs when possible.

    Tracks schema and channel IDs across multiple streams, only remapping when conflicts occur.
    Deduplicates identical schemas/channels by content to avoid duplicates.
    """

    __slots__ = (
        "_channel_lookup_fast",
        "_channel_lookup_slow",
        "_schema_lookup_fast",
        "_schema_lookup_slow",
        "_streams_with_remap",
        "_used_channel_ids",
        "_used_schema_ids",
    )

    def __init__(self) -> None:
        self._used_schema_ids: set[int] = set()
        self._used_channel_ids: set[int] = set()

        # Fast path: (stream_id, original_id) -> mapped object
        self._schema_lookup_fast: dict[tuple[int, int], Schema | None] = {}
        self._channel_lookup_fast: dict[tuple[int, int], Channel] = {}
        # Slow path: content-based deduplication
        self._schema_lookup_slow: dict[tuple[str, str, bytes], Schema] = {}
        self._channel_lookup_slow: dict[
            tuple[str, str, int, tuple[tuple[str, str], ...]], Channel
        ] = {}
        # Streams where at least one schema or channel id was changed.
        # Chunks from such streams may embed stale in-chunk Schema/Channel
        # records, so they must be decoded rather than fast-copied.
        self._streams_with_remap: set[int] = set()

    # General state
    def stream_had_remap(self, stream_id: int) -> bool:
        """Whether any schema or channel id changed during remapping for this stream.

        When true, chunks from this stream may embed stale Schema/Channel
        records referring to original ids, so they must not be fast-copied.
        """
        return stream_id in self._streams_with_remap

    def _remember_schema(self, stream_id: int, original_id: int, schema: Schema) -> Schema:
        self._schema_lookup_fast[(stream_id, original_id)] = schema
        if schema.id != original_id:
            self._streams_with_remap.add(stream_id)
        return schema

    def _remember_channel(self, stream_id: int, original_id: int, channel: Channel) -> Channel:
        self._channel_lookup_fast[(stream_id, original_id)] = channel
        if channel.id != original_id:
            self._streams_with_remap.add(stream_id)
        return channel

    # Schema IDs
    def reserve_schema_id(self, preferred_id: int) -> int:
        """Reserve a schema ID for an output-only schema."""
        return _allocate_id(self._used_schema_ids, preferred_id)

    @overload
    def remap_schema(self, stream_id: int, schema: None) -> None: ...
    @overload
    def remap_schema(self, stream_id: int, schema: Schema) -> Schema: ...
    def remap_schema(self, stream_id: int, schema: Schema | None) -> Schema | None:
        if schema is None:
            return None

        # Fast path: lookup by stream_id + schema.id
        fast_key = (stream_id, schema.id)
        mapped_schema = self._schema_lookup_fast.get(fast_key)
        if mapped_schema is not None:
            return mapped_schema

        # Slow path: lookup by schema content (deduplication)
        slow_key = (schema.name, schema.encoding, schema.data)
        mapped_schema = self._schema_lookup_slow.get(slow_key)
        if mapped_schema is not None:
            return self._remember_schema(stream_id, schema.id, mapped_schema)

        # Allocate ID, preserving original if possible
        new_id = _allocate_id(self._used_schema_ids, schema.id)
        # Direct construction is faster than dataclass.replace
        mapped_schema = Schema(
            id=new_id, name=schema.name, encoding=schema.encoding, data=schema.data
        )
        self._schema_lookup_slow[slow_key] = mapped_schema
        return self._remember_schema(stream_id, schema.id, mapped_schema)

    def was_schema_remapped(self, stream_id: int, original_id: int) -> bool:
        """Check if a schema ID was changed during remapping."""
        mapped = self._schema_lookup_fast.get((stream_id, original_id))
        return mapped is not None and mapped.id != original_id

    # Channel IDs
    def reserve_channel_id(self, preferred_id: int) -> int:
        """Reserve a channel ID for an output-only channel."""
        return _allocate_id(self._used_channel_ids, preferred_id)

    def remap_channel(self, stream_id: int, channel: Channel) -> Channel:
        # Fast path: lookup by stream_id + channel.id
        fast_key = (stream_id, channel.id)
        mapped_channel = self._channel_lookup_fast.get(fast_key)
        if mapped_channel is not None:
            return mapped_channel

        # Map schema_id to the remapped value first (needed for dedup key)
        # schema_id=0 means no schema (valid per MCAP spec)
        new_schema_id = 0
        if channel.schema_id != 0:
            mapped_schema = self._schema_lookup_fast.get((stream_id, channel.schema_id))
            if mapped_schema is None:
                raise SchemaNotFoundError(
                    channel.schema_id, topic=channel.topic, stream_id=stream_id
                )
            new_schema_id = mapped_schema.id

        # Slow path: lookup by channel content (deduplication)
        metadata_key = tuple(sorted(channel.metadata.items())) if channel.metadata else ()
        slow_key = (channel.topic, channel.message_encoding, new_schema_id, metadata_key)
        mapped_channel = self._channel_lookup_slow.get(slow_key)
        if mapped_channel is not None:
            return self._remember_channel(stream_id, channel.id, mapped_channel)

        # Allocate ID, preserving original if possible
        new_id = _allocate_id(self._used_channel_ids, channel.id)

        # Direct construction is faster than dataclass.replace
        mapped_channel = Channel(
            id=new_id,
            schema_id=new_schema_id,
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            metadata=channel.metadata,
        )
        self._channel_lookup_slow[slow_key] = mapped_channel
        return self._remember_channel(stream_id, channel.id, mapped_channel)

    def was_channel_remapped(self, stream_id: int, original_id: int) -> bool:
        """Check if a channel ID was changed during remapping."""
        mapped = self._channel_lookup_fast.get((stream_id, original_id))
        return mapped is not None and mapped.id != original_id

    def has_channel(self, stream_id: int, original_id: int) -> bool:
        """Check if a channel has been seen and mapped."""
        return (stream_id, original_id) in self._channel_lookup_fast

    def get_remapped_channel(self, stream_id: int, original_id: int) -> Channel | None:
        """Get the remapped channel for a given stream and original ID."""
        return self._channel_lookup_fast.get((stream_id, original_id))

    # Messages
    def channel_id_map(self, stream_id: int) -> dict[int, int]:
        """Input→output channel-id mapping for one stream (empty until channels are seen).

        Lets bulk consumers apply the same remap as :meth:`remap_message` without a per-message
        call — e.g. re-serializing a whole chunk's messages under their output channel ids.
        """
        return {
            in_id: channel.id
            for (sid, in_id), channel in self._channel_lookup_fast.items()
            if sid == stream_id
        }

    def remap_message(self, stream_id: int, message: Message) -> Message:
        """Remap a message's channel ID based on the remapped channel."""
        # Use try/except - faster than .get() when key exists (common case)
        try:
            mapped_channel = self._channel_lookup_fast[stream_id, message.channel_id]
        except KeyError:
            raise ChannelNotFoundError(message.channel_id, stream_id=stream_id) from None
        new_channel_id = mapped_channel.id
        if new_channel_id == message.channel_id:
            return message
        # dataclass.replace is up to 4x slower than just creating a new instance
        return Message(
            channel_id=new_channel_id,
            sequence=message.sequence,
            log_time=message.log_time,
            publish_time=message.publish_time,
            data=message.data,
        )
