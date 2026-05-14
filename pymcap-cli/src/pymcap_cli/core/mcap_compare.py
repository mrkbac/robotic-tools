"""Shared fast MCAP comparison primitives."""

from __future__ import annotations

import hashlib
import operator
import struct
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from typing import IO, Literal, Protocol, cast
from urllib.parse import unquote, urlparse

from small_mcap import (
    Channel,
    Chunk,
    ChunkIndex,
    RebuildInfo,
    Schema,
    Summary,
    get_header,
    get_summary,
    read_info_approximate,
    rebuild_summary,
)
from small_mcap.reader import _get_chunk_data_stream
from small_mcap.records import OPCODE_AND_LEN_STRUCT, Opcode

from pymcap_cli.core.input_handler import open_input

_PayloadHashFn = Callable[[bytes | bytearray | memoryview], str]
try:
    from xxhash import xxh3_128_hexdigest as _imported_xxh3_128_hexdigest
except ImportError:
    _xxh3_128_hexdigest: _PayloadHashFn | None = None
else:
    _xxh3_128_hexdigest = cast("_PayloadHashFn", _imported_xxh3_128_hexdigest)


class HashSink(Protocol):
    def update(self, data: bytes | bytearray | memoryview, /) -> None: ...


ReadMode = Literal["summary", "index", "rebuild"]
_RECORD_HEADER_SIZE = OPCODE_AND_LEN_STRUCT.size
_MESSAGE_HEADER_STRUCT = struct.Struct("<HIQQ")
_MESSAGE_HEADER_SIZE = _MESSAGE_HEADER_STRUCT.size
_PAYLOAD_CHUNK_CACHE_SIZE = 8


class IndexReadProgress(Protocol):
    def __call__(self, completed_indexes: int, total_indexes: int) -> None: ...


@dataclass(frozen=True, order=True, slots=True)
class SchemaFingerprint:
    name: str
    encoding: str
    data_sha256: str


@dataclass(frozen=True, order=True, slots=True)
class ChannelFingerprint:
    topic: str
    message_encoding: str
    schema_digest: str
    metadata: tuple[tuple[str, str], ...]
    message_count: int


@dataclass(frozen=True, slots=True)
class SummaryChannelRange:
    channel_semantic_digest: str
    topic: str
    message_count: int
    message_start_time: int | None
    message_end_time: int | None


@dataclass(frozen=True, slots=True)
class McapIdentity:
    digest: str
    channel_digests: tuple[str, ...]
    channel_semantic_digests: tuple[str, ...]
    channel_ranges: tuple[SummaryChannelRange, ...]
    message_count: int
    message_start_time: int
    message_end_time: int
    schema_count: int
    channel_count: int


@dataclass(frozen=True, slots=True)
class MessageIndexIdentity:
    digest: str
    channel_digests: tuple[str, ...]
    indexed_channels: tuple[IndexedChannelIdentity, ...]
    message_count: int
    message_start_time: int
    message_end_time: int
    schema_count: int
    channel_count: int


@dataclass(frozen=True, slots=True)
class IdentityReadResult:
    path: str
    size_bytes: int
    identity: McapIdentity
    read_mode: ReadMode


@dataclass(frozen=True, order=True, slots=True)
class IndexedChannelIdentity:
    channel_semantic_digest: str
    topic: str
    message_count: int
    message_start_time: int
    message_end_time: int
    timestamps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MessageIndexIdentityReadResult:
    path: str
    size_bytes: int
    identity: MessageIndexIdentity
    read_mode: ReadMode


@dataclass(frozen=True, slots=True)
class CompareReadResult:
    path: str
    size_bytes: int
    info: RebuildInfo
    read_mode: ReadMode


class IndexedCompareKind(Enum):
    EXACT = "exact"
    FULL_OVERLAP = "full_overlap"
    LEFT_SUBSET = "left_subset"
    RIGHT_SUBSET = "right_subset"
    EDGE_OVERLAP = "edge_overlap"
    PARTIAL_OVERLAP = "partial_overlap"
    PAYLOAD_MISMATCH = "payload_mismatch"
    NO_OVERLAP = "no_overlap"


@dataclass(frozen=True, slots=True)
class PayloadMismatch:
    topic: str
    log_time: int
    reason: str


@dataclass(frozen=True, slots=True)
class TimestampRangePreview:
    start_time: int
    end_time: int
    message_count: int


@dataclass(frozen=True, slots=True)
class TimestampRangeSummary:
    ranges: tuple[TimestampRangePreview, ...]
    hidden_messages: int


@dataclass(frozen=True, slots=True)
class TopicOverlapEvidence:
    topic: str
    shared_messages: int
    left_only_messages: int
    right_only_messages: int
    shared_start_time: int | None
    shared_end_time: int | None
    left_only_ranges: TimestampRangeSummary
    right_only_ranges: TimestampRangeSummary

    def swapped(self) -> TopicOverlapEvidence:
        return TopicOverlapEvidence(
            topic=self.topic,
            shared_messages=self.shared_messages,
            left_only_messages=self.right_only_messages,
            right_only_messages=self.left_only_messages,
            shared_start_time=self.shared_start_time,
            shared_end_time=self.shared_end_time,
            left_only_ranges=self.right_only_ranges,
            right_only_ranges=self.left_only_ranges,
        )


@dataclass(frozen=True, slots=True)
class IndexedOverlap:
    shared_channels: int
    shared_messages: int
    topics: tuple[TopicOverlapEvidence, ...]
    is_edge_overlap: bool


@dataclass(frozen=True, slots=True)
class IndexedComparison:
    left: MessageIndexIdentityReadResult
    right: MessageIndexIdentityReadResult
    kind: IndexedCompareKind
    overlap: IndexedOverlap
    payload_mismatch: PayloadMismatch | None = None

    @property
    def shared_channels(self) -> int:
        return self.overlap.shared_channels

    @property
    def shared_messages(self) -> int:
        return self.overlap.shared_messages

    @property
    def topics(self) -> tuple[TopicOverlapEvidence, ...]:
        return self.overlap.topics

    @property
    def left_extra_messages(self) -> int:
        return self.left.identity.message_count - self.shared_messages

    @property
    def right_extra_messages(self) -> int:
        return self.right.identity.message_count - self.shared_messages

    @property
    def has_overlap(self) -> bool:
        return self.shared_channels > 0 and self.shared_messages > 0


@dataclass(frozen=True, slots=True)
class _IndexedChannelView:
    topic: str
    timestamps: Counter[int]


@dataclass(frozen=True, order=True, slots=True)
class _PayloadLocator:
    log_time: int
    chunk_start_offset: int
    offset: int
    channel_id: int


@dataclass(frozen=True, order=True, slots=True)
class _PayloadRecordKey:
    publish_time: int
    sequence: int
    payload_size: int
    payload_digest: str


@dataclass(frozen=True, slots=True)
class PayloadLookup:
    path: str
    size_bytes: int
    identity: MessageIndexIdentity
    read_mode: ReadMode
    chunk_indexes: dict[int, ChunkIndex]
    topics_by_digest: dict[str, str]
    locators_by_digest: dict[str, tuple[_PayloadLocator, ...]]


@dataclass(slots=True)
class _TopicEvidenceAccumulator:
    topic: str
    shared_timestamps: Counter[int]
    left_only_timestamps: Counter[int]
    right_only_timestamps: Counter[int]

    @property
    def shared_messages(self) -> int:
        return _counter_message_count(self.shared_timestamps)


def _update_str(hasher: HashSink, value: str) -> None:
    data = value.encode("utf-8")
    hasher.update(b"s:%d:" % len(data))
    hasher.update(data)
    hasher.update(b";")


def _update_int(hasher: HashSink, value: int) -> None:
    hasher.update(b"i:%d;" % value)


def _update_schema(hasher: HashSink, schema: SchemaFingerprint) -> None:
    _update_str(hasher, "schema")
    _update_str(hasher, schema.name)
    _update_str(hasher, schema.encoding)
    _update_str(hasher, schema.data_sha256)


def _update_channel(hasher: HashSink, channel: ChannelFingerprint) -> None:
    _update_str(hasher, "channel")
    _update_str(hasher, channel.topic)
    _update_str(hasher, channel.message_encoding)
    _update_str(hasher, channel.schema_digest)
    _update_int(hasher, channel.message_count)
    _update_int(hasher, len(channel.metadata))
    for key, value in channel.metadata:
        _update_str(hasher, key)
        _update_str(hasher, value)


def _schema_fingerprint(schema: Schema) -> SchemaFingerprint:
    return SchemaFingerprint(
        name=schema.name,
        encoding=schema.encoding,
        data_sha256=hashlib.sha256(schema.data).hexdigest(),
    )


def _schema_digest(schema: SchemaFingerprint) -> str:
    hasher = hashlib.sha256()
    _update_schema(hasher, schema)
    return hasher.hexdigest()


def _channel_digest(channel: ChannelFingerprint) -> str:
    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.channel.v1")
    _update_channel(hasher, channel)
    return hasher.hexdigest()


def _channel_semantic_digest(channel: ChannelFingerprint) -> str:
    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.channel-semantic.v1")
    _update_str(hasher, channel.topic)
    _update_str(hasher, channel.message_encoding)
    _update_str(hasher, channel.schema_digest)
    _update_int(hasher, len(channel.metadata))
    for key, value in channel.metadata:
        _update_str(hasher, key)
        _update_str(hasher, value)
    return hasher.hexdigest()


def _encode_timestamps(timestamps: list[int]) -> bytes:
    return b"".join(b"i:%d;" % ts for ts in timestamps)


def _indexed_channel_digest(channel: ChannelFingerprint, ts_count: int, ts_bytes: bytes) -> str:
    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.indexed-channel.v1")
    _update_channel(hasher, channel)
    _update_int(hasher, ts_count)
    hasher.update(ts_bytes)
    return hasher.hexdigest()


def _unknown_indexed_channel_digest(channel_id: int, ts_count: int, ts_bytes: bytes) -> str:
    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.unknown-indexed-channel.v1")
    _update_int(hasher, channel_id)
    _update_int(hasher, ts_count)
    hasher.update(ts_bytes)
    return hasher.hexdigest()


def _channel_fingerprint(
    channel: Channel,
    schemas_by_id: dict[int, SchemaFingerprint],
    channel_message_counts: dict[int, int],
) -> ChannelFingerprint:
    schema = schemas_by_id.get(channel.schema_id)
    schema_digest = _schema_digest(schema) if schema is not None else ""
    return ChannelFingerprint(
        topic=channel.topic,
        message_encoding=channel.message_encoding,
        schema_digest=schema_digest,
        metadata=tuple(sorted(channel.metadata.items())),
        message_count=channel_message_counts.get(channel.id, 0),
    )


def _unknown_channel_topic(channel_id: int) -> str:
    return f"Channel_{channel_id}"


def _blake2b_128_hexdigest(data: bytes | bytearray | memoryview) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(data)
    return hasher.hexdigest()


@cache
def _payload_hash_fn() -> _PayloadHashFn:
    if _xxh3_128_hexdigest is not None:
        return _xxh3_128_hexdigest
    return _blake2b_128_hexdigest


def _payload_digest(data: bytes | bytearray | memoryview) -> str:
    return _payload_hash_fn()(data)


@dataclass(frozen=True, slots=True)
class _ChannelIntermediate:
    channel_id: int
    fingerprint: ChannelFingerprint
    semantic_digest: str
    digest: str


@dataclass(frozen=True, slots=True)
class _IdentityIntermediates:
    schemas: tuple[SchemaFingerprint, ...]
    channels: tuple[_ChannelIntermediate, ...]
    unknown_counts: tuple[tuple[int, int], ...]


def _build_identity_intermediates(info: RebuildInfo) -> _IdentityIntermediates:
    summary = info.summary
    statistics = summary.statistics
    if statistics is None:
        raise ValueError("missing statistics")

    schemas_by_id = {
        schema_id: _schema_fingerprint(schema) for schema_id, schema in summary.schemas.items()
    }
    channels: list[_ChannelIntermediate] = []
    for channel in summary.channels.values():
        fingerprint = _channel_fingerprint(
            channel, schemas_by_id, statistics.channel_message_counts
        )
        channels.append(
            _ChannelIntermediate(
                channel_id=channel.id,
                fingerprint=fingerprint,
                semantic_digest=_channel_semantic_digest(fingerprint),
                digest=_channel_digest(fingerprint),
            )
        )
    channels.sort(key=lambda entry: entry.fingerprint)

    known_channel_ids = set(summary.channels)
    unknown_counts = sorted(
        (channel_id, count)
        for channel_id, count in statistics.channel_message_counts.items()
        if channel_id not in known_channel_ids
    )

    return _IdentityIntermediates(
        schemas=tuple(sorted(schemas_by_id.values())),
        channels=tuple(channels),
        unknown_counts=tuple(unknown_counts),
    )


def _summary_channel_ranges(
    summary: Summary,
    channels: tuple[_ChannelIntermediate, ...],
    channel_message_counts: dict[int, int],
) -> tuple[SummaryChannelRange, ...]:
    start_by_channel: dict[int, int] = {}
    end_by_channel: dict[int, int] = {}

    for chunk_index in summary.chunk_indexes:
        for channel_id in chunk_index.message_index_offsets:
            if channel_id not in start_by_channel:
                start_by_channel[channel_id] = chunk_index.message_start_time
                end_by_channel[channel_id] = chunk_index.message_end_time
                continue
            start_by_channel[channel_id] = min(
                start_by_channel[channel_id], chunk_index.message_start_time
            )
            end_by_channel[channel_id] = max(
                end_by_channel[channel_id], chunk_index.message_end_time
            )

    entries_by_id = {entry.channel_id: entry for entry in channels}
    ranges: list[SummaryChannelRange] = []
    for channel_id, message_count in channel_message_counts.items():
        entry = entries_by_id.get(channel_id)
        if entry is not None:
            semantic_digest = entry.semantic_digest
            topic = entry.fingerprint.topic
        else:
            semantic_digest = _unknown_indexed_channel_digest(channel_id, 0, b"")
            topic = _unknown_channel_topic(channel_id)
        ranges.append(
            SummaryChannelRange(
                channel_semantic_digest=semantic_digest,
                topic=topic,
                message_count=message_count,
                message_start_time=start_by_channel.get(channel_id),
                message_end_time=end_by_channel.get(channel_id),
            )
        )

    return tuple(sorted(ranges, key=lambda item: (item.channel_semantic_digest, item.topic)))


def _identity_from_intermediates(
    info: RebuildInfo, intermediates: _IdentityIntermediates
) -> McapIdentity:
    statistics = info.summary.statistics
    assert statistics is not None

    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.identity.v1")
    _update_str(hasher, info.header.profile)

    _update_int(hasher, statistics.message_count)
    _update_int(hasher, statistics.message_start_time)
    _update_int(hasher, statistics.message_end_time)
    _update_int(hasher, statistics.schema_count)
    _update_int(hasher, statistics.channel_count)
    _update_int(hasher, statistics.attachment_count)
    _update_int(hasher, statistics.metadata_count)

    _update_int(hasher, len(intermediates.schemas))
    for schema in intermediates.schemas:
        _update_schema(hasher, schema)

    _update_int(hasher, len(intermediates.channels))
    for entry in intermediates.channels:
        _update_channel(hasher, entry.fingerprint)

    _update_int(hasher, len(intermediates.unknown_counts))
    for channel_id, count in intermediates.unknown_counts:
        _update_int(hasher, channel_id)
        _update_int(hasher, count)

    channel_digests = tuple(sorted(entry.digest for entry in intermediates.channels))
    channel_semantic_digests = tuple(
        sorted(entry.semantic_digest for entry in intermediates.channels)
    )
    channel_ranges = _summary_channel_ranges(
        info.summary, intermediates.channels, statistics.channel_message_counts
    )

    return McapIdentity(
        digest=hasher.hexdigest(),
        channel_digests=channel_digests,
        channel_semantic_digests=channel_semantic_digests,
        channel_ranges=channel_ranges,
        message_count=statistics.message_count,
        message_start_time=statistics.message_start_time,
        message_end_time=statistics.message_end_time,
        schema_count=statistics.schema_count,
        channel_count=statistics.channel_count,
    )


def recording_identity_from_info(info: RebuildInfo) -> McapIdentity:
    return _identity_from_intermediates(info, _build_identity_intermediates(info))


def message_index_identity_from_info(info: RebuildInfo) -> MessageIndexIdentity:
    intermediates = _build_identity_intermediates(info)
    summary_identity = _identity_from_intermediates(info, intermediates)

    timestamps_by_channel: dict[int, list[int]] = {}
    if info.chunk_information:
        for indexes in info.chunk_information.values():
            for message_index in indexes:
                timestamps_by_channel.setdefault(message_index.channel_id, []).extend(
                    message_index.timestamps
                )

    indexed_entries = [
        (entry, sorted(timestamps_by_channel.pop(entry.channel_id, [])))
        for entry in intermediates.channels
    ]
    indexed_entries.sort(key=lambda item: (item[0].fingerprint, item[1]))

    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.message-index.v1")
    _update_str(hasher, summary_identity.digest)
    _update_int(hasher, len(indexed_entries))
    indexed_channel_digests: list[str] = []
    indexed_channels: list[IndexedChannelIdentity] = []
    for entry, timestamps in indexed_entries:
        ts_count = len(timestamps)
        ts_bytes = _encode_timestamps(timestamps)
        _update_channel(hasher, entry.fingerprint)
        _update_int(hasher, ts_count)
        hasher.update(ts_bytes)
        indexed_channel_digests.append(
            _indexed_channel_digest(entry.fingerprint, ts_count, ts_bytes)
        )
        indexed_channels.append(
            IndexedChannelIdentity(
                channel_semantic_digest=entry.semantic_digest,
                topic=entry.fingerprint.topic,
                message_count=ts_count,
                message_start_time=timestamps[0] if timestamps else 0,
                message_end_time=timestamps[-1] if timestamps else 0,
                timestamps=tuple(timestamps),
            )
        )

    unknown_entries = sorted(
        (channel_id, sorted(values)) for channel_id, values in timestamps_by_channel.items()
    )
    _update_int(hasher, len(unknown_entries))
    for channel_id, timestamps in unknown_entries:
        ts_count = len(timestamps)
        ts_bytes = _encode_timestamps(timestamps)
        _update_int(hasher, channel_id)
        _update_int(hasher, ts_count)
        hasher.update(ts_bytes)
        indexed_channel_digests.append(
            _unknown_indexed_channel_digest(channel_id, ts_count, ts_bytes)
        )
        indexed_channels.append(
            IndexedChannelIdentity(
                channel_semantic_digest=_unknown_indexed_channel_digest(channel_id, 0, b""),
                topic=_unknown_channel_topic(channel_id),
                message_count=ts_count,
                message_start_time=timestamps[0] if timestamps else 0,
                message_end_time=timestamps[-1] if timestamps else 0,
                timestamps=tuple(timestamps),
            )
        )

    return MessageIndexIdentity(
        digest=hasher.hexdigest(),
        channel_digests=tuple(sorted(indexed_channel_digests)),
        indexed_channels=tuple(sorted(indexed_channels)),
        message_count=summary_identity.message_count,
        message_start_time=summary_identity.message_start_time,
        message_end_time=summary_identity.message_end_time,
        schema_count=summary_identity.schema_count,
        channel_count=summary_identity.channel_count,
    )


def payload_lookup_from_info(
    path: str,
    size_bytes: int,
    info: RebuildInfo,
    read_mode: ReadMode,
) -> PayloadLookup:
    intermediates = _build_identity_intermediates(info)
    identity = message_index_identity_from_info(info)
    semantic_by_channel_id = {
        entry.channel_id: entry.semantic_digest for entry in intermediates.channels
    }
    topic_by_channel_id = {
        entry.channel_id: entry.fingerprint.topic for entry in intermediates.channels
    }

    locators_by_digest: dict[str, list[_PayloadLocator]] = defaultdict(list)
    topics_by_digest: dict[str, str] = {}
    chunk_indexes = {
        chunk_index.chunk_start_offset: chunk_index for chunk_index in info.summary.chunk_indexes
    }

    for chunk_index in info.summary.chunk_indexes:
        for message_index in (info.chunk_information or {}).get(chunk_index.chunk_start_offset, ()):
            channel_id = message_index.channel_id
            channel_digest = semantic_by_channel_id.get(channel_id)
            topic = topic_by_channel_id.get(channel_id)
            if channel_digest is None or topic is None:
                channel_digest = _unknown_indexed_channel_digest(channel_id, 0, b"")
                topic = _unknown_channel_topic(channel_id)
            topics_by_digest.setdefault(channel_digest, topic)
            for log_time, offset in zip(
                message_index.timestamps, message_index.offsets, strict=True
            ):
                locators_by_digest[channel_digest].append(
                    _PayloadLocator(
                        log_time=log_time,
                        chunk_start_offset=chunk_index.chunk_start_offset,
                        offset=offset,
                        channel_id=channel_id,
                    )
                )

    return PayloadLookup(
        path=path,
        size_bytes=size_bytes,
        identity=identity,
        read_mode=read_mode,
        chunk_indexes=chunk_indexes,
        topics_by_digest=topics_by_digest,
        locators_by_digest={
            digest: tuple(sorted(locators)) for digest, locators in locators_by_digest.items()
        },
    )


class _PayloadReader:
    def __init__(self, stream: IO[bytes], chunk_indexes: dict[int, ChunkIndex]) -> None:
        self._stream = stream
        self._chunk_indexes = chunk_indexes
        self._chunk_cache: OrderedDict[int, bytes | memoryview] = OrderedDict()

    def _load_chunk_data(self, chunk_start_offset: int) -> bytes | memoryview:
        cached = self._chunk_cache.get(chunk_start_offset)
        if cached is not None:
            self._chunk_cache.move_to_end(chunk_start_offset)
            return cached

        chunk_index = self._chunk_indexes[chunk_start_offset]
        self._stream.seek(chunk_start_offset)
        raw = self._stream.read(chunk_index.chunk_length)
        if len(raw) != chunk_index.chunk_length:
            raise ValueError(
                f"short read for chunk at {chunk_start_offset}: "
                f"expected {chunk_index.chunk_length} bytes, got {len(raw)}"
            )

        view = memoryview(raw)
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack_from(view, 0)
        if opcode != Opcode.CHUNK:
            raise ValueError(f"expected Chunk at {chunk_start_offset}, got opcode {opcode}")
        if length + _RECORD_HEADER_SIZE > len(view):
            raise ValueError(f"truncated Chunk body at {chunk_start_offset}")

        chunk = Chunk.read(view[_RECORD_HEADER_SIZE : _RECORD_HEADER_SIZE + length])
        data = _get_chunk_data_stream(chunk, validate_crc=True)
        self._chunk_cache[chunk_start_offset] = data
        if len(self._chunk_cache) > _PAYLOAD_CHUNK_CACHE_SIZE:
            self._chunk_cache.popitem(last=False)
        return data

    def payload_key(self, locator: _PayloadLocator) -> _PayloadRecordKey:
        data = self._load_chunk_data(locator.chunk_start_offset)
        view = memoryview(data)
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack_from(view, locator.offset)
        if opcode != Opcode.MESSAGE:
            raise ValueError(
                f"expected Message at chunk {locator.chunk_start_offset} "
                f"offset {locator.offset}, got opcode {opcode}"
            )

        body_start = locator.offset + _RECORD_HEADER_SIZE
        body_end = body_start + length
        if body_end > len(view):
            raise ValueError(
                f"truncated Message at chunk {locator.chunk_start_offset} offset {locator.offset}"
            )

        channel_id, sequence, log_time, publish_time = _MESSAGE_HEADER_STRUCT.unpack_from(
            view, body_start
        )
        if channel_id != locator.channel_id:
            raise ValueError(
                f"message index channel_id={locator.channel_id} points to channel_id={channel_id}"
            )
        if log_time != locator.log_time:
            raise ValueError(
                f"message index log_time={locator.log_time} points to log_time={log_time}"
            )

        payload = view[body_start + _MESSAGE_HEADER_SIZE : body_end]
        return _PayloadRecordKey(
            publish_time=publish_time,
            sequence=sequence,
            payload_size=len(payload),
            payload_digest=_payload_digest(payload),
        )


def _topic_for_digest(
    digest: str,
    left: PayloadLookup,
    right: PayloadLookup,
) -> str:
    topic = left.topics_by_digest.get(digest)
    if topic is not None:
        return topic
    return right.topics_by_digest.get(digest, digest[:16])


def _same_log_time_group(
    locators: tuple[_PayloadLocator, ...],
    start: int,
) -> tuple[int, tuple[_PayloadLocator, ...], int]:
    log_time = locators[start].log_time
    end = start + 1
    while end < len(locators) and locators[end].log_time == log_time:
        end += 1
    return log_time, locators[start:end], end


def _compare_payload_groups(
    *,
    topic: str,
    log_time: int,
    left_group: tuple[_PayloadLocator, ...],
    right_group: tuple[_PayloadLocator, ...],
    left_reader: _PayloadReader,
    right_reader: _PayloadReader,
) -> PayloadMismatch | None:
    if len(left_group) != len(right_group):
        return PayloadMismatch(
            topic=topic,
            log_time=log_time,
            reason=(
                f"message count differs at timestamp: "
                f"{len(left_group)} left vs {len(right_group)} right"
            ),
        )

    left_payloads = Counter(left_reader.payload_key(locator) for locator in left_group)
    for locator in right_group:
        key = right_reader.payload_key(locator)
        count = left_payloads.get(key, 0)
        if count == 0:
            return PayloadMismatch(
                topic=topic,
                log_time=log_time,
                reason="payload, publish_time, or sequence differs",
            )
        if count == 1:
            del left_payloads[key]
        else:
            left_payloads[key] = count - 1

    if left_payloads:
        return PayloadMismatch(
            topic=topic,
            log_time=log_time,
            reason="payload, publish_time, or sequence differs",
        )
    return None


def verify_payloads_match(left: PayloadLookup, right: PayloadLookup) -> PayloadMismatch | None:
    if left.identity.message_count != right.identity.message_count:
        return PayloadMismatch(
            topic="",
            log_time=0,
            reason=(
                f"message count differs: {left.identity.message_count} left vs "
                f"{right.identity.message_count} right"
            ),
        )

    all_digests = sorted(left.locators_by_digest.keys() | right.locators_by_digest.keys())
    with (
        open_input(left.path, buffering=0) as (left_stream, _left_size),
        open_input(right.path, buffering=0) as (right_stream, _right_size),
    ):
        left_reader = _PayloadReader(left_stream, left.chunk_indexes)
        right_reader = _PayloadReader(right_stream, right.chunk_indexes)
        for digest in all_digests:
            topic = _topic_for_digest(digest, left, right)
            left_locators = left.locators_by_digest.get(digest, ())
            right_locators = right.locators_by_digest.get(digest, ())
            if not left_locators or not right_locators:
                return PayloadMismatch(
                    topic=topic,
                    log_time=0,
                    reason="channel is missing from one side",
                )

            left_index = 0
            right_index = 0
            while left_index < len(left_locators) and right_index < len(right_locators):
                left_log_time, left_group, left_index = _same_log_time_group(
                    left_locators, left_index
                )
                right_log_time, right_group, right_index = _same_log_time_group(
                    right_locators, right_index
                )
                if left_log_time != right_log_time:
                    return PayloadMismatch(
                        topic=topic,
                        log_time=min(left_log_time, right_log_time),
                        reason=(
                            f"log_time differs: {left_log_time} left vs {right_log_time} right"
                        ),
                    )

                mismatch = _compare_payload_groups(
                    topic=topic,
                    log_time=left_log_time,
                    left_group=left_group,
                    right_group=right_group,
                    left_reader=left_reader,
                    right_reader=right_reader,
                )
                if mismatch is not None:
                    return mismatch

            if left_index != len(left_locators) or right_index != len(right_locators):
                return PayloadMismatch(
                    topic=topic,
                    log_time=0,
                    reason="message count differs",
                )

    return None


def _counter_message_count(counter: Counter[int]) -> int:
    return sum(counter.values())


def split_timestamps_into_segments(
    sorted_timestamps: list[int], gap_multiplier: float = 3.0
) -> list[list[int]]:
    if len(sorted_timestamps) <= 1:
        return [sorted_timestamps[:]] if sorted_timestamps else []

    gaps = [
        sorted_timestamps[index + 1] - sorted_timestamps[index]
        for index in range(len(sorted_timestamps) - 1)
    ]
    median_gap = sorted(gaps)[len(gaps) // 2]
    threshold = median_gap * gap_multiplier

    segments: list[list[int]] = []
    current = [sorted_timestamps[0]]
    for index, gap in enumerate(gaps):
        if gap > threshold:
            segments.append(current)
            current = [sorted_timestamps[index + 1]]
        else:
            current.append(sorted_timestamps[index + 1])
    segments.append(current)
    return segments


def _counter_range_summary(counter: Counter[int], *, max_ranges: int) -> TimestampRangeSummary:
    if not counter:
        return TimestampRangeSummary(ranges=(), hidden_messages=0)

    segments = split_timestamps_into_segments(sorted(counter))
    ranges: list[TimestampRangePreview] = []
    displayed_messages = 0
    display_limit = max(max_ranges, 0)
    for segment in segments[:display_limit]:
        message_count = sum(counter[timestamp] for timestamp in segment)
        displayed_messages += message_count
        ranges.append(
            TimestampRangePreview(
                start_time=segment[0],
                end_time=segment[-1],
                message_count=message_count,
            )
        )

    return TimestampRangeSummary(
        ranges=tuple(ranges),
        hidden_messages=_counter_message_count(counter) - displayed_messages,
    )


def _channel_views_by_digest(
    channels: tuple[IndexedChannelIdentity, ...],
) -> dict[str, list[_IndexedChannelView]]:
    result: dict[str, list[_IndexedChannelView]] = defaultdict(list)
    for channel in channels:
        result[channel.channel_semantic_digest].append(
            _IndexedChannelView(topic=channel.topic, timestamps=Counter(channel.timestamps))
        )
    return result


def _channel_pair_overlap(
    left_channel: _IndexedChannelView,
    right_channel: _IndexedChannelView,
) -> _TopicEvidenceAccumulator | None:
    shared_timestamps = left_channel.timestamps & right_channel.timestamps
    if not shared_timestamps:
        return None

    return _TopicEvidenceAccumulator(
        topic=left_channel.topic,
        shared_timestamps=shared_timestamps,
        left_only_timestamps=left_channel.timestamps - shared_timestamps,
        right_only_timestamps=right_channel.timestamps - shared_timestamps,
    )


def _accumulate_channel_overlap(
    accumulators: dict[str, _TopicEvidenceAccumulator],
    overlap: _TopicEvidenceAccumulator,
) -> None:
    accumulator = accumulators.get(overlap.topic)
    if accumulator is None:
        accumulators[overlap.topic] = overlap
        return
    accumulator.shared_timestamps.update(overlap.shared_timestamps)
    accumulator.left_only_timestamps.update(overlap.left_only_timestamps)
    accumulator.right_only_timestamps.update(overlap.right_only_timestamps)


def _relative_side(counter: Counter[int], start_time: int, end_time: int) -> str:
    if not counter:
        return "none"

    has_before = False
    has_after = False
    has_inside = False
    for timestamp in counter:
        if timestamp < start_time:
            has_before = True
        elif timestamp > end_time:
            has_after = True
        else:
            has_inside = True

    if has_inside or (has_before and has_after):
        return "mixed"
    if has_before:
        return "before"
    if has_after:
        return "after"
    return "none"


def _is_edge_overlap(accumulators: dict[str, _TopicEvidenceAccumulator]) -> bool:
    orientation: tuple[str, str] | None = None
    for accumulator in accumulators.values():
        if not accumulator.shared_timestamps:
            continue

        shared_times = sorted(accumulator.shared_timestamps)
        shared_start = shared_times[0]
        shared_end = shared_times[-1]
        left_side = _relative_side(accumulator.left_only_timestamps, shared_start, shared_end)
        right_side = _relative_side(accumulator.right_only_timestamps, shared_start, shared_end)

        if left_side == "none" or right_side == "none":
            continue
        if (left_side, right_side) not in {("before", "after"), ("after", "before")}:
            return False
        if orientation is None:
            orientation = (left_side, right_side)
        elif orientation != (left_side, right_side):
            return False

    return orientation is not None


def _topic_evidence_from_accumulators(
    accumulators: dict[str, _TopicEvidenceAccumulator],
    *,
    max_range_preview: int,
) -> tuple[TopicOverlapEvidence, ...]:
    evidence: list[TopicOverlapEvidence] = []
    for accumulator in accumulators.values():
        shared_times = sorted(accumulator.shared_timestamps)
        evidence.append(
            TopicOverlapEvidence(
                topic=accumulator.topic,
                shared_messages=_counter_message_count(accumulator.shared_timestamps),
                left_only_messages=_counter_message_count(accumulator.left_only_timestamps),
                right_only_messages=_counter_message_count(accumulator.right_only_timestamps),
                shared_start_time=shared_times[0] if shared_times else None,
                shared_end_time=shared_times[-1] if shared_times else None,
                left_only_ranges=_counter_range_summary(
                    accumulator.left_only_timestamps,
                    max_ranges=max_range_preview,
                ),
                right_only_ranges=_counter_range_summary(
                    accumulator.right_only_timestamps,
                    max_ranges=max_range_preview,
                ),
            )
        )

    return tuple(
        sorted(
            evidence,
            key=lambda item: (
                -item.shared_messages,
                -(item.left_only_messages + item.right_only_messages),
                item.topic,
            ),
        )
    )


def indexed_overlap(
    left_channels: tuple[IndexedChannelIdentity, ...],
    right_channels: tuple[IndexedChannelIdentity, ...],
    *,
    max_range_preview: int = 1,
) -> IndexedOverlap:
    left_by_digest = _channel_views_by_digest(left_channels)
    right_by_digest = _channel_views_by_digest(right_channels)

    shared_channels = 0
    shared_messages_total = 0
    evidence_by_topic: dict[str, _TopicEvidenceAccumulator] = {}
    candidate_pairs: list[tuple[int, str, int, int, _TopicEvidenceAccumulator]] = []

    for channel_digest in left_by_digest.keys() & right_by_digest.keys():
        left_group = left_by_digest[channel_digest]
        right_group = right_by_digest[channel_digest]
        if len(left_group) == 1 and len(right_group) == 1:
            overlap = _channel_pair_overlap(left_group[0], right_group[0])
            if overlap is not None:
                shared_channels += 1
                shared_messages_total += overlap.shared_messages
                _accumulate_channel_overlap(evidence_by_topic, overlap)
            continue

        for left_index, left_channel in enumerate(left_group):
            for right_index, right_channel in enumerate(right_group):
                overlap = _channel_pair_overlap(left_channel, right_channel)
                if overlap is not None:
                    candidate_pairs.append(
                        (
                            overlap.shared_messages,
                            channel_digest,
                            left_index,
                            right_index,
                            overlap,
                        )
                    )

    used_left: set[tuple[str, int]] = set()
    used_right: set[tuple[str, int]] = set()
    for shared_messages, channel_digest, left_index, right_index, overlap in sorted(
        candidate_pairs,
        key=operator.itemgetter(0, 1, 2, 3),
        reverse=True,
    ):
        left_key = (channel_digest, left_index)
        right_key = (channel_digest, right_index)
        if left_key in used_left or right_key in used_right:
            continue
        used_left.add(left_key)
        used_right.add(right_key)
        shared_channels += 1
        shared_messages_total += shared_messages
        _accumulate_channel_overlap(evidence_by_topic, overlap)

    return IndexedOverlap(
        shared_channels=shared_channels,
        shared_messages=shared_messages_total,
        topics=_topic_evidence_from_accumulators(
            evidence_by_topic,
            max_range_preview=max_range_preview,
        ),
        is_edge_overlap=_is_edge_overlap(evidence_by_topic),
    )


def message_bearing_channel_count(identity: MessageIndexIdentity) -> int:
    return sum(1 for channel in identity.indexed_channels if channel.message_count)


def _is_full_index_overlap(
    left: MessageIndexIdentity,
    right: MessageIndexIdentity,
    overlap: IndexedOverlap,
    *,
    left_message_bearing_channels: int,
    right_message_bearing_channels: int,
) -> bool:
    return (
        overlap.shared_messages == left.message_count
        and overlap.shared_messages == right.message_count
        and overlap.shared_channels == left_message_bearing_channels
        and overlap.shared_channels == right_message_bearing_channels
    )


def _compare_kind(
    left: MessageIndexIdentity,
    right: MessageIndexIdentity,
    overlap: IndexedOverlap,
    *,
    left_message_bearing_channels: int,
    right_message_bearing_channels: int,
) -> IndexedCompareKind:
    if left.digest == right.digest:
        return IndexedCompareKind.EXACT

    if _is_full_index_overlap(
        left,
        right,
        overlap,
        left_message_bearing_channels=left_message_bearing_channels,
        right_message_bearing_channels=right_message_bearing_channels,
    ):
        return IndexedCompareKind.FULL_OVERLAP

    if overlap.shared_channels == 0 or overlap.shared_messages == 0:
        return IndexedCompareKind.NO_OVERLAP

    if (
        overlap.shared_messages == left.message_count
        and overlap.shared_channels == left_message_bearing_channels
    ):
        return IndexedCompareKind.LEFT_SUBSET

    if (
        overlap.shared_messages == right.message_count
        and overlap.shared_channels == right_message_bearing_channels
    ):
        return IndexedCompareKind.RIGHT_SUBSET

    if overlap.is_edge_overlap:
        return IndexedCompareKind.EDGE_OVERLAP

    return IndexedCompareKind.PARTIAL_OVERLAP


def compare_indexed_identities(
    left: MessageIndexIdentityReadResult,
    right: MessageIndexIdentityReadResult,
    *,
    max_range_preview: int = 1,
) -> IndexedComparison:
    overlap = indexed_overlap(
        left.identity.indexed_channels,
        right.identity.indexed_channels,
        max_range_preview=max_range_preview,
    )
    left_message_bearing_channels = message_bearing_channel_count(left.identity)
    right_message_bearing_channels = message_bearing_channel_count(right.identity)
    return IndexedComparison(
        left=left,
        right=right,
        kind=_compare_kind(
            left.identity,
            right.identity,
            overlap,
            left_message_bearing_channels=left_message_bearing_channels,
            right_message_bearing_channels=right_message_bearing_channels,
        ),
        overlap=overlap,
    )


def verify_comparison_payloads(
    comparison: IndexedComparison,
    left: PayloadLookup,
    right: PayloadLookup,
) -> IndexedComparison:
    if comparison.kind not in {IndexedCompareKind.EXACT, IndexedCompareKind.FULL_OVERLAP}:
        return comparison

    mismatch = verify_payloads_match(left, right)
    if mismatch is None:
        return comparison
    return IndexedComparison(
        left=comparison.left,
        right=comparison.right,
        kind=IndexedCompareKind.PAYLOAD_MISMATCH,
        overlap=comparison.overlap,
        payload_mismatch=mismatch,
    )


def read_summary_info(
    stream: IO[bytes], *, rebuild_missing: bool
) -> tuple[RebuildInfo, ReadMode] | None:
    header = get_header(stream)
    summary = get_summary(stream)
    if summary is not None and summary.statistics is not None:
        return RebuildInfo(header=header, summary=summary), "summary"

    if not rebuild_missing:
        return None

    stream.seek(0)
    rebuilt = rebuild_summary(
        stream,
        validate_crc=False,
        calculate_channel_sizes=False,
        exact_sizes=False,
    )
    return rebuilt, "rebuild"


def _indexed_message_count(info: RebuildInfo) -> int:
    if not info.chunk_information:
        return 0
    return sum(
        len(message_index.timestamps)
        for indexes in info.chunk_information.values()
        for message_index in indexes
    )


def _has_complete_message_indexes(info: RebuildInfo) -> bool:
    statistics = info.summary.statistics
    if statistics is None:
        return False
    if statistics.message_count == 0:
        return True
    return _indexed_message_count(info) == statistics.message_count


def read_indexed_info(
    stream: IO[bytes],
    *,
    rebuild_missing: bool,
    index_progress: IndexReadProgress | None = None,
) -> tuple[RebuildInfo, ReadMode] | None:
    info = read_info_approximate(stream, progress_callback=index_progress)
    if (
        info is not None
        and info.summary.statistics is not None
        and _has_complete_message_indexes(info)
    ):
        return info, "index"

    if not rebuild_missing:
        return None

    stream.seek(0)
    rebuilt = rebuild_summary(
        stream,
        validate_crc=False,
        calculate_channel_sizes=False,
        exact_sizes=False,
    )
    if not _has_complete_message_indexes(rebuilt):
        return None
    return rebuilt, "rebuild"


def read_identity_file(path: str, *, rebuild_missing: bool) -> IdentityReadResult | None:
    with open_input(path, buffering=0) as (stream, size_bytes):
        result = read_summary_info(stream, rebuild_missing=rebuild_missing)
        if result is None:
            return None
        info, read_mode = result
        return IdentityReadResult(
            path=path,
            size_bytes=size_bytes,
            identity=recording_identity_from_info(info),
            read_mode=read_mode,
        )


def read_message_index_identity_file(
    path: str,
    *,
    rebuild_missing: bool,
    index_progress: IndexReadProgress | None = None,
) -> MessageIndexIdentityReadResult | None:
    with open_input(path, buffering=0) as (stream, size_bytes):
        result = read_indexed_info(
            stream,
            rebuild_missing=rebuild_missing,
            index_progress=index_progress,
        )
        if result is None:
            return None
        info, read_mode = result
        return MessageIndexIdentityReadResult(
            path=path,
            size_bytes=size_bytes,
            identity=message_index_identity_from_info(info),
            read_mode=read_mode,
        )


def read_payload_lookup_file(path: str, *, rebuild_missing: bool) -> PayloadLookup | None:
    with open_input(path, buffering=0) as (stream, size_bytes):
        result = read_indexed_info(stream, rebuild_missing=rebuild_missing)
        if result is None:
            return None
        info, read_mode = result
        return payload_lookup_from_info(path, size_bytes, info, read_mode)


def read_compare_file(path: str, *, rebuild_missing: bool = True) -> CompareReadResult:
    with open_input(path, buffering=0) as (stream, size_bytes):
        result = read_indexed_info(stream, rebuild_missing=rebuild_missing)
        if result is None:
            raise ValueError("no complete message indexes")
        info, read_mode = result
        return CompareReadResult(
            path=path,
            size_bytes=size_bytes,
            info=info,
            read_mode=read_mode,
        )


def collect_message_timestamps(info: RebuildInfo) -> dict[int, set[int]]:
    timestamps_by_channel: dict[int, set[int]] = {}
    if not info.chunk_information:
        return timestamps_by_channel
    for msg_idx_list in info.chunk_information.values():
        for msg_idx in msg_idx_list:
            if not msg_idx.timestamps:
                continue
            channel_id = msg_idx.channel_id
            if channel_id not in timestamps_by_channel:
                timestamps_by_channel[channel_id] = set()
            timestamps_by_channel[channel_id].update(msg_idx.timestamps)
    return timestamps_by_channel


def is_remote_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def path_from_input(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(value)


def path_basename(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "file"}:
        return Path(unquote(parsed.path)).name or value
    return Path(value).name or value


def discover_mcap_candidates(inputs: list[str]) -> list[str]:
    candidates: list[str] = []
    seen_local: set[Path] = set()
    seen_remote: set[str] = set()

    for raw_input in inputs:
        if is_remote_url(raw_input):
            if raw_input not in seen_remote:
                seen_remote.add(raw_input)
                candidates.append(raw_input)
            continue

        path = path_from_input(raw_input)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {raw_input}")

        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file() or child.suffix.lower() != ".mcap":
                    continue
                resolved = child.resolve()
                if resolved not in seen_local:
                    seen_local.add(resolved)
                    candidates.append(str(child))
            continue

        if not path.is_file():
            raise ValueError(f"Input path is neither file nor directory: {raw_input}")

        resolved = path.resolve()
        if resolved not in seen_local:
            seen_local.add(resolved)
            candidates.append(str(path))

    return candidates
