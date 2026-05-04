"""Shared fast MCAP comparison primitives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, Protocol
from urllib.parse import unquote, urlparse

from small_mcap import (
    Channel,
    RebuildInfo,
    Schema,
    Summary,
    get_header,
    get_summary,
    read_info_approximate,
    rebuild_summary,
)

from pymcap_cli.core.input_handler import open_input


class HashSink(Protocol):
    def update(self, data: bytes, /) -> None: ...


ReadMode = Literal["summary", "index", "rebuild"]


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


def _update_str(hasher: HashSink, value: str) -> None:
    data = value.encode("utf-8")
    hasher.update(f"s:{len(data)}:".encode("ascii"))
    hasher.update(data)
    hasher.update(b";")


def _update_int(hasher: HashSink, value: int) -> None:
    hasher.update(f"i:{value};".encode("ascii"))


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


def _indexed_channel_digest(channel: ChannelFingerprint, timestamps: list[int]) -> str:
    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.indexed-channel.v1")
    _update_channel(hasher, channel)
    _update_int(hasher, len(timestamps))
    for timestamp in timestamps:
        _update_int(hasher, timestamp)
    return hasher.hexdigest()


def _unknown_indexed_channel_digest(channel_id: int, timestamps: list[int]) -> str:
    hasher = hashlib.sha256()
    _update_str(hasher, "pymcap-cli.compare.unknown-indexed-channel.v1")
    _update_int(hasher, channel_id)
    _update_int(hasher, len(timestamps))
    for timestamp in timestamps:
        _update_int(hasher, timestamp)
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
            semantic_digest = _unknown_indexed_channel_digest(channel_id, [])
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
        _update_channel(hasher, entry.fingerprint)
        _update_int(hasher, len(timestamps))
        for timestamp in timestamps:
            _update_int(hasher, timestamp)
        indexed_channel_digests.append(_indexed_channel_digest(entry.fingerprint, timestamps))
        indexed_channels.append(
            IndexedChannelIdentity(
                channel_semantic_digest=entry.semantic_digest,
                topic=entry.fingerprint.topic,
                message_count=len(timestamps),
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
        _update_int(hasher, channel_id)
        _update_int(hasher, len(timestamps))
        for timestamp in timestamps:
            _update_int(hasher, timestamp)
        indexed_channel_digests.append(_unknown_indexed_channel_digest(channel_id, timestamps))
        indexed_channels.append(
            IndexedChannelIdentity(
                channel_semantic_digest=_unknown_indexed_channel_digest(channel_id, []),
                topic=_unknown_channel_topic(channel_id),
                message_count=len(timestamps),
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
