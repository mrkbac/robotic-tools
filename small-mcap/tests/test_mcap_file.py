from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
import small_mcap
import small_mcap.reader as reader_module
from small_mcap import (
    Channel,
    McapWriter,
    Message,
    Schema,
    get_summary,
    include_topics,
    read_message,
)
from small_mcap.writer import CompressionType, IndexType

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def _write_recording(
    path: Path,
    *,
    use_chunking: bool = True,
    index_types: IndexType = IndexType.ALL,
    message_count: int = 24,
) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(
            stream,
            use_chunking=use_chunking,
            index_types=index_types,
            compression=CompressionType.NONE,
            chunk_size=180,
            enable_crcs=True,
        )
        writer.start()
        writer.add_schema(1, "example/Raw", "text", b"bytes data")
        writer.add_channel(1, "/camera", "raw", 1)
        writer.add_channel(2, "/lidar", "raw", 1)
        for sequence in range(message_count):
            channel_id = sequence % 2 + 1
            timestamp_ns = 1_000 + sequence * 10
            writer.add_message(
                channel_id,
                timestamp_ns,
                bytes([sequence]) * 48,
                publish_time=timestamp_ns,
                sequence=sequence,
            )
        writer.finish()


def _records(
    messages: Iterable[tuple[Schema | None, Channel, Message]],
) -> list[tuple[str, int, int, bytes]]:
    return [
        (channel.topic, message.log_time, message.sequence, bytes(message.data))
        for _schema, channel, message in messages
    ]


@pytest.mark.parametrize(
    ("topics", "start_time_ns", "end_time_ns", "reverse", "validate_crc"),
    [
        (None, 0, 2**63 - 1, False, False),
        (("/camera",), 0, 2**63 - 1, False, False),
        (("/camera", "/lidar"), 1_050, 1_170, False, False),
        (("/lidar",), 0, 1_151, True, False),
        (None, 1_020, 1_180, True, True),
    ],
)
def test_mcap_file_read_message_matches_free_function(
    tmp_path: Path,
    topics: tuple[str, ...] | None,
    start_time_ns: int,
    end_time_ns: int,
    reverse: bool,
    validate_crc: bool,
) -> None:
    path = tmp_path / "recording.mcap"
    _write_recording(path)
    should_include = (
        small_mcap.reader._should_include_all if topics is None else include_topics(topics)
    )
    with path.open("rb") as stream:
        expected = _records(
            read_message(
                stream,
                should_include=should_include,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                reverse=reverse,
                validate_crc=validate_crc,
            )
        )

    with small_mcap.McapFile.open(path) as recording:
        actual = _records(
            recording.read_message(
                should_include=should_include,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                reverse=reverse,
                validate_crc=validate_crc,
            )
        )

    assert actual == expected


def test_mcap_file_matches_free_function_for_overlapping_chunks(tmp_path: Path) -> None:
    path = tmp_path / "overlapping.mcap"
    with path.open("wb") as stream:
        writer = McapWriter(
            stream,
            compression=CompressionType.NONE,
            chunk_size=180,
        )
        writer.start()
        writer.add_schema(1, "example/Raw", "text", b"bytes data")
        writer.add_channel(1, "/camera", "raw", 1)
        for sequence, timestamp_ns in enumerate((1_000, 1_300, 1_100, 1_200, 1_400)):
            writer.add_message(1, timestamp_ns, bytes([sequence]) * 48, timestamp_ns, sequence)
        writer.finish()

    with path.open("rb") as stream:
        summary = get_summary(stream)
        expected = _records(read_message(stream, reverse=True))
    assert summary is not None
    assert any(
        left.message_end_time > right.message_start_time
        for left, right in pairwise(summary.chunk_indexes)
    )

    with small_mcap.McapFile.open(path) as recording:
        actual = _records(recording.read_message(reverse=True))

    assert actual == expected


@pytest.mark.parametrize(
    ("use_chunking", "index_types", "num_workers"),
    [
        (False, IndexType.ALL, 0),
        (True, IndexType.NONE, 0),
        (True, IndexType.ALL, 2),
    ],
)
def test_mcap_file_fallback_matches_free_function(
    tmp_path: Path,
    use_chunking: bool,
    index_types: IndexType,
    num_workers: int,
) -> None:
    path = tmp_path / "fallback.mcap"
    _write_recording(path, use_chunking=use_chunking, index_types=index_types)
    with path.open("rb") as stream:
        expected = _records(
            read_message(
                stream,
                should_include=include_topics("/camera"),
                start_time_ns=1_030,
                end_time_ns=1_190,
                num_workers=num_workers,
            )
        )
    with small_mcap.McapFile.open(path) as recording:
        actual = _records(
            recording.read_message(
                should_include=include_topics("/camera"),
                start_time_ns=1_030,
                end_time_ns=1_190,
                num_workers=num_workers,
            )
        )
    assert actual == expected


def test_mcap_file_iterators_are_independent(tmp_path: Path) -> None:
    path = tmp_path / "independent.mcap"
    _write_recording(path)

    with small_mcap.McapFile.open(path) as recording:
        forward = iter(recording.read_message(should_include=include_topics("/camera")))
        reverse = iter(
            recording.read_message(
                should_include=include_topics("/lidar"),
                reverse=True,
            )
        )
        assert next(forward)[2].sequence == 0
        assert next(reverse)[2].sequence == 23
        assert next(forward)[2].sequence == 2
        assert next(reverse)[2].sequence == 21


def test_mcap_file_reuses_decompressed_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cache.mcap"
    _write_recording(path)
    calls = 0
    original = reader_module._decompress_data_threadsafe

    def count_decompression(chunk: small_mcap.Chunk) -> bytes | memoryview:
        nonlocal calls
        calls += 1
        return original(chunk)

    monkeypatch.setattr(reader_module, "_decompress_data_threadsafe", count_decompression)
    with small_mcap.McapFile.open(path) as recording:
        first = next(iter(recording.read_message(start_time_ns=1_000, end_time_ns=1_001)))
        assert first[2].sequence == 0
        first_calls = calls
        latest = next(iter(recording.read_message(end_time_ns=1_001, reverse=True)))
        assert latest[2].sequence == 0
        assert calls == first_calls


def test_mcap_file_cache_separates_crc_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "crc.mcap"
    _write_recording(path)
    calls = 0
    original = reader_module._decompress_data_threadsafe

    def count_decompression(chunk: small_mcap.Chunk) -> bytes | memoryview:
        nonlocal calls
        calls += 1
        return original(chunk)

    monkeypatch.setattr(reader_module, "_decompress_data_threadsafe", count_decompression)
    with small_mcap.McapFile.open(path) as recording:
        next(iter(recording.read_message(end_time_ns=1_001)))
        next(iter(recording.read_message(end_time_ns=1_001, validate_crc=True)))
        validated_calls = calls
        next(iter(recording.read_message(end_time_ns=1_001, validate_crc=True)))
        assert calls == validated_calls
        assert validated_calls == 2


def test_mcap_file_oversized_chunk_is_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized.mcap"
    _write_recording(path)
    calls = 0
    original = reader_module._decompress_data_threadsafe

    def count_decompression(chunk: small_mcap.Chunk) -> bytes | memoryview:
        nonlocal calls
        calls += 1
        return original(chunk)

    monkeypatch.setattr(reader_module, "_decompress_data_threadsafe", count_decompression)
    with small_mcap.McapFile.open(path, chunk_cache_bytes=1) as recording:
        next(iter(recording.read_message(end_time_ns=1_001)))
        next(iter(recording.read_message(end_time_ns=1_001)))
        assert recording._cached_chunk_bytes <= 1
    assert calls == 2


def test_mcap_file_lru_evicts_least_recent_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "lru.mcap"
    _write_recording(path)
    with path.open("rb") as stream:
        summary = get_summary(stream)
    assert summary is not None
    assert len(summary.chunk_indexes) >= 3
    cache_bytes = max(index.uncompressed_size for index in summary.chunk_indexes)
    calls = 0
    original = reader_module._decompress_data_threadsafe

    def count_decompression(chunk: small_mcap.Chunk) -> bytes | memoryview:
        nonlocal calls
        calls += 1
        return original(chunk)

    monkeypatch.setattr(reader_module, "_decompress_data_threadsafe", count_decompression)
    first_time = summary.chunk_indexes[0].message_start_time
    second_time = summary.chunk_indexes[1].message_start_time
    with small_mcap.McapFile.open(path, chunk_cache_bytes=cache_bytes) as recording:
        next(iter(recording.read_message(start_time_ns=first_time, end_time_ns=first_time + 1)))
        next(iter(recording.read_message(start_time_ns=second_time, end_time_ns=second_time + 1)))
        next(iter(recording.read_message(start_time_ns=first_time, end_time_ns=first_time + 1)))
        assert recording._cached_chunk_bytes <= cache_bytes
    assert calls == 3


def test_mcap_file_close_is_idempotent_and_invalidates_iterators(tmp_path: Path) -> None:
    path = tmp_path / "closed.mcap"
    _write_recording(path)
    recording = small_mcap.McapFile.open(path)
    iterator = iter(recording.read_message())

    recording.close()
    recording.close()

    with pytest.raises(RuntimeError, match="closed"):
        next(iterator)
    with pytest.raises(RuntimeError, match="closed"):
        recording.read_message()
