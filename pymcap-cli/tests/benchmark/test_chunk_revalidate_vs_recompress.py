"""Micro-benchmark: validation-pass cost vs full re-emit/recompress cost.

When the smart chunk-copy gate falls back to DECODE for a stream that had
an id remap, the worker currently decompresses + parses records, then the
main thread re-emits them through the writer's chunk_builder (which
serializes each record and recompresses the result).

An alternative would be: after decoding, walk the records once to verify
that every embedded Schema/Channel matches the writer's view; if it does,
fast-copy the original compressed bytes and skip the re-emit entirely.
This benchmark measures the extra cost (validation pass) against the
work it would let us skip (re-emit + recompression).

The chunks here are produced via small-mcap with realistic payload sizes
and chunk targets, then decoded once outside the benchmark loop so we're
timing just the two operations under comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from small_mcap import (
    Channel,
    CompressionType,
    McapWriter,
    Message,
    Schema,
    breakup_chunk,
    stream_reader,
)
from small_mcap.records import Chunk
from small_mcap.writer import _ChunkBuilder

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture: a realistic compressed chunk + its decoded records
# ---------------------------------------------------------------------------


def _build_chunk_fixture(
    tmp_path: Path,
    *,
    compression: CompressionType,
    chunk_size: int,
    num_messages: int,
    payload_size: int,
    num_channels: int,
) -> tuple[Chunk, list]:
    """Write an MCAP with one large chunk, return that chunk + decoded records."""
    path = tmp_path / f"bench_{compression.value}_{payload_size}_{num_messages}.mcap"
    payload = b"x" * payload_size

    with path.open("wb") as f:
        writer = McapWriter(f, chunk_size=chunk_size, compression=compression)
        writer.start(profile="ros2", library="bench")
        writer.add_schema(schema_id=1, name="S", encoding="json", data=b'{"k":"v"}')
        for channel_id in range(1, num_channels + 1):
            writer.add_channel(
                channel_id=channel_id,
                topic=f"/t{channel_id}",
                message_encoding="json",
                schema_id=1,
            )
        for i in range(num_messages):
            writer.add_message(
                channel_id=(i % num_channels) + 1,
                log_time=i,
                data=payload,
                publish_time=i,
            )
        writer.finish()

    with path.open("rb") as stream:
        chunk = next(
            r
            for r in stream_reader(stream, emit_chunks=True, lazy_chunks=False)
            if isinstance(r, Chunk)
        )

    records = list(breakup_chunk(chunk, validate_crc=False))
    return chunk, records


@pytest.fixture(scope="module")
def chunk_fixture_zstd(tmp_path_factory):
    """One realistically-sized zstd chunk + its decoded records.

    Targeted ~1MB compressed, ~5MB uncompressed: 5000 messages of 1KB payload,
    fed into a single chunk by setting chunk_size large enough to absorb them.
    """
    tmp_path = tmp_path_factory.mktemp("chunk_bench_zstd")
    return _build_chunk_fixture(
        tmp_path,
        compression=CompressionType.ZSTD,
        chunk_size=64 * 1024 * 1024,
        num_messages=5000,
        payload_size=1024,
        num_channels=8,
    )


@pytest.fixture(scope="module")
def chunk_fixture_lz4(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("chunk_bench_lz4")
    return _build_chunk_fixture(
        tmp_path,
        compression=CompressionType.LZ4,
        chunk_size=64 * 1024 * 1024,
        num_messages=5000,
        payload_size=1024,
        num_channels=8,
    )


@pytest.fixture(scope="module")
def chunk_fixture_eager(chunk_fixture_zstd):
    """Simulate an eager-replay writer's chunk: many in-chunk Schema/Channel records.

    small-mcap writes schemas/channels at file level only, so its real chunks
    contain only Message records. Other writers (and some streaming tools)
    repeat the full schema/channel list inside every chunk. This fixture
    prepends a synthetic prefix of 50 schemas + 50 channels to a real
    decoded record list so the validators see the worst case they'd face
    in practice.
    """
    chunk, base_records = chunk_fixture_zstd
    schema_prefix = [
        Schema(id=sid, name=f"Type{sid}", encoding="json", data=b'{"k":"v"}')
        for sid in range(1, 51)
    ]
    channel_prefix = [
        Channel(
            id=cid,
            schema_id=((cid - 1) % 50) + 1,
            topic=f"/eager/{cid}",
            message_encoding="json",
            metadata={},
        )
        for cid in range(1, 51)
    ]
    return chunk, schema_prefix + channel_prefix + base_records


# ---------------------------------------------------------------------------
# Validation pass: what the optimization adds
# ---------------------------------------------------------------------------


def _validate_chunk_records(
    records: list,
    schemas: dict[int, Schema],
    channels: dict[int, Channel],
) -> bool:
    """Naive validator: isinstance dispatch, scans every record."""
    for record in records:
        if isinstance(record, Schema):
            existing = schemas.get(record.id)
            if existing is None or existing != record:
                return False
        elif isinstance(record, Channel):
            existing = channels.get(record.id)
            if existing is None or existing != record:
                return False
    return True


def _validate_chunk_records_typed(
    records: list,
    schemas: dict[int, Schema],
    channels: dict[int, Channel],
) -> bool:
    """Type-is dispatch, full scan: safer (handles interleaved records)."""
    for record in records:
        rtype = type(record)
        if rtype is Schema:
            existing = schemas.get(record.id)
            if existing is None or existing != record:
                return False
        elif rtype is Channel:
            existing = channels.get(record.id)
            if existing is None or existing != record:
                return False
    return True


def _validate_chunk_records_smart(
    records: list,
    schemas: dict[int, Schema],
    channels: dict[int, Channel],
) -> bool:
    """Smart validator: `type is` dispatch, stops at the first Message record.

    MCAP chunks place Schema and Channel records as a small prefix before any
    Messages that reference them. Once we see a Message we know the
    schema/channel section is over, so further iteration is wasted work.
    """
    for record in records:
        rtype = type(record)
        if rtype is Schema:
            existing = schemas.get(record.id)
            if existing is None or existing != record:
                return False
        elif rtype is Channel:
            existing = channels.get(record.id)
            if existing is None or existing != record:
                return False
        else:
            return True
    return True


# ---------------------------------------------------------------------------
# Re-emit pass: what the optimization would let us skip in the safe case
# ---------------------------------------------------------------------------


def _reemit_chunk_records(records: list, compression: CompressionType) -> bytes:
    """Push every Message through a fresh chunk_builder and compress the result.

    Mirrors the work the writer does today on the DECODE path: each message is
    re-serialized into the chunk buffer, and the buffer is compressed when the
    chunk is finalized.
    """
    builder = _ChunkBuilder(compression=compression, enable_crcs=False, chunk_size=1 << 30)
    for record in records:
        if isinstance(record, Message):
            builder.add_raw(
                channel_id=record.channel_id,
                log_time=record.log_time,
                data=record.data,
                publish_time=record.publish_time,
                sequence=record.sequence,
            )
    finalized = builder.finalize()
    assert finalized is not None
    chunk, _indices = finalized
    return chunk.data


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="chunk-revalidate-vs-recompress-zstd")
def test_benchmark_validate_naive_zstd(benchmark, chunk_fixture_zstd):
    """Naive validation: isinstance dispatch, full record sweep."""
    _chunk, records = chunk_fixture_zstd
    schemas = {r.id: r for r in records if isinstance(r, Schema)}
    channels = {r.id: r for r in records if isinstance(r, Channel)}

    result = benchmark(_validate_chunk_records, records, schemas, channels)
    assert result is True


@pytest.mark.benchmark(group="chunk-revalidate-vs-recompress-zstd")
def test_benchmark_validate_smart_zstd(benchmark, chunk_fixture_zstd):
    """Smart validation: `type is` dispatch + early-exit on first Message."""
    _chunk, records = chunk_fixture_zstd
    schemas = {r.id: r for r in records if isinstance(r, Schema)}
    channels = {r.id: r for r in records if isinstance(r, Channel)}

    result = benchmark(_validate_chunk_records_smart, records, schemas, channels)
    assert result is True


@pytest.mark.benchmark(group="chunk-revalidate-vs-recompress-zstd")
def test_benchmark_reemit_zstd(benchmark, chunk_fixture_zstd):
    """Cost of re-emitting + zstd-compressing the same records."""
    _chunk, records = chunk_fixture_zstd
    blob = benchmark(_reemit_chunk_records, records, CompressionType.ZSTD)
    assert len(blob) > 0


@pytest.mark.benchmark(group="chunk-revalidate-vs-recompress-lz4")
def test_benchmark_validate_naive_lz4(benchmark, chunk_fixture_lz4):
    _chunk, records = chunk_fixture_lz4
    schemas = {r.id: r for r in records if isinstance(r, Schema)}
    channels = {r.id: r for r in records if isinstance(r, Channel)}

    result = benchmark(_validate_chunk_records, records, schemas, channels)
    assert result is True


@pytest.mark.benchmark(group="chunk-revalidate-vs-recompress-lz4")
def test_benchmark_validate_smart_lz4(benchmark, chunk_fixture_lz4):
    _chunk, records = chunk_fixture_lz4
    schemas = {r.id: r for r in records if isinstance(r, Schema)}
    channels = {r.id: r for r in records if isinstance(r, Channel)}

    result = benchmark(_validate_chunk_records_smart, records, schemas, channels)
    assert result is True


@pytest.mark.benchmark(group="chunk-revalidate-vs-recompress-lz4")
def test_benchmark_reemit_lz4(benchmark, chunk_fixture_lz4):
    _chunk, records = chunk_fixture_lz4
    blob = benchmark(_reemit_chunk_records, records, CompressionType.LZ4)
    assert len(blob) > 0


@pytest.mark.benchmark(group="chunk-revalidate-eager-replay")
def test_benchmark_validate_naive_eager(benchmark, chunk_fixture_eager):
    """Naive validator on a chunk that embeds 50 schemas + 50 channels."""
    _chunk, records = chunk_fixture_eager
    schemas = {r.id: r for r in records if isinstance(r, Schema)}
    channels = {r.id: r for r in records if isinstance(r, Channel)}

    result = benchmark(_validate_chunk_records, records, schemas, channels)
    assert result is True


@pytest.mark.benchmark(group="chunk-revalidate-eager-replay")
def test_benchmark_validate_typed_eager(benchmark, chunk_fixture_eager):
    """Type-is dispatch, full scan: same correctness as naive, no early-exit."""
    _chunk, records = chunk_fixture_eager
    schemas = {r.id: r for r in records if isinstance(r, Schema)}
    channels = {r.id: r for r in records if isinstance(r, Channel)}

    result = benchmark(_validate_chunk_records_typed, records, schemas, channels)
    assert result is True


@pytest.mark.benchmark(group="chunk-revalidate-eager-replay")
def test_benchmark_validate_smart_eager(benchmark, chunk_fixture_eager):
    """Smart validator on the same eager-replay chunk."""
    _chunk, records = chunk_fixture_eager
    schemas = {r.id: r for r in records if isinstance(r, Schema)}
    channels = {r.id: r for r in records if isinstance(r, Channel)}

    result = benchmark(_validate_chunk_records_smart, records, schemas, channels)
    assert result is True


# ---------------------------------------------------------------------------
# Sanity: make sure the chunk fixture actually exercises a real-sized chunk
# ---------------------------------------------------------------------------


def test_chunk_fixture_is_realistically_sized(chunk_fixture_zstd):
    chunk, records = chunk_fixture_zstd
    assert chunk.uncompressed_size >= 4 * 1024 * 1024
    assert len(chunk.data) < chunk.uncompressed_size
    msg_count = sum(1 for r in records if isinstance(r, Message))
    assert msg_count >= 4000
