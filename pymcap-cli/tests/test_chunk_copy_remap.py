"""Tests for chunk fast-copy behavior under schema/channel id remapping.

When inputs are merged and id collisions force the remapper to reassign
an id, fast-copying any chunk from the affected stream is unsafe: the
chunk's compressed payload may embed Schema/Channel records under the
original (now stale) ids, since some MCAP writers eagerly include every
known channel in every chunk. The fix is to force DECODE for every
chunk from a stream that had any remap, not just chunks whose own
messages reference the remapped channel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.constants import DEFAULT_CHUNK_SIZE
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from small_mcap import CompressionType, McapWriter, get_summary

if TYPE_CHECKING:
    from pathlib import Path


def _write_mcap(
    path: Path,
    *,
    schemas: list[tuple[int, str, bytes]],
    channels: list[tuple[int, str, int]],
    messages: list[tuple[int, int, bytes]],
    chunk_size: int,
) -> None:
    """Write an MCAP fixture with the given schemas/channels/messages.

    schemas: list of (schema_id, name, data)
    channels: list of (channel_id, topic, schema_id)
    messages: list of (channel_id, log_time, data)
    """
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=chunk_size, compression=CompressionType.ZSTD)
        writer.start(profile="ros2", library="test")
        for schema_id, name, data in schemas:
            writer.add_schema(schema_id=schema_id, name=name, encoding="json", data=data)
        for channel_id, topic, schema_id in channels:
            writer.add_channel(
                channel_id=channel_id,
                topic=topic,
                message_encoding="json",
                schema_id=schema_id,
            )
        for channel_id, log_time, data in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=log_time,
            )
        writer.finish()


def _chunk_count(path: Path) -> int:
    with path.open("rb") as stream:
        summary = get_summary(stream)
    assert summary is not None
    assert summary.statistics is not None
    return summary.statistics.chunk_count


def _run_merge(inputs: list[Path], output: Path):
    open_streams = [p.open("rb") for p in inputs]
    try:
        out_stream = output.open("wb")
        try:
            options = ProcessingOptions(
                inputs=[
                    InputFile(
                        stream=s,
                        size=p.stat().st_size,
                        options=InputOptions.from_args(),
                    )
                    for s, p in zip(open_streams, inputs, strict=True)
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(compression="zstd", chunk_size=DEFAULT_CHUNK_SIZE),
            )
            return McapProcessor(options).process(out_stream)
        finally:
            out_stream.close()
    finally:
        for s in open_streams:
            s.close()


def test_single_file_passthrough_still_fast_copies(tmp_path: Path) -> None:
    """Single-file passthrough has no remap, so the new gate must not block fast-copy."""
    chunk_size = 4 * 1024
    payload = b'{"x":"' + b"a" * 256 + b'"}'

    src = tmp_path / "src.mcap"
    _write_mcap(
        src,
        schemas=[(1, "S", b'{"k":"v"}')],
        channels=[(1, "/t", 1)],
        messages=[(1, i, payload) for i in range(100)],
        chunk_size=chunk_size,
    )

    out = tmp_path / "out.mcap"
    stats = _run_merge([src], out)

    assert stats.errors_encountered == 0
    assert stats.chunks_processed > 0
    assert stats.chunks_copied == stats.chunks_processed
    assert stats.chunks_decoded == 0


def test_channel_id_collision_decodes_every_chunk_from_remapped_stream(
    tmp_path: Path,
) -> None:
    """Channel-id collision must decode all chunks from the colliding stream."""
    chunk_size = 4 * 1024
    payload = b'{"x":"' + b"a" * 256 + b'"}'

    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"

    # Both files use channel_id=1 but with different topics → collision on B.
    _write_mcap(
        a,
        schemas=[(1, "S", b'{"k":"v"}')],
        channels=[(1, "/a", 1)],
        messages=[(1, i, payload) for i in range(100)],
        chunk_size=chunk_size,
    )
    _write_mcap(
        b,
        schemas=[(1, "S", b'{"k":"v"}')],
        channels=[(1, "/b", 1)],
        messages=[(1, i, payload) for i in range(100)],
        chunk_size=chunk_size,
    )

    a_chunks = _chunk_count(a)
    b_chunks = _chunk_count(b)
    assert a_chunks > 1
    assert b_chunks > 1

    out = tmp_path / "merged.mcap"
    stats = _run_merge([a, b], out)

    assert stats.errors_encountered == 0
    # All B chunks were decoded; A's chunks fast-copy (no collision on A).
    assert stats.chunks_decoded >= b_chunks
    assert stats.chunks_copied >= a_chunks


def test_schema_id_collision_decodes_every_chunk_from_remapped_stream(
    tmp_path: Path,
) -> None:
    """Schema-id collision must decode all chunks from the colliding stream."""
    chunk_size = 4 * 1024
    payload = b'{"x":"' + b"a" * 256 + b'"}'

    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"

    # Same schema_id=1 but different content → schema gets remapped on B.
    _write_mcap(
        a,
        schemas=[(1, "TypeA", b'{"k":"a"}')],
        channels=[(1, "/a", 1)],
        messages=[(1, i, payload) for i in range(100)],
        chunk_size=chunk_size,
    )
    _write_mcap(
        b,
        schemas=[(1, "TypeB", b'{"k":"b"}')],
        channels=[(2, "/b", 1)],
        messages=[(2, i, payload) for i in range(100)],
        chunk_size=chunk_size,
    )

    b_chunks = _chunk_count(b)
    assert b_chunks > 1

    out = tmp_path / "merged.mcap"
    stats = _run_merge([a, b], out)

    assert stats.errors_encountered == 0
    assert stats.chunks_decoded >= b_chunks


def test_unrelated_chunks_in_remapped_stream_are_still_decoded(tmp_path: Path) -> None:
    """Regression: chunks whose messages don't reference the colliding channel.

    Before the per-stream gate, a chunk in the colliding stream that only
    carried messages on a non-colliding channel would fast-copy. If that
    chunk's payload embedded the colliding channel's (now stale) record,
    it would leak into the output. The fix is to force DECODE for every
    chunk from any stream that had any id remap, regardless of which
    channels each chunk's messages happen to reference.
    """
    chunk_size = 4 * 1024
    payload = b'{"x":"' + b"a" * 256 + b'"}'

    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"

    # A uses channel_id=1 for /a.
    _write_mcap(
        a,
        schemas=[(1, "S", b'{"k":"v"}')],
        channels=[(1, "/a", 1)],
        messages=[(1, i, payload) for i in range(40)],
        chunk_size=chunk_size,
    )

    # B has two channels:
    #   id=1 → /b (collides with A's id=1)
    #   id=2 → /c (no collision)
    # We write all /c messages first (filling several chunks) and add a
    # single /b message at the end. Most of B's chunks therefore have
    # only /c messages — the case the old per-channel gate missed.
    b_messages = [(2, i, payload) for i in range(100)]
    b_messages.append((1, 200, payload))
    _write_mcap(
        b,
        schemas=[(1, "S", b'{"k":"v"}')],
        channels=[(1, "/b", 1), (2, "/c", 1)],
        messages=b_messages,
        chunk_size=chunk_size,
    )

    b_chunks = _chunk_count(b)
    assert b_chunks > 1  # need multiple B chunks for the regression to mean anything

    out = tmp_path / "merged.mcap"
    stats = _run_merge([a, b], out)

    assert stats.errors_encountered == 0
    # Per-stream gate: every B chunk must be decoded, including the many
    # /c-only chunks that the old per-channel gate would have fast-copied.
    assert stats.chunks_decoded >= b_chunks
