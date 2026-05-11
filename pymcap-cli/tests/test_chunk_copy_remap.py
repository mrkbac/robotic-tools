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
    _chunk_records_match_writer_view,
)
from small_mcap import Channel, CompressionType, McapWriter, Message, Schema, get_summary

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
    """Channel-id collision: every B chunk references the remapped channel,
    so the per-channel gate forces DECODE for all of them."""
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
    # All B chunks were decoded (per-channel was_channel_remapped on id=1).
    # A's chunks fast-copy (no collision on A).
    assert stats.chunks_decoded >= b_chunks
    assert stats.chunks_copied >= a_chunks


def test_schema_id_collision_verifies_then_fast_copies_remapped_stream(
    tmp_path: Path,
) -> None:
    """Schema-id collision without a channel-id collision routes through
    DECODE_VERIFY: small-mcap chunks have no in-chunk metadata so verify
    passes for every B chunk, and they fast-copy."""
    chunk_size = 4 * 1024
    payload = b'{"x":"' + b"a" * 256 + b'"}'

    a = tmp_path / "a.mcap"
    b = tmp_path / "b.mcap"

    # Same schema_id=1 but different content → schema gets remapped on B.
    # Channel ids don't collide (A uses 1, B uses 2) so no per-channel
    # remap fires for B's chunks.
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
    # Every B chunk went through verify and fast-copied (no in-chunk
    # records to disagree with the writer's remapped view).
    assert stats.chunks_verified >= b_chunks
    assert stats.chunks_decoded == 0


def test_unrelated_chunks_in_remapped_stream_are_verified_and_fast_copied(
    tmp_path: Path,
) -> None:
    """Stream-level remap on /b doesn't propagate to chunks carrying only /c.

    Before the per-stream gate, /c-only chunks fast-copied without any
    safety check. With the gate alone they were forced through full
    DECODE. With the verify pass they're decoded just enough to confirm
    no in-chunk records disagree with the writer's view, then fast-copied
    via ``add_chunk`` (no recompression).
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
    #   id=1 → /b (collides with A's id=1; will be remapped)
    #   id=10 → /c (no collision; preserved as 10 because 10 is unused)
    # Most messages are on /c so most chunks contain only /c messages and
    # avoid the per-channel remap gate — those are the ones that exercise
    # DECODE_VERIFY.
    b_messages = [(10, i, payload) for i in range(100)]
    b_messages.append((1, 200, payload))
    _write_mcap(
        b,
        schemas=[(1, "S", b'{"k":"v"}')],
        channels=[(1, "/b", 1), (10, "/c", 1)],
        messages=b_messages,
        chunk_size=chunk_size,
    )

    b_chunks = _chunk_count(b)
    assert b_chunks > 1

    out = tmp_path / "merged.mcap"
    stats = _run_merge([a, b], out)

    assert stats.errors_encountered == 0
    # The /c-only chunks went through verify + fast-copy. The single chunk
    # that contains the /b message also references the remapped channel,
    # so it goes through full DECODE. Together they cover every B chunk.
    assert stats.chunks_verified >= b_chunks - 1
    assert stats.chunks_decoded >= 1


# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------


def _msg(channel_id: int = 1) -> Message:
    return Message(channel_id=channel_id, sequence=0, log_time=0, publish_time=0, data=b"")


def _schema(sid: int, name: str = "S", data: bytes = b"{}") -> Schema:
    return Schema(id=sid, name=name, encoding="json", data=data)


def _channel(cid: int, schema_id: int = 1, topic: str = "/t") -> Channel:
    return Channel(
        id=cid, schema_id=schema_id, topic=topic, message_encoding="json", metadata={}
    )


def test_validator_empty_records_returns_true() -> None:
    assert _chunk_records_match_writer_view([], {}, {}) is True


def test_validator_messages_only_returns_true() -> None:
    """Selective-replay chunks (no in-chunk metadata) early-exit on message #1."""
    records = [_msg(), _msg(), _msg()]
    assert _chunk_records_match_writer_view(records, {}, {}) is True


def test_validator_matching_metadata_returns_true() -> None:
    schema = _schema(1)
    channel = _channel(1, 1, "/t")
    records = [schema, channel, _msg(1)]
    assert (
        _chunk_records_match_writer_view(records, {1: schema}, {1: channel}) is True
    )


def test_validator_unknown_schema_returns_false() -> None:
    schema = _schema(1)
    records = [schema]
    assert _chunk_records_match_writer_view(records, {}, {}) is False


def test_validator_mismatched_schema_returns_false() -> None:
    in_chunk = _schema(1, name="OldName")
    in_writer = _schema(1, name="NewName")
    assert (
        _chunk_records_match_writer_view([in_chunk], {1: in_writer}, {}) is False
    )


def test_validator_mismatched_channel_returns_false() -> None:
    in_chunk = _channel(1, 1, "/old")
    in_writer = _channel(1, 1, "/new")
    assert (
        _chunk_records_match_writer_view([in_chunk], {}, {1: in_writer}) is False
    )


def test_validator_early_exits_on_first_message() -> None:
    """A Schema record placed AFTER a Message is not inspected — by spec
    convention metadata always appears as a small prefix. This documents
    that assumption; if a writer ever interleaves we'd need a full scan."""
    bad_schema = _schema(99, name="WouldNotMatch")
    records = [_msg(), bad_schema]
    # bad_schema is never validated because we early-exit on the first Message.
    assert _chunk_records_match_writer_view(records, {}, {}) is True
