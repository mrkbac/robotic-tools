from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from small_mcap import (
    MAGIC_SIZE,
    CompressionType,
    CRCValidationError,
    McapFileReplacedError,
    McapFileTruncatedError,
    McapFollower,
    McapWriter,
    Opcode,
    try_read_record,
)
from small_mcap.records import OPCODE_AND_LEN_STRUCT

if TYPE_CHECKING:
    from pathlib import Path


def _recording_bytes(
    *,
    message_count: int = 4,
    chunk_size: int = 64,
    compression: CompressionType = CompressionType.NONE,
) -> bytes:
    stream = io.BytesIO()
    writer = McapWriter(
        stream,
        chunk_size=chunk_size,
        compression=compression,
    )
    writer.start()
    writer.add_schema(1, "example/msg/Sample", "jsonschema", b"{}")
    writer.add_channel(1, "/sample", "json", 1)
    for sequence in range(message_count):
        writer.add_message(1, sequence, f'{{"value": {sequence}}}'.encode(), sequence, sequence)
    writer.finish()
    return stream.getvalue()


def _first_record_end(data: bytes, expected_opcode: Opcode) -> int:
    offset = MAGIC_SIZE
    while offset + OPCODE_AND_LEN_STRUCT.size <= len(data):
        opcode, length = OPCODE_AND_LEN_STRUCT.unpack_from(data, offset)
        end = offset + OPCODE_AND_LEN_STRUCT.size + length
        if opcode == expected_opcode:
            return end
        offset = end
    raise AssertionError(f"opcode {expected_opcode.name} not found")


@pytest.mark.parametrize("compression", [CompressionType.NONE, CompressionType.ZSTD])
def test_follower_appended_one_byte_at_a_time_emits_exactly_once(
    tmp_path: Path,
    compression: CompressionType,
) -> None:
    data = _recording_bytes(message_count=6, compression=compression)
    path = tmp_path / "growing.mcap"
    path.touch()
    messages = []
    committed_offsets: list[int] = []

    with McapFollower.open(path, validate_crc=True) as follower:
        for byte in data:
            with path.open("ab") as stream:
                stream.write(bytes([byte]))
            batch = follower.poll_messages(max_messages=2, max_bytes=32)
            messages.extend(batch.messages)
            committed_offsets.append(batch.committed_offset)
        while not batch.is_final:
            batch = follower.poll_messages(max_messages=2, max_bytes=32)
            messages.extend(batch.messages)

    assert [message.sequence for _schema, _channel, message in messages] == list(range(6))
    assert committed_offsets == sorted(committed_offsets)
    assert batch.committed_offset == len(data)
    assert batch.is_final


@pytest.mark.parametrize("missing_bytes", [1, 8, 9, 10])
def test_try_read_record_restores_offset_for_partial_record(missing_bytes: int) -> None:
    data = _recording_bytes(message_count=0)
    header_end = _first_record_end(data, Opcode.HEADER)
    encoded_header = data[MAGIC_SIZE:header_end]
    stream = io.BytesIO(encoded_header[:-missing_bytes])

    assert try_read_record(stream) is None
    assert stream.tell() == 0


def test_follower_repeated_poll_without_growth_does_not_move_cursor(tmp_path: Path) -> None:
    data = _recording_bytes()
    header_end = _first_record_end(data, Opcode.HEADER)
    path = tmp_path / "growing.mcap"
    path.write_bytes(data[:header_end])

    with McapFollower.open(path) as follower:
        first = follower.poll_messages()
        second = follower.poll_messages()

    assert first.messages == ()
    assert second.messages == ()
    assert second.committed_offset == first.committed_offset == header_end
    assert not second.is_final


def test_follower_complete_chunk_precedes_partial_message_index(tmp_path: Path) -> None:
    data = _recording_bytes(message_count=3, chunk_size=1024)
    chunk_end = _first_record_end(data, Opcode.CHUNK)
    path = tmp_path / "growing.mcap"
    path.write_bytes(data[: chunk_end + 4])

    with McapFollower.open(path) as follower:
        first = follower.poll_messages()
        second = follower.poll_messages()

    assert [message.sequence for _schema, _channel, message in first.messages] == [0, 1, 2]
    assert second.messages == ()
    assert second.committed_offset == first.committed_offset == chunk_end


def test_follower_crc_failure_waits_for_complete_chunk(tmp_path: Path) -> None:
    data = bytearray(_recording_bytes(message_count=1, chunk_size=1024))
    chunk_end = _first_record_end(data, Opcode.CHUNK)
    data[chunk_end - 1] ^= 0xFF
    path = tmp_path / "growing.mcap"
    path.write_bytes(data[: chunk_end - 1])

    with McapFollower.open(path, validate_crc=True) as follower:
        follower.poll_messages()
        with path.open("ab") as stream:
            stream.write(data[chunk_end - 1 : chunk_end])
        with pytest.raises(CRCValidationError):
            follower.poll_messages()


def test_follower_respects_message_budget_without_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "complete.mcap"
    path.write_bytes(_recording_bytes(message_count=5, chunk_size=1024))

    with McapFollower.open(path) as follower:
        batches = [follower.poll_messages(max_messages=2, max_bytes=1024) for _ in range(3)]

    assert [len(batch.messages) for batch in batches] == [2, 2, 1]
    assert [
        message.sequence for batch in batches for _schema, _channel, message in batch.messages
    ] == list(range(5))


def test_follower_raises_typed_truncation_error(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    path.write_bytes(_recording_bytes())

    with McapFollower.open(path) as follower:
        batch = follower.poll_messages()
        path.write_bytes(path.read_bytes()[: batch.committed_offset - 1])
        with pytest.raises(McapFileTruncatedError):
            follower.poll_messages()


def test_follower_raises_typed_replacement_error(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    data = _recording_bytes()
    path.write_bytes(data)
    replacement = tmp_path / "replacement.mcap"
    replacement.write_bytes(data)

    with McapFollower.open(path) as follower:
        follower.poll_messages(max_messages=1)
        replacement.replace(path)
        with pytest.raises(McapFileReplacedError):
            follower.poll_messages()
