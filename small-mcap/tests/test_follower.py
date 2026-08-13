from __future__ import annotations

import io
import threading
import time
from typing import TYPE_CHECKING

import pytest
import small_mcap.follower as follower_module
from small_mcap import (
    MAGIC,
    MAGIC_SIZE,
    Attachment,
    Channel,
    ChannelNotFoundError,
    CompressionType,
    CRCValidationError,
    DataEnd,
    Footer,
    Header,
    InvalidHeaderError,
    InvalidMagicError,
    McapFileReplacedError,
    McapFileTruncatedError,
    McapFollower,
    McapRecord,
    McapWriter,
    Message,
    Opcode,
    RecordLengthLimitExceededError,
    Schema,
    SchemaNotFoundError,
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


def _file_bytes(*records: McapRecord) -> bytes:
    stream = io.BytesIO()
    stream.write(MAGIC)
    for record in records:
        record.write_record_to(stream)
    stream.write(MAGIC)
    return stream.getvalue()


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


def test_try_read_record_rejects_record_over_size_limit() -> None:
    stream = io.BytesIO(OPCODE_AND_LEN_STRUCT.pack(Opcode.MESSAGE, 11))

    with pytest.raises(RecordLengthLimitExceededError):
        try_read_record(stream, record_size_limit=10)

    assert stream.tell() == 0


def test_follower_rejects_non_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "not-regular.mcap"
    path.touch()
    monkeypatch.setattr(follower_module.stat, "S_ISREG", lambda _mode: False)

    with pytest.raises(ValueError, match="local regular file"):
        McapFollower.open(path)


def test_follower_rejects_closed_and_invalid_poll_budgets(tmp_path: Path) -> None:
    path = tmp_path / "empty.mcap"
    path.touch()
    follower = McapFollower.open(path)

    with pytest.raises(ValueError, match="max_messages"):
        follower.poll_messages(max_messages=0)
    with pytest.raises(ValueError, match="max_bytes"):
        follower.poll_messages(max_bytes=0)

    follower.close()
    follower.close()
    with pytest.raises(ValueError, match="closed"):
        follower.poll_messages()


def test_follower_iterates_complete_file_to_final(tmp_path: Path) -> None:
    path = tmp_path / "complete.mcap"
    path.write_bytes(_recording_bytes(message_count=3))

    with McapFollower.open(path) as follower:
        messages = list(follower.iter_messages(poll_interval=0.001))

    assert [message.sequence for _schema, _channel, message in messages] == [0, 1, 2]


def test_follower_iteration_validates_intervals_and_stops_when_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "empty.mcap"
    path.touch()
    ticks = iter((0.0, 0.0, 1.0))
    sleeps: list[float] = []
    monkeypatch.setattr(follower_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(follower_module.time, "sleep", sleeps.append)

    with McapFollower.open(path) as follower:
        with pytest.raises(ValueError, match="poll_interval"):
            list(follower.iter_messages(poll_interval=0))
        with pytest.raises(ValueError, match="idle_timeout"):
            list(follower.iter_messages(idle_timeout=-1))
        assert list(follower.iter_messages(poll_interval=0.25, idle_timeout=0.5)) == []

    assert sleeps == [0.25]


def test_follower_rejects_missing_followed_path(tmp_path: Path) -> None:
    path = tmp_path / "removed.mcap"
    path.touch()

    with McapFollower.open(path) as follower:
        path.unlink()
        with pytest.raises(McapFileReplacedError):
            follower.poll_messages()


def test_follower_rejects_invalid_leading_and_trailing_magic(tmp_path: Path) -> None:
    leading = tmp_path / "bad-leading.mcap"
    leading.write_bytes(b"x" * MAGIC_SIZE)
    with McapFollower.open(leading) as follower, pytest.raises(InvalidMagicError):
        follower.poll_messages()

    trailing = tmp_path / "bad-trailing.mcap"
    data = bytearray(_file_bytes(Header("", "test"), DataEnd(0), Footer(0, 0, 0)))
    data[-MAGIC_SIZE:] = b"x" * MAGIC_SIZE
    trailing.write_bytes(data)
    with McapFollower.open(trailing) as follower, pytest.raises(InvalidMagicError):
        follower.poll_messages()


def test_follower_rejects_non_header_first_record(tmp_path: Path) -> None:
    path = tmp_path / "no-header.mcap"
    path.write_bytes(_file_bytes(Schema(1, "example", "jsonschema", b"{}")))

    with McapFollower.open(path) as follower, pytest.raises(InvalidHeaderError):
        follower.poll_messages()


def test_follower_rejects_data_section_crc_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad-data-crc.mcap"
    path.write_bytes(_file_bytes(Header("", "test"), DataEnd(1), Footer(0, 0, 0)))

    with McapFollower.open(path, validate_crc=True) as follower, pytest.raises(CRCValidationError):
        follower.poll_messages()


def test_follower_rejects_attachment_crc_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad-attachment-crc.mcap"
    data = bytearray(
        _file_bytes(
            Header("", "test"),
            Attachment(0, 0, "note", "text/plain", b"hello"),
            DataEnd(0),
            Footer(0, 0, 0),
        )
    )
    attachment_end = _first_record_end(data, Opcode.ATTACHMENT)
    data[attachment_end - 1] ^= 0xFF
    path.write_bytes(data)

    with McapFollower.open(path, validate_crc=True) as follower, pytest.raises(CRCValidationError):
        follower.poll_messages()


def test_follower_rejects_conflicting_schema_and_channel_ids(tmp_path: Path) -> None:
    conflicting_schema = tmp_path / "conflicting-schema.mcap"
    conflicting_schema.write_bytes(
        _file_bytes(
            Header("", "test"),
            Schema(1, "first", "jsonschema", b"{}"),
            Schema(1, "second", "jsonschema", b"{}"),
        )
    )
    with (
        McapFollower.open(conflicting_schema) as follower,
        pytest.raises(ValueError, match="conflicting schema"),
    ):
        follower.poll_messages()

    conflicting_channel = tmp_path / "conflicting-channel.mcap"
    conflicting_channel.write_bytes(
        _file_bytes(
            Header("", "test"),
            Channel(1, 0, "/first", "json", {}),
            Channel(1, 0, "/second", "json", {}),
        )
    )
    with (
        McapFollower.open(conflicting_channel) as follower,
        pytest.raises(ValueError, match="conflicting channel"),
    ):
        follower.poll_messages()


def test_follower_rejects_missing_schema_and_channel_references(tmp_path: Path) -> None:
    missing_schema = tmp_path / "missing-schema.mcap"
    missing_schema.write_bytes(
        _file_bytes(Header("", "test"), Channel(1, 1, "/sample", "json", {}))
    )
    with McapFollower.open(missing_schema) as follower, pytest.raises(SchemaNotFoundError):
        follower.poll_messages()

    missing_channel = tmp_path / "missing-channel.mcap"
    missing_channel.write_bytes(_file_bytes(Header("", "test"), Message(1, 0, 0, 0, b"{}")))
    with McapFollower.open(missing_channel) as follower, pytest.raises(ChannelNotFoundError):
        follower.poll_messages()


def test_follower_ignores_non_content_record(tmp_path: Path) -> None:
    path = tmp_path / "empty.mcap"
    path.touch()

    with McapFollower.open(path) as follower:
        follower._process_content_record(Header("", "test"))


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


def test_follower_chunk_messages_own_payload_bytes(tmp_path: Path) -> None:
    path = tmp_path / "complete.mcap"
    path.write_bytes(_recording_bytes(message_count=3, chunk_size=1024))

    with McapFollower.open(path) as follower:
        batch = follower.poll_messages()

    assert batch.messages
    assert all(isinstance(message.data, bytes) for _schema, _channel, message in batch.messages)


def test_follower_concurrent_writer_soak_emits_every_message_once(tmp_path: Path) -> None:
    data = _recording_bytes(
        message_count=500,
        chunk_size=512,
        compression=CompressionType.ZSTD,
    )
    path = tmp_path / "growing.mcap"
    path.touch()
    writer_done = threading.Event()

    def append_recording() -> None:
        try:
            with path.open("ab", buffering=0) as stream:
                offset = 0
                block_sizes = (1, 17, 257, 4096, 31)
                while offset < len(data):
                    block_size = block_sizes[offset % len(block_sizes)]
                    end = min(offset + block_size, len(data))
                    stream.write(data[offset:end])
                    offset = end
                    time.sleep(0.0001)
        finally:
            writer_done.set()

    writer = threading.Thread(target=append_recording, daemon=True)
    writer.start()
    messages = []
    deadline = time.monotonic() + 10
    with McapFollower.open(path, validate_crc=True) as follower:
        while time.monotonic() < deadline:
            batch = follower.poll_messages(max_messages=7, max_bytes=2048)
            messages.extend(batch.messages)
            if batch.is_final:
                break
            if not batch.messages:
                writer_done.wait(0.001)
        else:
            pytest.fail("follower did not reach trailing magic before timeout")
    writer.join(timeout=1)

    assert not writer.is_alive()
    assert [message.sequence for _schema, _channel, message in messages] == list(range(500))


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


def test_follower_detects_replacement_during_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "recording.mcap"
    data = _recording_bytes()
    path.write_bytes(data)
    replacement = tmp_path / "replacement.mcap"
    replacement.write_bytes(data)
    original = McapFollower._process_complete_record
    was_replaced = False

    def replace_after_record(self, opcode, record, header, body) -> None:
        nonlocal was_replaced
        original(self, opcode, record, header, body)
        if not was_replaced:
            replacement.replace(path)
            was_replaced = True

    monkeypatch.setattr(McapFollower, "_process_complete_record", replace_after_record)

    with McapFollower.open(path) as follower, pytest.raises(McapFileReplacedError):
        follower.poll_messages()

    assert was_replaced
