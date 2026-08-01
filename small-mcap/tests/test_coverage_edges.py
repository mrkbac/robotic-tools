import io
import struct
import sys
import zlib
from pathlib import Path

import pytest
import small_mcap.mcap_file as mcap_file_module
import small_mcap.reader as reader_module
import small_mcap.rebuild as rebuild_module
import small_mcap.records as records_module
from small_mcap.exceptions import (
    ChannelNotFoundError,
    CRCValidationError,
    InvalidMagicError,
    RecordLengthLimitExceededError,
    SchemaNotFoundError,
    UnsupportedCompressionError,
)
from small_mcap.json_decoder import JSONDecoderFactory
from small_mcap.mcap_file import McapFile
from small_mcap.reader import (
    _breakup_chunk_data_with_indexes,
    _get_chunk_data_stream,
    _iter_stream_chunks,
    _LoadedChunk,
    _predecompress_chunk,
    _read_inner,
    _read_message_indexed,
    _read_message_seeking_unchunked,
    read_message,
)
from small_mcap.rebuild import _breakup_chunk_with_offsets, read_info_approximate, rebuild_summary
from small_mcap.records import (
    MAGIC,
    OPCODE_AND_LEN_STRUCT,
    Channel,
    Chunk,
    ChunkIndex,
    Footer,
    Header,
    LazyChunk,
    Message,
    MessageIndex,
    Opcode,
    Schema,
    Summary,
)
from small_mcap.remapper import Remapper


def _chunk_index(**overrides) -> ChunkIndex:
    values = {
        "message_start_time": 0,
        "message_end_time": 1,
        "chunk_start_offset": 0,
        "chunk_length": 0,
        "message_index_offsets": {},
        "message_index_length": 0,
        "compression": "",
        "compressed_size": 0,
        "uncompressed_size": 0,
    }
    values.update(overrides)
    return ChunkIndex(**values)


def test_small_factories_and_error_messages() -> None:
    assert JSONDecoderFactory().decoder_for("cdr", None) is None
    assert "stream_id=7" in str(ChannelNotFoundError(3, stream_id=7))

    remapper = Remapper()
    assert remapper.channel_id_map(1) == {}
    with pytest.raises(ChannelNotFoundError, match="stream_id=1"):
        remapper.remap_message(1, Message(3, 0, 0, 0, b""))


def test_record_read_errors_and_message_index_edges(monkeypatch) -> None:
    with pytest.raises(EOFError, match="record header"):
        Header.read_record(io.BytesIO(b"short"))
    with pytest.raises(EOFError, match="record content"):
        Header.read_record(io.BytesIO(OPCODE_AND_LEN_STRUCT.pack(Opcode.HEADER, 2) + b"x"))
    with pytest.raises(ValueError, match="Expected opcode"):
        Header.read_record(io.BytesIO(OPCODE_AND_LEN_STRUCT.pack(Opcode.FOOTER, 20) + bytes(20)))

    malformed = struct.pack("<HI", 1, 1) + b"x"
    with pytest.raises(struct.error):
        _ = MessageIndex(1, raw_content=malformed).timestamps

    index = MessageIndex(1, timestamps=[10], offsets=[20])
    assert index != "not an index"
    assert "channel_id=1" in repr(index)
    raw_index = MessageIndex.read(struct.pack("<HIQQ", 1, 16, 10, 20))
    assert raw_index.num_entries == 1

    monkeypatch.setattr(records_module.sys, "byteorder", "big")
    output = io.BytesIO()
    index.write_record_to(output)
    decoded = MessageIndex.read_record(io.BytesIO(output.getvalue()))
    assert decoded.timestamps == [10]
    assert decoded.offsets == [20]


def test_chunk_validation_and_indexed_opcode_errors() -> None:
    bad_compression = Chunk(0, 0, 0, 0, 123, b"")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedCompressionError, match="must be a string"):
        _get_chunk_data_stream(bad_compression)

    chunk = Chunk(0, 0, 1, 123, "", b"x")
    with pytest.raises(CRCValidationError):
        _predecompress_chunk(chunk, validate_crc=True)

    schema_buffer = io.BytesIO()
    Schema(1, "example/Message", "ros2msg", b"").write_record_to(schema_buffer)
    with pytest.raises(Exception, match="illegal opcode"):
        list(
            _breakup_chunk_data_with_indexes(
                schema_buffer.getvalue(),
                [MessageIndex(1, timestamps=[0], offsets=[0])],
                reverse=False,
            )
        )


def test_read_inner_reports_missing_references() -> None:
    message = Message(9, 0, 0, 0, b"")
    with pytest.raises(ChannelNotFoundError):
        list(_read_inner([message], lambda *_: True, set(), 0, sys.maxsize))

    channel = Channel(1, 9, "/topic", "cdr", {})
    with pytest.raises(SchemaNotFoundError):
        list(_read_inner([channel], lambda *_: True, set(), 0, sys.maxsize))


def test_seeking_reader_rejects_bad_inputs_and_duplicate_channel() -> None:
    with pytest.raises(InvalidMagicError):
        list(_read_message_seeking_unchunked(io.BytesIO(b"not mcap"), lambda *_: True, 0, 1))

    oversized = MAGIC + OPCODE_AND_LEN_STRUCT.pack(
        Opcode.MESSAGE, reader_module._RECORD_SIZE_LIMIT + 1
    )
    with pytest.raises(RecordLengthLimitExceededError):
        list(_read_message_seeking_unchunked(io.BytesIO(oversized), lambda *_: True, 0, 1))

    channel = Channel(1, 0, "/topic", "raw", {})
    body = io.BytesIO(MAGIC)
    body.seek(0, io.SEEK_END)
    channel.write_record_to(body)
    channel.write_record_to(body)
    Footer(0, 0, 0).write_record_to(body)
    body.seek(0)
    assert list(_read_message_seeking_unchunked(body, lambda *_: True, 0, 1)) == []

    for record in (
        Channel(1, 0, "/topic", "raw", {}),
        Schema(1, "example/Message", "ros2msg", b""),
    ):
        truncated = io.BytesIO(MAGIC)
        truncated.seek(0, io.SEEK_END)
        encoded = io.BytesIO()
        record.write_record_to(encoded)
        truncated.write(encoded.getvalue()[:-1])
        truncated.seek(0)
        assert list(_read_message_seeking_unchunked(truncated, lambda *_: True, 0, 1)) == []

    chunk_payload = io.BytesIO()
    Schema(1, "example/Message", "ros2msg", b"").write_record_to(chunk_payload)
    Channel(1, 1, "/topic", "cdr", {}).write_record_to(chunk_payload)
    Message(1, 0, 0, 0, b"value").write_record_to(chunk_payload)
    chunk = Chunk(0, 0, len(chunk_payload.getvalue()), 0, "", chunk_payload.getvalue())
    chunked = io.BytesIO(MAGIC)
    chunked.seek(0, io.SEEK_END)
    chunk.write_record_to(chunked)
    Footer(0, 0, 0).write_record_to(chunked)
    chunked.seek(0)
    decoded = list(_read_message_seeking_unchunked(chunked, lambda *_: True, 0, 1))
    assert decoded[0][2].data == b"value"

    missing_channel_payload = io.BytesIO()
    Message(9, 0, 0, 0, b"value").write_record_to(missing_channel_payload)
    missing_channel_chunk = Chunk(
        0, 0, len(missing_channel_payload.getvalue()), 0, "", missing_channel_payload.getvalue()
    )
    missing_channel = io.BytesIO(MAGIC)
    missing_channel.seek(0, io.SEEK_END)
    missing_channel_chunk.write_record_to(missing_channel)
    missing_channel.seek(0)
    with pytest.raises(ChannelNotFoundError):
        list(_read_message_seeking_unchunked(missing_channel, lambda *_: True, 0, 1))


def test_indexed_reader_skips_fully_excluded_chunk() -> None:
    index = _chunk_index(message_index_offsets={1: 10})
    summary = Summary(
        channels={1: Channel(1, 0, "/excluded", "raw", {})},
        chunk_indexes=[index],
    )

    messages = _read_message_indexed(
        summary,
        lambda *_: False,
        0,
        sys.maxsize,
        False,
        False,
        0,
        lambda *_: pytest.fail("excluded chunk must not be loaded"),
    )

    assert list(messages) == []
    with pytest.raises(TypeError, match="Unsupported stream type"):
        list(read_message(123))  # type: ignore[arg-type]


def test_indexed_reader_merges_overlapping_chunks_with_definition_records() -> None:
    payload = io.BytesIO()
    Schema(1, "example/Message", "ros2msg", b"").write_record_to(payload)
    Channel(1, 1, "/topic", "cdr", {}).write_record_to(payload)
    Message(1, 0, 1, 1, b"value").write_record_to(payload)
    loaded = _LoadedChunk(None, (), payload.getvalue())
    indexes = [
        _chunk_index(message_start_time=0, message_end_time=2, chunk_start_offset=1),
        _chunk_index(message_start_time=1, message_end_time=3, chunk_start_offset=2),
    ]
    summary = Summary(
        schemas={1: Schema(1, "example/Message", "ros2msg", b"")},
        channels={1: Channel(1, 1, "/topic", "cdr", {})},
        chunk_indexes=indexes,
    )

    messages = list(
        _read_message_indexed(
            summary,
            lambda *_: True,
            0,
            sys.maxsize,
            False,
            False,
            0,
            lambda *_: loaded,
        )
    )

    assert [message.data for _, _, message in messages] == [b"value", b"value"]


def test_stream_chunk_iterator_flushes_final_pending_chunk(monkeypatch) -> None:
    chunk = Chunk(0, 0, 0, 0, "", b"")
    monkeypatch.setattr(reader_module, "stream_reader", lambda *_args, **_kwargs: iter([chunk]))

    assert list(_iter_stream_chunks(io.BytesIO(), False)) == [(chunk, [])]


def test_get_summary_rejects_non_footer(monkeypatch) -> None:
    stream = io.BytesIO(bytes(100) + MAGIC)
    monkeypatch.setattr(
        reader_module,
        "stream_reader",
        lambda *_args, **_kwargs: iter([Header("", "test")]),
    )

    assert reader_module.get_summary(stream) is None


class _NonSeekable(io.BytesIO):
    def seekable(self) -> bool:
        return False


def test_approximate_info_handles_unseekable_and_indexless_chunks(monkeypatch) -> None:
    assert read_info_approximate(_NonSeekable()) is None
    monkeypatch.setattr(rebuild_module, "get_header", lambda _: Header("", "test"))
    monkeypatch.setattr(rebuild_module, "get_summary", lambda _: None)
    assert read_info_approximate(io.BytesIO()) is None
    monkeypatch.setattr(
        rebuild_module,
        "get_summary",
        lambda _: Summary(chunk_indexes=[_chunk_index(chunk_start_offset=10)]),
    )

    result = read_info_approximate(io.BytesIO())

    assert result is not None
    assert result.chunk_information == {10: []}


def test_breakup_chunk_with_offsets_handles_schema_and_channel() -> None:
    payload = io.BytesIO()
    Schema(1, "example/Message", "ros2msg", b"").write_record_to(payload)
    Channel(1, 1, "/topic", "cdr", {}).write_record_to(payload)
    chunk_data = payload.getvalue()
    chunk = Chunk(0, 0, len(chunk_data), zlib.crc32(chunk_data), "", chunk_data)

    records = list(_breakup_chunk_with_offsets(chunk, validate_crc=True))

    assert [type(record) for _, record in records] == [Schema, Channel]


def test_mcap_file_open_and_read_edges(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        McapFile.open(tmp_path / "missing.mcap", chunk_cache_bytes=-1)

    opened = io.BytesIO(b"invalid")
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: opened)

    def broken_summary(_stream):
        raise ValueError("broken summary")

    monkeypatch.setattr(mcap_file_module, "get_summary", broken_summary)
    with pytest.raises(ValueError, match="broken summary"):
        McapFile.open(tmp_path / "invalid.mcap")
    assert opened.closed

    file = McapFile(tmp_path / "short.mcap", io.BytesIO(b"x"), None, 0)
    with pytest.raises(Exception, match="expected 2 bytes"):
        file._read_at(0, 2)


def test_mcap_file_chunk_cache_race_uses_existing_entry(monkeypatch, tmp_path: Path) -> None:
    index = _chunk_index(chunk_start_offset=10, chunk_length=5)
    file = McapFile(tmp_path / "file.mcap", io.BytesIO(), Summary(), 100)
    key = (10, False)
    existing = _LoadedChunk(None, (), b"cached")

    def read_at(_offset: int, _length: int) -> bytes:
        file._chunk_cache[key] = existing
        return b"raw"

    monkeypatch.setattr(file, "_read_at", read_at)
    monkeypatch.setattr(
        mcap_file_module,
        "_read_chunk_and_indexes",
        lambda _: (Chunk(0, 0, 0, 0, "", b""), []),
    )
    monkeypatch.setattr(
        mcap_file_module,
        "_get_chunk_data_stream",
        lambda *_args, **_kwargs: b"new",
    )

    assert file._load_chunk(index, False) is existing


def test_rebuild_strict_mode_reraises_stream_and_finish_errors(monkeypatch) -> None:
    def broken_reader(*_args, **_kwargs):
        yield Header("", "test")
        raise ValueError("broken stream")

    monkeypatch.setattr(rebuild_module, "stream_reader", broken_reader)
    with pytest.raises(ValueError, match="broken stream"):
        rebuild_summary(
            io.BytesIO(),
            validate_crc=False,
            calculate_channel_sizes=False,
            exact_sizes=True,
            allow_incomplete_tail_only=True,
        )


def test_rebuild_registers_in_chunk_definitions_and_estimates_sizes(monkeypatch) -> None:
    lazy = LazyChunk(0, 10, 100, 0, "", 0, 0)
    schema = Schema(1, "example/Message", "ros2msg", b"")
    channel = Channel(1, 1, "/topic", "cdr", {})
    monkeypatch.setattr(
        rebuild_module,
        "stream_reader",
        lambda *_args, **_kwargs: iter([Header("", "test"), lazy]),
    )
    monkeypatch.setattr(
        LazyChunk,
        "to_chunk",
        lambda *_args, **_kwargs: Chunk(0, 10, 0, 0, "", b""),
    )
    monkeypatch.setattr(
        rebuild_module,
        "_breakup_chunk_with_offsets",
        lambda *_args, **_kwargs: iter([(0, channel), (10, schema)]),
    )

    rebuilt = rebuild_summary(
        io.BytesIO(),
        validate_crc=False,
        calculate_channel_sizes=False,
        exact_sizes=False,
    )

    assert rebuilt.summary.channels == {1: channel}
    assert rebuilt.summary.schemas == {1: schema}

    index = MessageIndex(1, timestamps=[1, 2], offsets=[0, 40])
    monkeypatch.setattr(
        rebuild_module,
        "stream_reader",
        lambda *_args, **_kwargs: iter([Header("", "test"), lazy, index]),
    )
    estimated = rebuild_summary(
        io.BytesIO(),
        validate_crc=False,
        calculate_channel_sizes=True,
        exact_sizes=False,
        allow_incomplete_tail_only=True,
    )

    assert estimated.estimated_channel_sizes is True
    assert estimated.channel_sizes[1] > 0

    lazy = LazyChunk(0, 0, 0, 0, "", 0, 0)
    monkeypatch.setattr(
        rebuild_module,
        "stream_reader",
        lambda *_args, **_kwargs: iter([Header("", "test"), lazy]),
    )
    monkeypatch.setattr(
        LazyChunk,
        "to_chunk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("broken chunk")),
    )
    with pytest.raises(ValueError, match="broken chunk"):
        rebuild_summary(
            io.BytesIO(),
            validate_crc=False,
            calculate_channel_sizes=False,
            exact_sizes=True,
            allow_incomplete_tail_only=True,
        )
