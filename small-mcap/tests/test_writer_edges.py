import io
from types import SimpleNamespace
from typing import cast

import pytest
import small_mcap.writer as writer_module
from small_mcap.exceptions import WriterNotStartedError
from small_mcap.reader import stream_reader
from small_mcap.records import Channel, Chunk, LazyChunk, Message, MessageIndex, Opcode, Schema
from small_mcap.writer import (
    CompressionType,
    McapWriter,
    McapWriterRaw,
    _calculate_summary_crc,
    _calculate_summary_offset_start,
    _ChunkBuilder,
    _compress_chunk_data,
    _CRCWriter,
    _write_summary_section,
)


def _chunk() -> Chunk:
    return Chunk(
        message_start_time=10,
        message_end_time=20,
        uncompressed_size=4,
        uncompressed_crc=0,
        compression="",
        data=b"data",
    )


def _index() -> MessageIndex:
    return MessageIndex(channel_id=1, timestamps=[10], offsets=[0])


def test_crc_writer_copy_and_io_methods() -> None:
    output = io.BytesIO()
    writer = _CRCWriter(output, enable_crc=True)
    writer.write(b"prefix")
    writer.copy_from(io.BytesIO(b"payload"), 7, bytearray(3))
    writer.flush()

    assert output.getvalue() == b"prefixpayload"
    assert writer.tell() == 13
    assert writer.crc != 0

    with pytest.raises(EOFError, match="Unexpected EOF"):
        writer.copy_from(io.BytesIO(b"x"), 2, bytearray(1))

    unvalidated_output = io.BytesIO()
    unvalidated = _CRCWriter(unvalidated_output, enable_crc=False)
    unvalidated.copy_from(io.BytesIO(b"ok"), 2, bytearray(2))
    assert unvalidated.crc == 0
    unvalidated.close()
    assert unvalidated_output.closed


def test_summary_helpers_handle_disabled_and_empty_sections() -> None:
    buffer = io.BytesIO()
    offsets = []

    _write_summary_section(buffer, offsets, Opcode.SCHEMA, [], 10)

    assert offsets == []
    assert _calculate_summary_offset_start(10, b"summary", offsets, True) == 0
    assert _calculate_summary_offset_start(10, b"summary", offsets, False) == 0
    assert _calculate_summary_crc(b"summary", 10, offsets, True, False) == 0


def test_raw_writer_validates_state_and_references() -> None:
    writer = McapWriterRaw(io.BytesIO())

    for operation in (
        lambda: writer.add_schema(1, "schema", "jsonschema", b"{}"),
        lambda: writer.add_channel(1, "/topic", "json", 0),
        lambda: writer.add_message(1, 0, b"", 0),
        lambda: writer.add_attachment(0, 0, "name", "text/plain", b""),
        lambda: writer.add_metadata("name", {}),
        lambda: writer.add_chunk(_chunk(), {1: _index()}),
        writer.finish,
    ):
        with pytest.raises(WriterNotStartedError):
            operation()

    writer.start()
    with pytest.raises(RuntimeError, match="already started"):
        writer.start()
    with pytest.raises(ValueError, match="Schema ID cannot be 0"):
        writer.add_schema(0, "schema", "jsonschema", b"{}")
    with pytest.raises(ValueError, match="Schema ID 9 does not exist"):
        writer.add_channel(1, "/topic", "json", 9)
    writer.add_channel(1, "/topic", "json", 0)
    with pytest.raises(ValueError, match="Channel ID 9 does not exist"):
        writer.add_message(9, 0, b"", 0)

    writer.finish()
    writer.finish()
    for operation in (
        lambda: writer.add_schema(1, "schema", "jsonschema", b"{}"),
        lambda: writer.add_channel(2, "/other", "json", 0),
        lambda: writer.add_message(1, 0, b"", 0),
        lambda: writer.add_attachment(0, 0, "name", "text/plain", b""),
        lambda: writer.add_metadata("name", {}),
        lambda: writer.add_chunk(_chunk(), {1: _index()}),
    ):
        with pytest.raises(RuntimeError, match="already finished"):
            operation()


@pytest.mark.parametrize("chunk_size", [0, 1024])
def test_writer_accepts_identical_but_rejects_conflicting_duplicate_ids(chunk_size) -> None:
    output = io.BytesIO()
    writer = McapWriter(
        output,
        chunk_size=chunk_size,
        repeat_schemas=False,
        repeat_channels=False,
    )
    writer.start()

    writer.add_schema(1, "schema", "json", b"{}")
    writer.add_schema(1, "schema", "json", b"{}")
    with pytest.raises(ValueError, match="Conflicting schema ID 1"):
        writer.add_schema(1, "different", "json", b"{}")

    writer.add_channel(1, "/topic", "json", 1)
    writer.add_channel(1, "/topic", "json", 1)
    with pytest.raises(ValueError, match="Conflicting channel ID 1"):
        writer.add_channel(1, "/other", "json", 1)

    writer.finish()
    records = list(stream_reader(io.BytesIO(output.getvalue()), emit_chunks=True))

    assert [record.name for record in records if isinstance(record, Schema)] == [
        "schema",
        "schema",
    ]
    assert [record.topic for record in records if isinstance(record, Channel)] == [
        "/topic",
        "/topic",
    ]


def test_raw_chunk_copy_preserves_input_position_and_reuses_buffer() -> None:
    chunk = _chunk()
    source = io.BytesIO()
    chunk.write_record_to(source)
    lazy = LazyChunk(
        message_start_time=chunk.message_start_time,
        message_end_time=chunk.message_end_time,
        uncompressed_size=chunk.uncompressed_size,
        uncompressed_crc=chunk.uncompressed_crc,
        compression=chunk.compression,
        record_start=0,
        data_len=len(chunk.data),
    )
    source.seek(3)
    writer = McapWriterRaw(io.BytesIO())

    with pytest.raises(WriterNotStartedError):
        writer.add_chunk_raw(source, lazy, {1: _index()})

    writer.start()
    writer.add_chunk_raw(source, lazy, {1: _index()})
    first_buffer = writer._raw_copy_buffer
    writer.add_chunk_raw(source, lazy, {})

    assert source.tell() == 3
    assert writer._raw_copy_buffer is first_buffer
    writer.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        writer.add_chunk_raw(source, lazy, {})


def test_compression_errors_and_fallbacks(monkeypatch) -> None:
    monkeypatch.setattr(writer_module, "_zstd_compress", None)
    with pytest.raises(ImportError, match="zstd compression requires"):
        _compress_chunk_data(b"data", CompressionType.ZSTD)

    monkeypatch.setattr(writer_module, "lz4_compress", None)
    with pytest.raises(ImportError, match="lz4 module not available"):
        _compress_chunk_data(b"data", CompressionType.LZ4)

    with pytest.raises(ValueError, match="Unsupported compression"):
        _compress_chunk_data(b"data", cast("CompressionType", SimpleNamespace()))

    monkeypatch.setattr(writer_module, "_zstd_compress", lambda data, _level: bytes(data))
    assert _compress_chunk_data(b"data", CompressionType.ZSTD) == (b"data", "")
    assert _compress_chunk_data(b"data", CompressionType.NONE) == (b"data", "")


def test_chunk_builder_object_path_and_empty_finalize() -> None:
    builder = _ChunkBuilder(CompressionType.NONE, enable_crcs=True)
    assert builder.extract() is None
    assert builder.finalize() is None

    builder.add(Message(channel_id=1, sequence=0, log_time=20, publish_time=20, data=b"a"))
    builder.add(Message(channel_id=1, sequence=1, log_time=10, publish_time=10, data=b"b"))
    finalized = builder.finalize()

    assert finalized is not None
    chunk, indexes = finalized
    assert (chunk.message_start_time, chunk.message_end_time) == (10, 20)
    assert indexes[1].timestamps == [20, 10]


class _NoEncoderFactory:
    profile = ""
    encoding = ""
    message_encoding = "raw"

    def encoder_for(self, _schema):
        return None


def test_high_level_writer_encoder_and_empty_queue_errors() -> None:
    writer = McapWriter(io.BytesIO(), use_chunking=False)
    writer.start()
    writer.add_channel(1, "/topic", "raw", 0)
    with pytest.raises(RuntimeError, match="encoder_factory"):
        writer.add_message_encode(1, 0, b"", 0)
    writer.encoder_factory = _NoEncoderFactory()
    with pytest.raises(ValueError, match="Channel ID 9"):
        writer.add_message_encode(9, 0, b"", 0)
    writer._drain_one()

    with pytest.raises(ValueError, match="No encoder found"):
        writer.add_message_encode(1, 0, b"", 0)
    writer.finish()


def test_high_level_writer_chunk_passthroughs_finalize_pending_data() -> None:
    writer = McapWriter(
        io.BytesIO(),
        compression=CompressionType.NONE,
        chunk_size=1,
    )
    writer.start()
    writer.add_channel(1, "/topic", "raw", 0)
    writer._submit_or_write_chunk()
    writer.add_message(1, 1, b"a", 1)
    writer.add_chunk(_chunk(), {1: _index()})
    writer.add_message(1, 2, b"b", 2)

    raw_source = io.BytesIO()
    raw_chunk = _chunk()
    raw_chunk.write_record_to(raw_source)
    lazy = LazyChunk(10, 20, 4, 0, "", 0, 4)
    writer.add_chunk_raw(raw_source, lazy, {})

    writer.finish()
