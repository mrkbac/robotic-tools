from __future__ import annotations

import io
import struct
import zlib
from typing import TYPE_CHECKING

from lz4.frame import compress as lz4_compress
from pymcap_cli import doctor as doctor_module
from pymcap_cli.cmd import doctor_cmd
from pymcap_cli.doctor import DoctorReport, examine_mcap
from rich.console import Console
from small_mcap import CompressionType, McapWriter
from small_mcap.records import (
    MAGIC,
    Attachment,
    AttachmentIndex,
    Channel,
    Chunk,
    ChunkIndex,
    DataEnd,
    Footer,
    Header,
    McapRecord,
    Message,
    MessageIndex,
    Metadata,
    MetadataIndex,
    Opcode,
    Schema,
    Statistics,
    SummaryOffset,
)
from zstandard import ZstdCompressor

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _record_bytes(record: McapRecord) -> bytes:
    buffer = io.BytesIO()
    record.write_record_to(buffer)
    return buffer.getvalue()


def _padded_record_bytes(record: McapRecord, padding: bytes = b"pad") -> bytes:
    raw = _record_bytes(record)
    opcode, length = struct.unpack_from("<BQ", raw, 0)
    body = raw[9:]
    return struct.pack("<BQ", opcode, length + len(padding)) + body + padding


def _finish_data_section(prefix: bytes, *, data_crc: int | None = None) -> bytes:
    crc = zlib.crc32(prefix) if data_crc is None else data_crc
    return prefix + _record_bytes(DataEnd(crc)) + _record_bytes(Footer(0, 0, 0)) + MAGIC


def _finish_with_summary(
    prefix: bytes,
    summary_records: list[bytes],
    summary_offset_records: list[bytes] | None = None,
) -> bytes:
    summary_offset_records = [] if summary_offset_records is None else summary_offset_records
    data_end = _record_bytes(DataEnd(zlib.crc32(prefix)))
    summary = b"".join(summary_records)
    summary_offsets = b"".join(summary_offset_records)
    summary_start = len(prefix) + len(data_end) if summary else 0
    summary_offset_start = len(prefix) + len(data_end) + len(summary) if summary_offsets else 0
    return (
        prefix
        + data_end
        + summary
        + summary_offsets
        + _record_bytes(Footer(summary_start, summary_offset_start, 0))
        + MAGIC
    )


def _simple_chunked_mcap(
    compression: CompressionType = CompressionType.ZSTD,
) -> bytes:
    buffer = io.BytesIO()
    writer = McapWriter(buffer, chunk_size=1024, compression=compression)
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=1)
    for index in range(3):
        writer.add_message(
            channel_id=1,
            log_time=index,
            publish_time=index,
            data=b"{}",
        )
    writer.finish()
    return buffer.getvalue()


def _build_chunk(log_times: list[int]) -> bytes:
    payload = b""
    for log_time in log_times:
        payload += _record_bytes(
            Message(
                channel_id=1,
                sequence=0,
                log_time=log_time,
                publish_time=log_time,
                data=b"{}",
            )
        )
    return _record_bytes(
        Chunk(
            message_start_time=min(log_times),
            message_end_time=max(log_times),
            uncompressed_size=len(payload),
            uncompressed_crc=0,
            compression="",
            data=payload,
        )
    )


def _build_zstd_chunk_without_content_size(log_times: list[int]) -> bytes:
    payload = b""
    for log_time in log_times:
        payload += _record_bytes(
            Message(
                channel_id=1,
                sequence=0,
                log_time=log_time,
                publish_time=log_time,
                data=b"{}",
            )
        )
    compressed = ZstdCompressor(write_content_size=False).compress(payload)
    return _record_bytes(
        Chunk(
            message_start_time=min(log_times),
            message_end_time=max(log_times),
            uncompressed_size=len(payload),
            uncompressed_crc=0,
            compression="zstd",
            data=compressed,
        )
    )


def _build_lz4_chunk(log_times: list[int]) -> bytes:
    payload = b""
    for log_time in log_times:
        payload += _record_bytes(
            Message(
                channel_id=1,
                sequence=0,
                log_time=log_time,
                publish_time=log_time,
                data=b"{}",
            )
        )
    compressed = lz4_compress(payload)
    return _record_bytes(
        Chunk(
            message_start_time=min(log_times),
            message_end_time=max(log_times),
            uncompressed_size=len(payload),
            uncompressed_crc=0,
            compression="lz4",
            data=compressed,
        )
    )


def _interleaved_chunks_mcap(log_times_per_chunk: list[list[int]]) -> bytes:
    data = MAGIC
    data += _record_bytes(Header(profile="", library="doctor-test"))
    data += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    data += _record_bytes(
        Channel(
            id=1,
            schema_id=1,
            topic="/test",
            message_encoding="json",
            metadata={},
        )
    )
    for chunk_log_times in log_times_per_chunk:
        data += _build_chunk(chunk_log_times)
    return _finish_data_section(data)


def _simple_unchunked_messages(log_times: list[int], *, data_crc: int | None = None) -> bytes:
    data = MAGIC
    data += _record_bytes(Header(profile="", library="doctor-test"))
    data += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    data += _record_bytes(
        Channel(
            id=1,
            schema_id=1,
            topic="/test",
            message_encoding="json",
            metadata={},
        )
    )
    for log_time in log_times:
        data += _record_bytes(
            Message(
                channel_id=1,
                sequence=0,
                log_time=log_time,
                publish_time=log_time,
                data=b"{}",
            )
        )
    return _finish_data_section(data, data_crc=data_crc)


def _empty_message_index_mcap() -> bytes:
    data = MAGIC
    data += _record_bytes(Header(profile="", library="doctor-test"))
    data += _record_bytes(
        Chunk(
            message_start_time=0,
            message_end_time=0,
            uncompressed_size=0,
            uncompressed_crc=0,
            compression="",
            data=b"",
        )
    )
    data += _record_bytes(MessageIndex(channel_id=1, timestamps=[], offsets=[]))
    return _finish_data_section(data)


def _report(data: bytes) -> DoctorReport:
    return examine_mcap(io.BytesIO(data), len(data), "test.mcap")


def _codes(report: DoctorReport) -> set[str]:
    return {finding.code.value for finding in report.findings}


def _severity(report: DoctorReport, code: str) -> str:
    for finding in report.findings:
        if finding.code == code:
            return finding.severity
    raise AssertionError(f"missing finding {code}")


def test_doctor_accepts_small_mcap_writer_output() -> None:
    report = _report(_simple_chunked_mcap())

    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.message_count == 3
    assert report.chunk_count == 1


def test_doctor_accepts_lz4_mcap_writer_output() -> None:
    report = _report(_simple_chunked_mcap(CompressionType.LZ4))

    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.message_count == 3
    assert report.chunk_count == 1


def test_doctor_reports_data_end_crc_mismatch() -> None:
    report = _report(_simple_unchunked_messages([1], data_crc=123))

    assert report.error_count == 1
    assert "DATA_SECTION_CRC_MISMATCH" in _codes(report)


def test_doctor_reports_non_monotonic_messages_as_info_by_default() -> None:
    report = _report(_simple_unchunked_messages([10, 1]))

    assert report.error_count == 0
    assert report.warning_count == 0
    assert "NON_MONOTONIC_LOG_TIME" in _codes(report)
    assert _severity(report, "NON_MONOTONIC_LOG_TIME") == "info"


def test_doctor_accepts_interleaved_chunks_with_overlapping_time_ranges() -> None:
    report = _report(_interleaved_chunks_mcap([[10, 20, 30], [5, 15, 25]]))

    assert report.error_count == 0
    assert "NON_MONOTONIC_LOG_TIME" not in _codes(report)


def test_doctor_accepts_zstd_chunk_without_frame_content_size() -> None:
    data = MAGIC
    data += _record_bytes(Header(profile="", library="doctor-test"))
    data += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    data += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    data += _build_zstd_chunk_without_content_size([1])

    report = _report(_finish_data_section(data))

    assert report.error_count == 0
    assert report.message_count == 1
    assert "CHUNK_DECOMPRESSION_FAILED" not in _codes(report)


def test_doctor_reports_non_monotonic_messages_within_a_chunk_as_info() -> None:
    report = _report(_interleaved_chunks_mcap([[10, 5, 20]]))

    assert report.error_count == 0
    assert "NON_MONOTONIC_LOG_TIME" in _codes(report)
    assert _severity(report, "NON_MONOTONIC_LOG_TIME") == "info"


def test_doctor_strict_message_order_errors_for_non_monotonic_messages() -> None:
    data = _simple_unchunked_messages([10, 1])
    report = examine_mcap(
        io.BytesIO(data),
        len(data),
        "test.mcap",
        strict_message_order=True,
    )

    assert report.error_count == 1
    assert "NON_MONOTONIC_LOG_TIME" in _codes(report)


def test_doctor_reports_empty_message_index_as_info() -> None:
    report = _report(_empty_message_index_mcap())

    assert report.error_count == 0
    assert "EMPTY_MESSAGE_INDEX" in _codes(report)
    assert "MISSING_CHUNK_INDEXES" in _codes(report)
    assert "EMPTY_CHUNK" in _codes(report)
    assert _severity(report, "EMPTY_MESSAGE_INDEX") == "info"
    assert _severity(report, "MISSING_CHUNK_INDEXES") == "info"
    assert _severity(report, "EMPTY_CHUNK") == "info"


def test_doctor_reports_schema_and_channel_advisory_warnings() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(Schema(id=1, name="", encoding="customschema", data=b"schema"))
    for channel_id in (1, 2):
        prefix += _record_bytes(
            Channel(
                id=channel_id,
                schema_id=1,
                topic="",
                message_encoding="json",
                metadata={},
            )
        )

    report = _report(_finish_data_section(prefix))

    for code in (
        "SCHEMA_NAME_EMPTY",
        "SCHEMA_ENCODING_UNKNOWN",
        "CHANNEL_TOPIC_EMPTY",
        "CHANNEL_TOPIC_DUPLICATE",
    ):
        assert code in _codes(report)
        assert _severity(report, code) == "warning"


def test_doctor_reports_message_attachment_and_metadata_advisory_warnings() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    prefix += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    prefix += _record_bytes(
        Message(channel_id=1, sequence=1, log_time=5, publish_time=10, data=b"{}")
    )
    prefix += _record_bytes(
        Attachment(log_time=1, create_time=1, name="note.txt", media_type="", data=b"x")
    )
    prefix += _record_bytes(Metadata(name="device", metadata={"serial": "abc"}))
    prefix += _record_bytes(Metadata(name="device", metadata={"serial": "def"}))

    report = _report(_finish_data_section(prefix))

    for code in (
        "MESSAGE_PUBLISH_TIME_AFTER_LOG_TIME",
        "ATTACHMENT_MEDIA_TYPE_EMPTY",
        "METADATA_NAME_DUPLICATE",
    ):
        assert code in _codes(report)
        assert _severity(report, code) == "warning"


def test_doctor_reports_chunk_compression_advisory_warning() -> None:
    data = MAGIC
    data += _record_bytes(Header(profile="", library="doctor-test"))
    data += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    data += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    data += _build_lz4_chunk([1])

    report = _report(_finish_data_section(data))

    assert "CHUNK_COMPRESSION_INEFFICIENT" in _codes(report)
    assert _severity(report, "CHUNK_COMPRESSION_INEFFICIENT") == "warning"


def test_doctor_reports_index_records_without_summary_warning() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    data_end = _record_bytes(DataEnd(zlib.crc32(prefix)))
    statistics = _record_bytes(
        Statistics(
            message_count=0,
            schema_count=0,
            channel_count=0,
            attachment_count=0,
            metadata_count=0,
            chunk_count=0,
            message_start_time=0,
            message_end_time=0,
            channel_message_counts={},
        )
    )

    report = _report(prefix + data_end + statistics + _record_bytes(Footer(0, 0, 0)) + MAGIC)

    assert "INDEX_SECTION_PRESENT_WITHOUT_SUMMARY" in _codes(report)
    assert _severity(report, "INDEX_SECTION_PRESENT_WITHOUT_SUMMARY") == "warning"


def test_doctor_reports_absence_and_sequence_infos() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(
        Channel(id=1, schema_id=0, topic="/raw", message_encoding="json", metadata={})
    )
    prefix += _record_bytes(
        Message(channel_id=1, sequence=0, log_time=1, publish_time=1, data=b"{}")
    )

    report = _report(_finish_data_section(prefix))

    for code in (
        "NO_ATTACHMENTS",
        "NO_METADATA",
        "NO_SCHEMAS",
        "NO_SUMMARY_SECTION",
        "ZERO_MESSAGE_SEQUENCE",
    ):
        assert code in _codes(report)
        assert _severity(report, code) == "info"


def test_doctor_reports_uncompressed_and_small_chunk_infos() -> None:
    data = MAGIC
    data += _record_bytes(Header(profile="", library="doctor-test"))
    data += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    data += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    for log_time in range(10):
        data += _build_chunk([log_time])

    report = _report(_finish_data_section(data))

    assert "NO_COMPRESSION" in _codes(report)
    assert "SMALL_CHUNKS" in _codes(report)
    assert _severity(report, "NO_COMPRESSION") == "info"
    assert _severity(report, "SMALL_CHUNKS") == "info"


def test_doctor_reports_mixed_compression_and_large_chunk_infos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor_module, "_LARGE_CHUNK_BYTES", 1)
    data = MAGIC
    data += _record_bytes(Header(profile="", library="doctor-test"))
    data += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    data += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    data += _build_chunk([1])
    data += _build_zstd_chunk_without_content_size([2])

    report = _report(_finish_data_section(data))

    assert "MIXED_COMPRESSION_TYPES" in _codes(report)
    assert "LARGE_CHUNK" in _codes(report)
    assert _severity(report, "MIXED_COMPRESSION_TYPES") == "info"
    assert _severity(report, "LARGE_CHUNK") == "info"


def test_doctor_accepts_trailing_extension_bytes_in_extendable_records() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _padded_record_bytes(
        Attachment(log_time=0, create_time=0, name="note.txt", media_type="text/plain", data=b"x")
    )
    prefix += _record_bytes(
        Chunk(
            message_start_time=0,
            message_end_time=0,
            uncompressed_size=0,
            uncompressed_crc=0,
            compression="",
            data=b"",
        )
    )
    prefix += _padded_record_bytes(MessageIndex(channel_id=1, timestamps=[], offsets=[]))

    data_end_len = len(_record_bytes(DataEnd(0)))
    summary_start = len(prefix) + data_end_len
    statistics = _padded_record_bytes(
        Statistics(
            message_count=0,
            schema_count=0,
            channel_count=0,
            attachment_count=1,
            metadata_count=0,
            chunk_count=1,
            message_start_time=0,
            message_end_time=0,
            channel_message_counts={},
        )
    )
    summary_offset = _padded_record_bytes(
        SummaryOffset(
            group_opcode=Opcode.STATISTICS,
            group_start=summary_start,
            group_length=len(statistics),
        )
    )
    report = _report(_finish_with_summary(prefix, [statistics], [summary_offset]))

    assert report.error_count == 0
    assert "RECORD_PARSE_ERROR" not in _codes(report)
    assert "TRAILING_RECORD_BYTES" in _codes(report)
    assert _severity(report, "TRAILING_RECORD_BYTES") == "info"


def test_doctor_reports_summary_schema_and_channel_mismatch() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    prefix += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    summary = [
        _record_bytes(Schema(id=1, name="other", encoding="json", data=b"{}")),
        _record_bytes(
            Channel(id=1, schema_id=1, topic="/other", message_encoding="json", metadata={})
        ),
    ]

    report = _report(_finish_with_summary(prefix, summary))

    assert "SUMMARY_SCHEMA_MISMATCH" in _codes(report)
    assert "SUMMARY_CHANNEL_MISMATCH" in _codes(report)


def test_doctor_reports_chunk_inner_schema_field_violations() -> None:
    chunk_payload = _record_bytes(Schema(id=0, name="empty", encoding="", data=b"not empty"))
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(
        Chunk(
            message_start_time=0,
            message_end_time=0,
            uncompressed_size=len(chunk_payload),
            uncompressed_crc=0,
            compression="",
            data=chunk_payload,
        )
    )

    report = _report(_finish_data_section(prefix))

    assert "RESERVED_SCHEMA_ID" in _codes(report)
    assert "SCHEMA_DATA_WITH_EMPTY_ENCODING" in _codes(report)


def test_doctor_reports_missing_statistics_channel_count() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    for channel_id in (1, 2):
        prefix += _record_bytes(
            Channel(
                id=channel_id,
                schema_id=1,
                topic=f"/test/{channel_id}",
                message_encoding="json",
                metadata={},
            )
        )
    for channel_id in (1, 2):
        prefix += _record_bytes(
            Message(
                channel_id=channel_id,
                sequence=0,
                log_time=channel_id,
                publish_time=channel_id,
                data=b"{}",
            )
        )
    summary = [
        _record_bytes(
            Channel(
                id=1,
                schema_id=1,
                topic="/test/1",
                message_encoding="json",
                metadata={},
            )
        ),
        _record_bytes(
            Channel(
                id=2,
                schema_id=1,
                topic="/test/2",
                message_encoding="json",
                metadata={},
            )
        ),
        _record_bytes(
            Statistics(
                message_count=2,
                schema_count=1,
                channel_count=2,
                attachment_count=0,
                metadata_count=0,
                chunk_count=0,
                message_start_time=1,
                message_end_time=2,
                channel_message_counts={1: 1},
            )
        ),
    ]

    report = _report(_finish_with_summary(prefix, summary))

    assert "STATISTICS_CHANNEL_COUNT_MISSING" in _codes(report)


def test_doctor_reports_duplicate_attachment_and_metadata_indexes() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    attachment_offset = len(prefix)
    attachment = _record_bytes(
        Attachment(log_time=1, create_time=2, name="note.txt", media_type="text/plain", data=b"x")
    )
    prefix += attachment
    metadata_offset = len(prefix)
    metadata = _record_bytes(Metadata(name="device", metadata={"serial": "abc"}))
    prefix += metadata
    attachment_index = _record_bytes(
        AttachmentIndex(
            offset=attachment_offset,
            length=len(attachment),
            log_time=1,
            create_time=2,
            data_size=1,
            name="note.txt",
            media_type="text/plain",
        )
    )
    metadata_index = _record_bytes(
        MetadataIndex(offset=metadata_offset, length=len(metadata), name="device")
    )

    report = _report(
        _finish_with_summary(
            prefix,
            [attachment_index, attachment_index, metadata_index, metadata_index],
        )
    )

    assert "DUPLICATE_ATTACHMENT_INDEX" in _codes(report)
    assert "DUPLICATE_METADATA_INDEX" in _codes(report)


def test_doctor_warns_for_top_level_message_when_chunk_indexes_exist() -> None:
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    prefix += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    prefix += _record_bytes(
        Message(channel_id=1, sequence=0, log_time=1, publish_time=1, data=b"{}")
    )
    chunk_offset = len(prefix)
    chunk = _record_bytes(
        Chunk(
            message_start_time=0,
            message_end_time=0,
            uncompressed_size=0,
            uncompressed_crc=0,
            compression="",
            data=b"",
        )
    )
    prefix += chunk
    chunk_index = _record_bytes(
        ChunkIndex(
            message_start_time=0,
            message_end_time=0,
            chunk_start_offset=chunk_offset,
            chunk_length=len(chunk),
            message_index_offsets={},
            message_index_length=0,
            compression="",
            compressed_size=0,
            uncompressed_size=0,
        )
    )

    report = _report(_finish_with_summary(prefix, [chunk_index]))

    assert report.error_count == 0
    assert "MESSAGE_OUTSIDE_CHUNK_WITH_CHUNK_INDEX" in _codes(report)
    assert _severity(report, "MESSAGE_OUTSIDE_CHUNK_WITH_CHUNK_INDEX") == "warning"


def test_doctor_reports_duplicate_message_index_offsets() -> None:
    payload = _record_bytes(
        Message(channel_id=1, sequence=0, log_time=1, publish_time=1, data=b"{}")
    )
    prefix = MAGIC
    prefix += _record_bytes(Header(profile="", library="doctor-test"))
    prefix += _record_bytes(Schema(id=1, name="test", encoding="json", data=b"{}"))
    prefix += _record_bytes(
        Channel(id=1, schema_id=1, topic="/test", message_encoding="json", metadata={})
    )
    prefix += _record_bytes(
        Chunk(
            message_start_time=1,
            message_end_time=1,
            uncompressed_size=len(payload),
            uncompressed_crc=0,
            compression="",
            data=payload,
        )
    )
    prefix += _record_bytes(MessageIndex(channel_id=1, timestamps=[1, 1], offsets=[0, 0]))

    report = _report(_finish_data_section(prefix))

    assert "DUPLICATE_MESSAGE_INDEX_OFFSET" in _codes(report)


def test_doctor_reports_missing_trailing_magic() -> None:
    data = _simple_chunked_mcap()[:-8] + b"notmagic"
    report = _report(data)

    assert report.error_count >= 1
    assert "BAD_TRAILING_MAGIC" in _codes(report)


def test_doctor_registered_command_accepts_valid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "valid.mcap"
    path.write_bytes(_simple_chunked_mcap())
    output = io.StringIO()
    monkeypatch.setattr(
        doctor_cmd,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=160),
    )

    assert doctor_cmd.doctor(str(path)) == 0
    assert "passed MCAP doctor checks" in output.getvalue()
