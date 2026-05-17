from __future__ import annotations

import struct
import zlib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import IO, TYPE_CHECKING, cast

import yaml
from lz4.frame import decompress as lz4_decompress
from small_mcap.records import MAGIC, MAGIC_SIZE, Opcode

from pymcap_cli.types.qos import Durability, History, Liveliness, Reliability

if TYPE_CHECKING:
    from collections.abc import Mapping


_QOS_POLICY_FIELDS: tuple[
    tuple[str, type[Reliability | Durability | History | Liveliness]],
    ...,
] = (
    ("reliability", Reliability),
    ("durability", Durability),
    ("history", History),
    ("liveliness", Liveliness),
)


def _qos_issues(metadata: Mapping[str, str] | None) -> list[str]:
    """Return human-readable problems with a channel's QoS metadata.

    Empty list when ``offered_qos_profiles`` is missing or parses cleanly.
    Tolerant parsing lives in :mod:`pymcap_cli.core.qos`; this is the
    paranoid sibling that doctor uses to surface malformed profiles.
    """
    if not metadata:
        return []
    raw = metadata.get("offered_qos_profiles")
    if not raw:
        return []
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return [f"offered_qos_profiles YAML is malformed: {exc}"]
    if not isinstance(loaded, list):
        return [f"offered_qos_profiles must be a YAML list, got {type(loaded).__name__}"]

    issues: list[str] = []
    for index, entry in enumerate(loaded):
        if not isinstance(entry, dict):
            issues.append(f"profile #{index} is not a dict")
            continue
        entry_dict = cast("dict[str, object]", entry)
        for field_name, enum_cls in _QOS_POLICY_FIELDS:
            value = entry_dict.get(field_name)
            if value is None:
                continue
            if isinstance(value, bool):
                issues.append(f"profile #{index}: {field_name} is a bool (expected int or str)")
                continue
            if isinstance(value, int):
                try:
                    enum_cls(value)
                except ValueError:
                    issues.append(f"profile #{index}: unknown {field_name} code {value}")
            elif isinstance(value, str):
                if value.strip().upper() not in enum_cls.__members__:
                    issues.append(f"profile #{index}: unknown {field_name} name {value.strip()!r}")
            else:
                issues.append(
                    f"profile #{index}: {field_name} has unexpected type {type(value).__name__}"
                )
    return issues


if TYPE_CHECKING:
    from collections.abc import Callable

# zstd backend dispatch — Python 3.14 ships `compression.zstd` in the stdlib;
# older interpreters use the third-party `zstandard` package. Either may be
# absent (zstd is an optional extra of small-mcap).
_zstd_decompress: Callable[[bytes | memoryview, int], bytes] | None
_ZstdError: type[Exception]

if TYPE_CHECKING:
    _zstd_decompress = None
    _ZstdError = Exception
else:
    try:
        from compression import zstd as _stdlib_zstd

        _ZstdError = _stdlib_zstd.ZstdError

        def _zstd_decompress(data: bytes | memoryview, _expected_size: int) -> bytes:
            return _stdlib_zstd.decompress(data)
    except ImportError:
        try:
            from zstandard import ZstdDecompressor
            from zstandard import ZstdError as _ZstdError

            def _zstd_decompress(data: bytes | memoryview, expected_size: int) -> bytes:
                return ZstdDecompressor().decompress(data, max_output_size=max(expected_size, 1))
        except ImportError:
            _zstd_decompress = None
            _ZstdError = Exception


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_RECORD_HEADER = struct.Struct("<BQ")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_MESSAGE_HEADER = struct.Struct("<HIQQ")
_CHUNK_HEADER = struct.Struct("<QQQI")
_ATTACHMENT_INDEX_FIXED = struct.Struct("<QQQQQ")
_CHUNK_INDEX_FIXED = struct.Struct("<QQQQI")
_INDEX_OFFSET_ENTRY = struct.Struct("<HQ")
_STATISTICS_FIXED = struct.Struct("<QHIIIIQQI")
_SUMMARY_OFFSET = struct.Struct("<BQQ")
_KNOWN_SCHEMA_ENCODINGS = frozenset(
    {
        "flatbuffer",
        "idl",
        "json",
        "jsonschema",
        "omgidl",
        "protobuf",
        "ros1msg",
        "ros2msg",
        "text",
    }
)
_LARGE_CHUNK_BYTES = 64 * 1024 * 1024
_SMALL_CHUNK_MEDIAN_BYTES = 64 * 1024
_SMALL_CHUNK_MIN_COUNT = 10


class Section(str, Enum):
    UNKNOWN = "unknown"
    DATA = "data"
    SUMMARY = "summary"
    SUMMARY_OFFSET = "summary-offset"
    FOOTER = "footer"
    AFTER_FOOTER = "after-footer"


class FindingCode(str, Enum):
    """Every doctor finding lives here so that codes stay typed, unique, and reviewable."""

    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,  # noqa: ARG004
        count: int,  # noqa: ARG004
        last_values: list[object],  # noqa: ARG004
    ) -> str:
        return name

    ATTACHMENT_CRC_MISMATCH = auto()
    ATTACHMENT_INDEX_BAD_OFFSET = auto()
    ATTACHMENT_INDEX_MISMATCH = auto()
    ATTACHMENT_MEDIA_TYPE_EMPTY = auto()
    BAD_FOOTER_SUMMARY_START = auto()
    BAD_LEADING_MAGIC = auto()
    BAD_SUMMARY_OFFSET_START = auto()
    BAD_TRAILING_MAGIC = auto()
    CHANNEL_TOPIC_DUPLICATE = auto()
    CHANNEL_TOPIC_EMPTY = auto()
    CHUNK_CRC_MISMATCH = auto()
    CHUNK_COMPRESSION_INEFFICIENT = auto()
    CHUNK_DECOMPRESSION_FAILED = auto()
    CHUNK_END_TIME_MISMATCH = auto()
    CHUNK_INDEX_BAD_OFFSET = auto()
    CHUNK_INDEX_MESSAGE_INDEX_LENGTH_MISMATCH = auto()
    CHUNK_INDEX_MESSAGE_INDEX_OFFSETS_MISMATCH = auto()
    CHUNK_INDEX_MISMATCH = auto()
    CHUNK_START_TIME_MISMATCH = auto()
    CHUNK_UNCOMPRESSED_SIZE_MISMATCH = auto()
    CONFLICTING_CHANNEL = auto()
    CONFLICTING_SCHEMA = auto()
    DATA_SECTION_CRC_MISMATCH = auto()
    DUPLICATE_CHANNEL = auto()
    DUPLICATE_ATTACHMENT_INDEX = auto()
    DUPLICATE_CHUNK_INDEX = auto()
    DUPLICATE_MAP_KEY = auto()
    DUPLICATE_METADATA_INDEX = auto()
    DUPLICATE_MESSAGE_INDEX = auto()
    DUPLICATE_MESSAGE_INDEX_OFFSET = auto()
    DUPLICATE_SCHEMA = auto()
    DUPLICATE_SUMMARY_OFFSET = auto()
    EMPTY_CHANNEL_MESSAGE_COUNTS = auto()
    EMPTY_CHUNK = auto()
    EMPTY_CHUNK_INDEX_MESSAGE_OFFSETS = auto()
    EMPTY_HEADER_LIBRARY = auto()
    EMPTY_MESSAGE_ENCODING = auto()
    EMPTY_MESSAGE_INDEX = auto()
    EMPTY_SCHEMA = auto()
    EXTRA_MESSAGE_INDEX = auto()
    FILE_TOO_SMALL = auto()
    FOOTER_NOT_LAST = auto()
    HEADER_NOT_FIRST = auto()
    ILLEGAL_DATA_RECORD = auto()
    ILLEGAL_RECORD_IN_CHUNK = auto()
    ILLEGAL_SUMMARY_OFFSET_RECORD = auto()
    ILLEGAL_SUMMARY_RECORD = auto()
    INDEXED_CHANNEL_MISSING_FROM_SUMMARY = auto()
    INDEXED_SCHEMA_MISSING_FROM_SUMMARY = auto()
    INDEX_SECTION_PRESENT_WITHOUT_SUMMARY = auto()
    INVALID_OPCODE_ZERO = auto()
    INVALID_QOS_PROFILE = auto()
    INVALID_UTF8 = auto()
    LARGE_CHUNK = auto()
    MESSAGE_BEFORE_CHANNEL = auto()
    MESSAGE_INDEX_BAD_OFFSET = auto()
    MESSAGE_INDEX_CHANNEL_MISMATCH = auto()
    MESSAGE_INDEX_OFFSETS_MISMATCH = auto()
    MESSAGE_INDEX_OFFSET_OUT_OF_RANGE = auto()
    MESSAGE_INDEX_TIME_MISMATCH = auto()
    MESSAGE_INDEX_WITHOUT_CHUNK = auto()
    MESSAGE_OUTSIDE_CHUNK_WITH_CHUNK_INDEX = auto()
    MESSAGE_PUBLISH_TIME_AFTER_LOG_TIME = auto()
    METADATA_INDEX_BAD_OFFSET = auto()
    METADATA_INDEX_MISMATCH = auto()
    METADATA_NAME_DUPLICATE = auto()
    MIXED_COMPRESSION_TYPES = auto()
    MISSING_ATTACHMENT_INDEX = auto()
    MISSING_ATTACHMENT_INDEXES = auto()
    MISSING_CHUNK_INDEX = auto()
    MISSING_CHUNK_INDEXES = auto()
    MISSING_DATA_END = auto()
    MISSING_FOOTER = auto()
    MISSING_MESSAGE_INDEX = auto()
    MISSING_MESSAGE_INDEXES = auto()
    MISSING_METADATA_INDEX = auto()
    MISSING_METADATA_INDEXES = auto()
    MISSING_STATISTICS = auto()
    MISSING_SUMMARY_OFFSET = auto()
    MISSING_SUMMARY_OFFSETS = auto()
    MULTIPLE_DATA_END = auto()
    MULTIPLE_FOOTERS = auto()
    MULTIPLE_HEADERS = auto()
    MULTIPLE_STATISTICS = auto()
    NO_ATTACHMENTS = auto()
    NO_COMPRESSION = auto()
    NO_METADATA = auto()
    NO_SCHEMAS = auto()
    NO_SUMMARY_SECTION = auto()
    NON_MONOTONIC_LOG_TIME = auto()
    RECORD_PARSE_ERROR = auto()
    RESERVED_SCHEMA_ID = auto()
    SCHEMA_ENCODING_UNKNOWN = auto()
    SCHEMA_NAME_EMPTY = auto()
    SCHEMA_DATA_WITH_EMPTY_ENCODING = auto()
    SMALL_CHUNKS = auto()
    STATISTICS_CHANNEL_COUNT_MISSING = auto()
    STATISTICS_CHANNEL_AFTER_STATISTICS = auto()
    STATISTICS_CHANNEL_COUNT_MISMATCH = auto()
    STATISTICS_CHANNEL_MISSING_FROM_SUMMARY = auto()
    STATISTICS_MISMATCH = auto()
    STATISTICS_UNKNOWN_CHANNEL = auto()
    SUMMARY_CRC_MISMATCH = auto()
    SUMMARY_CHANNEL_MISMATCH = auto()
    SUMMARY_CHANNEL_MISSING_IN_DATA = auto()
    SUMMARY_NOT_GROUPED = auto()
    SUMMARY_OFFSET_RANGE_MISMATCH = auto()
    SUMMARY_OFFSET_WITHOUT_GROUP = auto()
    SUMMARY_OFFSET_WITHOUT_SUMMARY = auto()
    SUMMARY_SCHEMA_MISMATCH = auto()
    SUMMARY_SCHEMA_MISSING_IN_DATA = auto()
    SUMMARY_RECORDS_WITH_ZERO_SUMMARY_START = auto()
    TRAILING_RECORD_BYTES = auto()
    TRUNCATED_CHUNK_RECORD_BODY = auto()
    TRUNCATED_CHUNK_RECORD_HEADER = auto()
    TRUNCATED_RECORD_BODY = auto()
    TRUNCATED_RECORD_HEADER = auto()
    UNKNOWN_PROFILE = auto()
    UNKNOWN_RESERVED_OPCODE = auto()
    UNKNOWN_SCHEMA_ID = auto()
    UNSORTED_MESSAGE_INDEX = auto()
    UNSUPPORTED_COMPRESSION = auto()
    ZERO_MESSAGE_SEQUENCE = auto()


_WARNING_CODES: frozenset[FindingCode] = frozenset(
    {
        FindingCode.ATTACHMENT_MEDIA_TYPE_EMPTY,
        FindingCode.CHANNEL_TOPIC_DUPLICATE,
        FindingCode.CHANNEL_TOPIC_EMPTY,
        FindingCode.CHUNK_COMPRESSION_INEFFICIENT,
        FindingCode.INVALID_QOS_PROFILE,
        FindingCode.DUPLICATE_CHANNEL,
        FindingCode.DUPLICATE_MAP_KEY,
        FindingCode.DUPLICATE_SCHEMA,
        FindingCode.EXTRA_MESSAGE_INDEX,
        FindingCode.INDEX_SECTION_PRESENT_WITHOUT_SUMMARY,
        FindingCode.MESSAGE_OUTSIDE_CHUNK_WITH_CHUNK_INDEX,
        FindingCode.MESSAGE_PUBLISH_TIME_AFTER_LOG_TIME,
        FindingCode.METADATA_NAME_DUPLICATE,
        FindingCode.SCHEMA_ENCODING_UNKNOWN,
        FindingCode.SCHEMA_NAME_EMPTY,
        FindingCode.UNKNOWN_RESERVED_OPCODE,
    }
)

_INFO_CODES: frozenset[FindingCode] = frozenset(
    {
        FindingCode.EMPTY_CHANNEL_MESSAGE_COUNTS,
        FindingCode.EMPTY_CHUNK,
        FindingCode.EMPTY_CHUNK_INDEX_MESSAGE_OFFSETS,
        FindingCode.EMPTY_HEADER_LIBRARY,
        FindingCode.EMPTY_MESSAGE_ENCODING,
        FindingCode.EMPTY_MESSAGE_INDEX,
        FindingCode.EMPTY_SCHEMA,
        FindingCode.LARGE_CHUNK,
        FindingCode.MISSING_ATTACHMENT_INDEXES,
        FindingCode.MISSING_CHUNK_INDEXES,
        FindingCode.MISSING_MESSAGE_INDEXES,
        FindingCode.MISSING_METADATA_INDEXES,
        FindingCode.MISSING_STATISTICS,
        FindingCode.MISSING_SUMMARY_OFFSET,
        FindingCode.MISSING_SUMMARY_OFFSETS,
        FindingCode.MIXED_COMPRESSION_TYPES,
        FindingCode.NO_ATTACHMENTS,
        FindingCode.NO_COMPRESSION,
        FindingCode.NO_METADATA,
        FindingCode.NO_SCHEMAS,
        FindingCode.NO_SUMMARY_SECTION,
        FindingCode.NON_MONOTONIC_LOG_TIME,
        FindingCode.SMALL_CHUNKS,
        FindingCode.TRAILING_RECORD_BYTES,
        FindingCode.UNKNOWN_PROFILE,
        FindingCode.UNSORTED_MESSAGE_INDEX,
        FindingCode.ZERO_MESSAGE_SEQUENCE,
    }
)


def default_severity(code: FindingCode) -> Severity:
    """Default severity for a finding code.

    Informational findings flag valid-but-notable structure. Warnings flag
    deterministic-decoding or indexed-read risks. Errors flag conditions where
    a spec-conformant reader can't read the file correctly.
    """
    if code in _INFO_CODES:
        return Severity.INFO
    return Severity.WARNING if code in _WARNING_CODES else Severity.ERROR


@dataclass(slots=True)
class Finding:
    severity: Severity
    code: FindingCode
    message: str
    offset: int | None = None
    section: Section = Section.UNKNOWN
    record: str = ""


@dataclass(slots=True)
class DecodedString:
    value: str
    raw: bytes
    offset: int
    end: int


@dataclass(slots=True)
class MapString:
    values: dict[str, str]
    end: int


@dataclass(slots=True)
class HeaderRecord:
    profile: str
    library: str


@dataclass(slots=True)
class FooterRecord:
    summary_start: int
    summary_offset_start: int
    summary_crc: int


@dataclass(slots=True)
class SchemaRecord:
    schema_id: int
    name: str
    encoding: str
    data: bytes


@dataclass(slots=True)
class ChannelRecord:
    channel_id: int
    schema_id: int
    topic: str
    message_encoding: str
    metadata: MapString


@dataclass(slots=True)
class MessageRecord:
    channel_id: int
    sequence: int
    log_time: int
    publish_time: int
    data_size: int


@dataclass(slots=True)
class ChunkInnerRecord:
    offset: int
    opcode: int
    length: int
    parsed: ParsedRecord | None
    record_name: str

    @property
    def end_offset(self) -> int:
        return self.offset + _RECORD_HEADER.size + self.length


@dataclass(slots=True)
class ChunkRecord:
    message_start_time: int
    message_end_time: int
    uncompressed_size: int
    uncompressed_crc: int
    compression: str
    compressed_size: int
    records: bytes
    inner_records: list[ChunkInnerRecord] = field(default_factory=list)
    messages_by_channel: dict[int, list[tuple[int, MessageRecord]]] = field(default_factory=dict)
    message_by_offset: dict[int, MessageRecord] = field(default_factory=dict)
    referenced_channels: set[int] = field(default_factory=set)


@dataclass(slots=True)
class MessageIndexRecord:
    channel_id: int
    timestamps: list[int]
    offsets: list[int]


@dataclass(slots=True)
class AttachmentRecord:
    log_time: int
    create_time: int
    name: str
    media_type: str
    data_size: int
    crc: int


@dataclass(slots=True)
class AttachmentIndexRecord:
    offset: int
    length: int
    log_time: int
    create_time: int
    data_size: int
    name: str
    media_type: str


@dataclass(slots=True)
class ChunkIndexRecord:
    message_start_time: int
    message_end_time: int
    chunk_start_offset: int
    chunk_length: int
    message_index_offsets: dict[int, int]
    message_index_length: int
    compression: str
    compressed_size: int
    uncompressed_size: int


@dataclass(slots=True)
class MetadataRecord:
    name: str
    metadata: MapString


@dataclass(slots=True)
class MetadataIndexRecord:
    offset: int
    length: int
    name: str


@dataclass(slots=True)
class StatisticsRecord:
    message_count: int
    schema_count: int
    channel_count: int
    attachment_count: int
    metadata_count: int
    chunk_count: int
    message_start_time: int
    message_end_time: int
    channel_message_counts: dict[int, int]


@dataclass(slots=True)
class SummaryOffsetRecord:
    group_opcode: int
    group_start: int
    group_length: int


@dataclass(slots=True)
class DataEndRecord:
    data_section_crc: int


@dataclass(slots=True)
class UnknownRecord:
    opcode: int


ParsedRecord = (
    HeaderRecord
    | FooterRecord
    | SchemaRecord
    | ChannelRecord
    | MessageRecord
    | ChunkRecord
    | MessageIndexRecord
    | ChunkIndexRecord
    | AttachmentRecord
    | AttachmentIndexRecord
    | MetadataRecord
    | MetadataIndexRecord
    | StatisticsRecord
    | SummaryOffsetRecord
    | DataEndRecord
    | UnknownRecord
)


@dataclass(slots=True)
class Frame:
    offset: int
    opcode: int
    length: int
    body: bytes
    parsed: ParsedRecord | None
    section: Section = Section.UNKNOWN

    @property
    def body_offset(self) -> int:
        return self.offset + _RECORD_HEADER.size

    @property
    def end_offset(self) -> int:
        return self.offset + _RECORD_HEADER.size + self.length

    @property
    def record_name(self) -> str:
        return opcode_name(self.opcode)


@dataclass(slots=True)
class ChunkIndexSequence:
    chunk: Frame
    indexes: list[Frame] = field(default_factory=list)


@dataclass(slots=True)
class DoctorReport:
    path: str
    findings: list[Finding]
    record_count: int
    message_count: int
    chunk_count: int

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity is Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity is Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity is Severity.INFO)


class ParseError(ValueError):
    pass


class McapDoctor:
    def __init__(
        self,
        *,
        strict_message_order: bool = False,
        severity_overrides: dict[FindingCode, Severity] | None = None,
    ) -> None:
        self.findings: list[Finding] = []
        self.frames: list[Frame] = []
        self._file_size = 0
        self._data_end_checksum: int | None = None
        self.strict_message_order = strict_message_order
        self._severity: dict[FindingCode, Severity] = {
            code: default_severity(code) for code in FindingCode
        }
        if strict_message_order:
            self._severity[FindingCode.NON_MONOTONIC_LOG_TIME] = Severity.ERROR
        if severity_overrides:
            self._severity.update(severity_overrides)

    def examine(self, stream: IO[bytes], size: int, path: str) -> DoctorReport:
        self.findings = []
        self.frames = []
        self._file_size = size
        self._data_end_checksum = None

        self._scan(stream)
        self._classify_sections()
        self._validate_file_structure(stream)
        self._validate_records()
        self._validate_summary_offsets()
        self._validate_indexes()
        counts = self._actual_counts()
        self._validate_statistics(counts)
        self._validate_advisory_findings(counts)

        return DoctorReport(
            path=path,
            findings=self.findings,
            record_count=len(self.frames),
            message_count=counts.message_count,
            chunk_count=counts.chunk_count,
        )

    def _emit(
        self,
        code: FindingCode,
        message: str,
        *,
        offset: int | None = None,
        section: Section = Section.UNKNOWN,
        record: str = "",
    ) -> None:
        severity = self._severity[code]
        self.findings.append(Finding(severity, code, message, offset, section, record))

    def _scan(self, stream: IO[bytes]) -> None:
        if self._file_size < MAGIC_SIZE * 2:
            self._emit(
                FindingCode.FILE_TOO_SMALL,
                "file is too small to contain leading and trailing MCAP magic bytes",
                offset=0,
            )
            return

        stream.seek(0)
        leading_magic = stream.read(MAGIC_SIZE)
        if leading_magic != MAGIC:
            self._emit(
                FindingCode.BAD_LEADING_MAGIC,
                "file does not begin with MCAP magic bytes",
                offset=0,
            )
            return

        trailing_magic = self._read_at(stream, self._file_size - MAGIC_SIZE, MAGIC_SIZE)
        has_trailing_magic = trailing_magic == MAGIC
        if not has_trailing_magic:
            self._emit(
                FindingCode.BAD_TRAILING_MAGIC,
                "file does not end with MCAP magic bytes",
                offset=max(self._file_size - MAGIC_SIZE, 0),
            )

        records_end = self._file_size - MAGIC_SIZE if has_trailing_magic else self._file_size
        stream.seek(MAGIC_SIZE)
        data_crc = zlib.crc32(MAGIC)
        is_before_data_end = True

        while stream.tell() < records_end:
            offset = stream.tell()
            header = stream.read(_RECORD_HEADER.size)
            if len(header) == 0:
                break
            if len(header) != _RECORD_HEADER.size:
                self._emit(
                    FindingCode.TRUNCATED_RECORD_HEADER,
                    "record header is truncated",
                    offset=offset,
                )
                break

            opcode, length = _RECORD_HEADER.unpack(header)
            end_offset = offset + _RECORD_HEADER.size + length
            if end_offset > records_end:
                self._emit(
                    FindingCode.TRUNCATED_RECORD_BODY,
                    f"{opcode_name(opcode)} record length extends past available bytes",
                    offset=offset,
                    record=opcode_name(opcode),
                )
                break

            body = stream.read(length)
            if len(body) != length:
                self._emit(
                    FindingCode.TRUNCATED_RECORD_BODY,
                    f"{opcode_name(opcode)} record body is truncated",
                    offset=offset,
                    record=opcode_name(opcode),
                )
                break

            checksum_before_frame = data_crc
            if is_before_data_end and opcode != Opcode.DATA_END:
                data_crc = zlib.crc32(header, data_crc)
                data_crc = zlib.crc32(body, data_crc)

            parsed = self._parse_record(opcode, body, offset, Section.UNKNOWN, opcode_name(opcode))
            frame = Frame(offset=offset, opcode=opcode, length=length, body=body, parsed=parsed)
            self.frames.append(frame)

            if opcode == Opcode.DATA_END and is_before_data_end:
                self._data_end_checksum = checksum_before_frame
                is_before_data_end = False

    def _read_at(self, stream: IO[bytes], offset: int, size: int) -> bytes:
        current = stream.tell()
        stream.seek(offset)
        data = stream.read(size)
        stream.seek(current)
        return data

    def _parse_record(
        self,
        opcode: int,
        body: bytes,
        record_offset: int,
        section: Section,
        record_name: str,
    ) -> ParsedRecord | None:
        if opcode == 0:
            self._emit(
                FindingCode.INVALID_OPCODE_ZERO,
                "opcode 0x00 is not valid",
                offset=record_offset,
                section=section,
                record=record_name,
            )
            return UnknownRecord(opcode)

        if is_private_opcode(opcode):
            return UnknownRecord(opcode)

        if opcode not in _KNOWN_OPCODES:
            self._emit(
                FindingCode.UNKNOWN_RESERVED_OPCODE,
                f"reserved opcode 0x{opcode:02x} is not known to this MCAP version",
                offset=record_offset,
                section=section,
                record=record_name,
            )
            return UnknownRecord(opcode)

        try:
            if opcode == Opcode.HEADER:
                return self._parse_header(body, record_offset, section, record_name)
            if opcode == Opcode.FOOTER:
                return self._parse_footer(body, record_offset, section, record_name)
            if opcode == Opcode.SCHEMA:
                return self._parse_schema(body, record_offset, section, record_name)
            if opcode == Opcode.CHANNEL:
                return self._parse_channel(body, record_offset, section, record_name)
            if opcode == Opcode.MESSAGE:
                return self._parse_message(body, record_offset, section, record_name)
            if opcode == Opcode.CHUNK:
                return self._parse_chunk(body, record_offset, section, record_name)
            if opcode == Opcode.MESSAGE_INDEX:
                return self._parse_message_index(body, record_offset, section, record_name)
            if opcode == Opcode.CHUNK_INDEX:
                return self._parse_chunk_index(body, record_offset, section, record_name)
            if opcode == Opcode.ATTACHMENT:
                return self._parse_attachment(body, record_offset, section, record_name)
            if opcode == Opcode.ATTACHMENT_INDEX:
                return self._parse_attachment_index(body, record_offset, section, record_name)
            if opcode == Opcode.STATISTICS:
                return self._parse_statistics(body, record_offset, section, record_name)
            if opcode == Opcode.METADATA:
                return self._parse_metadata(body, record_offset, section, record_name)
            if opcode == Opcode.METADATA_INDEX:
                return self._parse_metadata_index(body, record_offset, section, record_name)
            if opcode == Opcode.SUMMARY_OFFSET:
                return self._parse_summary_offset(body, record_offset, section, record_name)
            if opcode == Opcode.DATA_END:
                return self._parse_data_end(body, record_offset, section, record_name)
        except ParseError as exc:
            self._emit(
                FindingCode.RECORD_PARSE_ERROR,
                str(exc),
                offset=record_offset,
                section=section,
                record=record_name,
            )
            return None

        return UnknownRecord(opcode)

    def _parse_header(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> HeaderRecord:
        profile = self._read_string(body, 0, record_offset, section, record_name, "profile")
        library = self._read_string(
            body, profile.end, record_offset, section, record_name, "library"
        )
        self._warn_trailing(body, library.end, record_offset, section, record_name)
        return HeaderRecord(profile.value, library.value)

    def _parse_footer(
        self, body: bytes, _record_offset: int, _section: Section, _record_name: str
    ) -> FooterRecord:
        if len(body) != 20:
            raise ParseError(f"Footer record length must be 20 bytes, got {len(body)}")
        summary_start, summary_offset_start = struct.unpack_from("<QQ", body, 0)
        summary_crc = _U32.unpack_from(body, 16)[0]
        return FooterRecord(summary_start, summary_offset_start, summary_crc)

    def _parse_schema(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> SchemaRecord:
        self._require(body, 0, 2, "Schema.id")
        schema_id = _U16.unpack_from(body, 0)[0]
        name = self._read_string(body, 2, record_offset, section, record_name, "name")
        encoding = self._read_string(
            body, name.end, record_offset, section, record_name, "encoding"
        )
        self._require(body, encoding.end, 4, "Schema.data length")
        data_len = _U32.unpack_from(body, encoding.end)[0]
        data_offset = encoding.end + 4
        self._require(body, data_offset, data_len, "Schema.data")
        data = body[data_offset : data_offset + data_len]
        end = data_offset + data_len
        self._warn_trailing(body, end, record_offset, section, record_name)
        return SchemaRecord(schema_id, name.value, encoding.value, data)

    def _parse_channel(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> ChannelRecord:
        self._require(body, 0, 4, "Channel.id and schema_id")
        channel_id, schema_id = struct.unpack_from("<HH", body, 0)
        topic = self._read_string(body, 4, record_offset, section, record_name, "topic")
        encoding = self._read_string(
            body, topic.end, record_offset, section, record_name, "message_encoding"
        )
        metadata = self._read_map_string(
            body,
            encoding.end,
            record_offset,
            section,
            record_name,
            "metadata",
        )
        self._warn_trailing(body, metadata.end, record_offset, section, record_name)
        return ChannelRecord(channel_id, schema_id, topic.value, encoding.value, metadata)

    def _parse_message(
        self, body: bytes, _record_offset: int, _section: Section, _record_name: str
    ) -> MessageRecord:
        if len(body) < _MESSAGE_HEADER.size:
            raise ParseError(
                f"Message record is too short: got {len(body)} bytes, "
                f"need at least {_MESSAGE_HEADER.size}"
            )
        channel_id, sequence, log_time, publish_time = _MESSAGE_HEADER.unpack_from(body, 0)
        return MessageRecord(
            channel_id, sequence, log_time, publish_time, len(body) - _MESSAGE_HEADER.size
        )

    def _parse_chunk(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> ChunkRecord:
        self._require(body, 0, _CHUNK_HEADER.size, "Chunk fixed fields")
        start_time, end_time, uncompressed_size, uncompressed_crc = _CHUNK_HEADER.unpack_from(
            body, 0
        )
        compression = self._read_string(
            body, _CHUNK_HEADER.size, record_offset, section, record_name, "compression"
        )
        self._require(body, compression.end, 8, "Chunk.records length")
        records_len = _U64.unpack_from(body, compression.end)[0]
        records_offset = compression.end + 8
        self._require(body, records_offset, records_len, "Chunk.records")
        records = body[records_offset : records_offset + records_len]
        end = records_offset + records_len
        self._warn_trailing(body, end, record_offset, section, record_name)
        chunk = ChunkRecord(
            start_time,
            end_time,
            uncompressed_size,
            uncompressed_crc,
            compression.value,
            len(records),
            records,
        )
        self._validate_chunk_payload(chunk, record_offset, section, record_name)
        return chunk

    def _parse_message_index(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> MessageIndexRecord:
        self._require(body, 0, 6, "MessageIndex header")
        channel_id, records_len = struct.unpack_from("<HI", body, 0)
        self._require(body, 6, records_len, "MessageIndex.records")
        if records_len % 16 != 0:
            raise ParseError("MessageIndex.records length must be a multiple of 16 bytes")
        timestamps: list[int] = []
        offsets: list[int] = []
        for timestamp, offset in struct.iter_unpack("<QQ", body[6 : 6 + records_len]):
            timestamps.append(timestamp)
            offsets.append(offset)
        self._warn_trailing(body, 6 + records_len, record_offset, section, record_name)
        return MessageIndexRecord(channel_id, timestamps, offsets)

    def _parse_attachment(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> AttachmentRecord:
        self._require(body, 0, 16, "Attachment times")
        log_time, create_time = struct.unpack_from("<QQ", body, 0)
        name = self._read_string(body, 16, record_offset, section, record_name, "name")
        media_type = self._read_string(
            body, name.end, record_offset, section, record_name, "media_type"
        )
        self._require(body, media_type.end, 8, "Attachment.data length")
        data_size = _U64.unpack_from(body, media_type.end)[0]
        data_offset = media_type.end + 8
        self._require(body, data_offset, data_size, "Attachment.data")
        crc_offset = data_offset + data_size
        self._require(body, crc_offset, 4, "Attachment.crc")
        crc = _U32.unpack_from(body, crc_offset)[0]
        if crc != 0:
            calculated = zlib.crc32(body[:crc_offset])
            if calculated != crc:
                self._emit(
                    FindingCode.ATTACHMENT_CRC_MISMATCH,
                    f"Attachment.crc is 0x{crc:08x}, calculated 0x{calculated:08x}",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
        self._warn_trailing(body, crc_offset + 4, record_offset, section, record_name)
        return AttachmentRecord(log_time, create_time, name.value, media_type.value, data_size, crc)

    def _parse_attachment_index(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> AttachmentIndexRecord:
        self._require(body, 0, _ATTACHMENT_INDEX_FIXED.size, "AttachmentIndex fixed fields")
        offset, length, log_time, create_time, data_size = _ATTACHMENT_INDEX_FIXED.unpack_from(
            body, 0
        )
        name = self._read_string(
            body, _ATTACHMENT_INDEX_FIXED.size, record_offset, section, record_name, "name"
        )
        media_type = self._read_string(
            body, name.end, record_offset, section, record_name, "media_type"
        )
        self._warn_trailing(body, media_type.end, record_offset, section, record_name)
        return AttachmentIndexRecord(
            offset, length, log_time, create_time, data_size, name.value, media_type.value
        )

    def _parse_chunk_index(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> ChunkIndexRecord:
        self._require(body, 0, _CHUNK_INDEX_FIXED.size, "ChunkIndex fixed fields")
        (
            message_start_time,
            message_end_time,
            chunk_start_offset,
            chunk_length,
            offsets_len,
        ) = _CHUNK_INDEX_FIXED.unpack_from(body, 0)
        offsets_start = _CHUNK_INDEX_FIXED.size
        self._require(body, offsets_start, offsets_len, "ChunkIndex.message_index_offsets")
        if offsets_len % _INDEX_OFFSET_ENTRY.size != 0:
            raise ParseError("ChunkIndex.message_index_offsets length must be a multiple of 10")
        message_index_offsets: dict[int, int] = {}
        seen_channels: set[int] = set()
        for channel_id, offset in struct.iter_unpack(
            "<HQ", body[offsets_start : offsets_start + offsets_len]
        ):
            if channel_id in seen_channels:
                self._emit(
                    FindingCode.DUPLICATE_MAP_KEY,
                    f"ChunkIndex.message_index_offsets contains duplicate channel {channel_id}",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
            seen_channels.add(channel_id)
            message_index_offsets[channel_id] = offset

        cursor = offsets_start + offsets_len
        self._require(body, cursor, 8, "ChunkIndex.message_index_length")
        message_index_length = _U64.unpack_from(body, cursor)[0]
        compression = self._read_string(
            body, cursor + 8, record_offset, section, record_name, "compression"
        )
        self._require(body, compression.end, 16, "ChunkIndex sizes")
        compressed_size, uncompressed_size = struct.unpack_from("<QQ", body, compression.end)
        self._warn_trailing(body, compression.end + 16, record_offset, section, record_name)
        return ChunkIndexRecord(
            message_start_time,
            message_end_time,
            chunk_start_offset,
            chunk_length,
            message_index_offsets,
            message_index_length,
            compression.value,
            compressed_size,
            uncompressed_size,
        )

    def _parse_statistics(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> StatisticsRecord:
        self._require(body, 0, _STATISTICS_FIXED.size, "Statistics fixed fields")
        (
            message_count,
            schema_count,
            channel_count,
            attachment_count,
            metadata_count,
            chunk_count,
            message_start_time,
            message_end_time,
            counts_len,
        ) = _STATISTICS_FIXED.unpack_from(body, 0)
        self._require(body, _STATISTICS_FIXED.size, counts_len, "Statistics.channel_message_counts")
        if counts_len % _INDEX_OFFSET_ENTRY.size != 0:
            raise ParseError("Statistics.channel_message_counts length must be a multiple of 10")
        counts: dict[int, int] = {}
        seen: set[int] = set()
        for channel_id, count in struct.iter_unpack(
            "<HQ", body[_STATISTICS_FIXED.size : _STATISTICS_FIXED.size + counts_len]
        ):
            if channel_id in seen:
                self._emit(
                    FindingCode.DUPLICATE_MAP_KEY,
                    f"Statistics.channel_message_counts contains duplicate channel {channel_id}",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
            seen.add(channel_id)
            counts[channel_id] = count
        self._warn_trailing(
            body,
            _STATISTICS_FIXED.size + counts_len,
            record_offset,
            section,
            record_name,
        )
        return StatisticsRecord(
            message_count,
            schema_count,
            channel_count,
            attachment_count,
            metadata_count,
            chunk_count,
            message_start_time,
            message_end_time,
            counts,
        )

    def _parse_metadata(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> MetadataRecord:
        name = self._read_string(body, 0, record_offset, section, record_name, "name")
        metadata = self._read_map_string(
            body,
            name.end,
            record_offset,
            section,
            record_name,
            "metadata",
        )
        self._warn_trailing(body, metadata.end, record_offset, section, record_name)
        return MetadataRecord(name.value, metadata)

    def _parse_metadata_index(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> MetadataIndexRecord:
        self._require(body, 0, 16, "MetadataIndex fixed fields")
        offset, length = struct.unpack_from("<QQ", body, 0)
        name = self._read_string(body, 16, record_offset, section, record_name, "name")
        self._warn_trailing(body, name.end, record_offset, section, record_name)
        return MetadataIndexRecord(offset, length, name.value)

    def _parse_summary_offset(
        self, body: bytes, record_offset: int, section: Section, record_name: str
    ) -> SummaryOffsetRecord:
        if len(body) < _SUMMARY_OFFSET.size:
            raise ParseError(
                f"SummaryOffset record length must be at least {_SUMMARY_OFFSET.size}, "
                f"got {len(body)}"
            )
        group_opcode, group_start, group_length = _SUMMARY_OFFSET.unpack_from(body, 0)
        self._warn_trailing(body, _SUMMARY_OFFSET.size, record_offset, section, record_name)
        return SummaryOffsetRecord(group_opcode, group_start, group_length)

    def _parse_data_end(
        self, body: bytes, _record_offset: int, _section: Section, _record_name: str
    ) -> DataEndRecord:
        if len(body) != 4:
            raise ParseError(f"DataEnd record length must be 4 bytes, got {len(body)}")
        return DataEndRecord(_U32.unpack_from(body, 0)[0])

    def _require(self, body: bytes, offset: int, size: int, field_name: str) -> None:
        if offset + size > len(body):
            raise ParseError(f"{field_name} extends past the end of the record")

    def _read_string(
        self,
        body: bytes,
        offset: int,
        record_offset: int,
        section: Section,
        record_name: str,
        field_name: str,
    ) -> DecodedString:
        self._require(body, offset, 4, f"{record_name}.{field_name} length")
        length = _U32.unpack_from(body, offset)[0]
        data_offset = offset + 4
        self._require(body, data_offset, length, f"{record_name}.{field_name}")
        raw = body[data_offset : data_offset + length]
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError:
            value = raw.decode("utf-8", "replace")
            self._emit(
                FindingCode.INVALID_UTF8,
                f"{record_name}.{field_name} is not valid UTF-8",
                offset=record_offset,
                section=section,
                record=record_name,
            )
        return DecodedString(value, raw, offset, data_offset + length)

    def _read_map_string(
        self,
        body: bytes,
        offset: int,
        record_offset: int,
        section: Section,
        record_name: str,
        field_name: str,
    ) -> MapString:
        self._require(body, offset, 4, f"{record_name}.{field_name} length")
        length = _U32.unpack_from(body, offset)[0]
        cursor = offset + 4
        end = cursor + length
        self._require(body, cursor, length, f"{record_name}.{field_name}")
        values: dict[str, str] = {}
        seen_keys: set[bytes] = set()
        while cursor < end:
            key = self._read_string(
                body, cursor, record_offset, section, record_name, f"{field_name}.key"
            )
            cursor = key.end
            value = self._read_string(
                body, cursor, record_offset, section, record_name, f"{field_name}.value"
            )
            cursor = value.end
            if cursor > end:
                raise ParseError(f"{record_name}.{field_name} entry extends past map length")
            if key.raw in seen_keys:
                self._emit(
                    FindingCode.DUPLICATE_MAP_KEY,
                    f"{record_name}.{field_name} contains duplicate key {key.value!r}",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
            seen_keys.add(key.raw)
            values[key.value] = value.value
        if cursor != end:
            raise ParseError(f"{record_name}.{field_name} did not end on an entry boundary")
        return MapString(values, end)

    def _warn_trailing(
        self,
        _body: bytes,
        _end: int,
        _record_offset: int,
        _section: Section,
        _record_name: str,
    ) -> None:
        if _end < len(_body):
            self._emit(
                FindingCode.TRAILING_RECORD_BYTES,
                f"{_record_name} contains {len(_body) - _end} trailing bytes "
                "ignored by this version",
                offset=_record_offset,
                section=_section,
                record=_record_name,
            )

    def _validate_chunk_payload(
        self,
        chunk: ChunkRecord,
        record_offset: int,
        section: Section,
        record_name: str,
    ) -> None:
        if chunk.compression == "":
            uncompressed = chunk.records
        elif chunk.compression == "zstd":
            if _zstd_decompress is None:
                self._emit(
                    FindingCode.CHUNK_DECOMPRESSION_FAILED,
                    "zstd compression used but neither compression.zstd nor zstandard is installed",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
                return
            try:
                uncompressed = _zstd_decompress(chunk.records, chunk.uncompressed_size)
            except _ZstdError as exc:
                self._emit(
                    FindingCode.CHUNK_DECOMPRESSION_FAILED,
                    f"zstd decompression failed: {exc}",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
                return
        elif chunk.compression == "lz4":
            try:
                uncompressed = lz4_decompress(chunk.records)
            except RuntimeError as exc:
                self._emit(
                    FindingCode.CHUNK_DECOMPRESSION_FAILED,
                    f"lz4 decompression failed: {exc}",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
                return
        else:
            self._emit(
                FindingCode.UNSUPPORTED_COMPRESSION,
                f"unsupported chunk compression {chunk.compression!r}",
                offset=record_offset,
                section=section,
                record=record_name,
            )
            return

        if len(uncompressed) != chunk.uncompressed_size:
            self._emit(
                FindingCode.CHUNK_UNCOMPRESSED_SIZE_MISMATCH,
                "Chunk.uncompressed_size does not match decompressed records size",
                offset=record_offset,
                section=section,
                record=record_name,
            )
        if chunk.uncompressed_crc != 0:
            calculated = zlib.crc32(uncompressed)
            if calculated != chunk.uncompressed_crc:
                self._emit(
                    FindingCode.CHUNK_CRC_MISMATCH,
                    f"Chunk.uncompressed_crc is 0x{chunk.uncompressed_crc:08x}, "
                    f"calculated 0x{calculated:08x}",
                    offset=record_offset,
                    section=section,
                    record=record_name,
                )
        self._scan_chunk_records(chunk, record_offset, uncompressed)
        chunk.records = b""

    def _scan_chunk_records(self, chunk: ChunkRecord, chunk_file_offset: int, data: bytes) -> None:
        cursor = 0
        while cursor < len(data):
            record_offset = cursor
            if cursor + _RECORD_HEADER.size > len(data):
                self._emit(
                    FindingCode.TRUNCATED_CHUNK_RECORD_HEADER,
                    "record header inside chunk is truncated",
                    offset=chunk_file_offset,
                    record="Chunk",
                )
                return
            opcode, length = _RECORD_HEADER.unpack_from(data, cursor)
            cursor += _RECORD_HEADER.size
            if cursor + length > len(data):
                self._emit(
                    FindingCode.TRUNCATED_CHUNK_RECORD_BODY,
                    f"{opcode_name(opcode)} record inside chunk is truncated",
                    offset=chunk_file_offset,
                    record="Chunk",
                )
                return
            body = data[cursor : cursor + length]
            cursor += length

            inner_record_name = f"Chunk.{opcode_name(opcode)}"
            parsed = self._parse_record(
                opcode,
                body,
                chunk_file_offset,
                Section.DATA,
                inner_record_name,
            )
            inner = ChunkInnerRecord(record_offset, opcode, length, parsed, opcode_name(opcode))
            chunk.inner_records.append(inner)
            if isinstance(parsed, MessageRecord):
                chunk.referenced_channels.add(parsed.channel_id)
                chunk.message_by_offset[record_offset] = parsed
                chunk.messages_by_channel.setdefault(parsed.channel_id, []).append(
                    (record_offset, parsed)
                )
            elif opcode in _KNOWN_OPCODES and opcode not in {
                Opcode.SCHEMA,
                Opcode.CHANNEL,
                Opcode.MESSAGE,
            }:
                self._emit(
                    FindingCode.ILLEGAL_RECORD_IN_CHUNK,
                    f"{opcode_name(opcode)} records must not appear inside Chunk.records",
                    offset=chunk_file_offset,
                    record="Chunk",
                )

    def _classify_sections(self) -> None:
        footer_frame = self._single_footer()
        footer = (
            footer_frame.parsed
            if footer_frame and isinstance(footer_frame.parsed, FooterRecord)
            else None
        )
        data_end_frame = self._first_frame(Opcode.DATA_END)

        for frame in self.frames:
            if frame.opcode == Opcode.FOOTER:
                frame.section = Section.FOOTER
            elif footer_frame and frame.offset > footer_frame.offset:
                frame.section = Section.AFTER_FOOTER
            elif (
                footer is not None
                and footer.summary_offset_start
                and frame.offset >= footer.summary_offset_start
            ):
                frame.section = Section.SUMMARY_OFFSET
            elif (
                footer is not None and footer.summary_start and frame.offset >= footer.summary_start
            ) or (data_end_frame is not None and frame.offset > data_end_frame.offset):
                frame.section = Section.SUMMARY
            else:
                frame.section = Section.DATA

    def _validate_file_structure(self, stream: IO[bytes]) -> None:
        if not self.frames:
            return

        first = self.frames[0]
        if first.opcode != Opcode.HEADER:
            self._emit(
                FindingCode.HEADER_NOT_FIRST,
                "first record after leading magic must be Header",
                offset=first.offset,
                section=first.section,
                record=first.record_name,
            )

        header_count = sum(1 for frame in self.frames if frame.opcode == Opcode.HEADER)
        if header_count > 1:
            self._emit(
                FindingCode.MULTIPLE_HEADERS,
                "file contains multiple Header records",
                offset=first.offset,
            )

        footer_frames = [frame for frame in self.frames if frame.opcode == Opcode.FOOTER]
        if not footer_frames:
            self._emit(FindingCode.MISSING_FOOTER, "file does not contain a Footer record")
        elif len(footer_frames) > 1:
            self._emit(
                FindingCode.MULTIPLE_FOOTERS,
                "file contains multiple Footer records",
                offset=footer_frames[1].offset,
                section=footer_frames[1].section,
                record="Footer",
            )
        else:
            footer_frame = footer_frames[0]
            expected_end = self._file_size - MAGIC_SIZE
            if footer_frame.end_offset != expected_end:
                self._emit(
                    FindingCode.FOOTER_NOT_LAST,
                    "Footer must be the last record before trailing magic",
                    offset=footer_frame.offset,
                    section=footer_frame.section,
                    record="Footer",
                )

        data_end_frames = [frame for frame in self.frames if frame.opcode == Opcode.DATA_END]
        if not data_end_frames:
            self._emit(FindingCode.MISSING_DATA_END, "file does not contain a DataEnd record")
        elif len(data_end_frames) > 1:
            self._emit(
                FindingCode.MULTIPLE_DATA_END,
                "file contains multiple DataEnd records",
                offset=data_end_frames[1].offset,
                section=data_end_frames[1].section,
                record="DataEnd",
            )

        data_end_frame = data_end_frames[0] if data_end_frames else None
        data_end = (
            data_end_frame.parsed
            if data_end_frame is not None and isinstance(data_end_frame.parsed, DataEndRecord)
            else None
        )
        if data_end_frame is not None and data_end is not None and data_end.data_section_crc != 0:
            calculated = self._data_end_checksum
            if calculated is not None and data_end.data_section_crc != calculated:
                self._emit(
                    FindingCode.DATA_SECTION_CRC_MISMATCH,
                    f"DataEnd.data_section_crc is 0x{data_end.data_section_crc:08x}, "
                    f"calculated 0x{calculated:08x}",
                    offset=data_end_frame.offset,
                    section=data_end_frame.section,
                    record="DataEnd",
                )

        footer_frame = footer_frames[0] if len(footer_frames) == 1 else None
        footer = (
            footer_frame.parsed
            if footer_frame is not None and isinstance(footer_frame.parsed, FooterRecord)
            else None
        )
        if footer_frame is not None and footer is not None:
            self._validate_footer_offsets(stream, footer_frame, footer, data_end_frame)

        self._validate_section_allowed_records()

    def _validate_footer_offsets(
        self,
        stream: IO[bytes],
        footer_frame: Frame,
        footer: FooterRecord,
        data_end_frame: Frame | None,
    ) -> None:
        if footer.summary_start:
            if footer.summary_start >= footer_frame.offset:
                self._emit(
                    FindingCode.BAD_FOOTER_SUMMARY_START,
                    "Footer.summary_start must point before the Footer record",
                    offset=footer_frame.offset,
                    section=Section.FOOTER,
                    record="Footer",
                )
            if data_end_frame is not None and footer.summary_start != data_end_frame.end_offset:
                self._emit(
                    FindingCode.BAD_FOOTER_SUMMARY_START,
                    "Footer.summary_start must equal the first byte after DataEnd",
                    offset=footer_frame.offset,
                    section=Section.FOOTER,
                    record="Footer",
                )
        elif data_end_frame is not None and data_end_frame.end_offset != footer_frame.offset:
            self._emit(
                FindingCode.SUMMARY_RECORDS_WITH_ZERO_SUMMARY_START,
                "records exist between DataEnd and Footer but Footer.summary_start is zero",
                offset=footer_frame.offset,
                section=Section.FOOTER,
                record="Footer",
            )

        if footer.summary_offset_start:
            if footer.summary_start == 0:
                self._emit(
                    FindingCode.SUMMARY_OFFSET_WITHOUT_SUMMARY,
                    "Footer.summary_offset_start is nonzero while summary_start is zero",
                    offset=footer_frame.offset,
                    section=Section.FOOTER,
                    record="Footer",
                )
            if footer.summary_offset_start < footer.summary_start:
                self._emit(
                    FindingCode.BAD_SUMMARY_OFFSET_START,
                    "Footer.summary_offset_start must not be before summary_start",
                    offset=footer_frame.offset,
                    section=Section.FOOTER,
                    record="Footer",
                )
            if footer.summary_offset_start >= footer_frame.offset:
                self._emit(
                    FindingCode.BAD_SUMMARY_OFFSET_START,
                    "Footer.summary_offset_start must point before the Footer record",
                    offset=footer_frame.offset,
                    section=Section.FOOTER,
                    record="Footer",
                )

        if footer.summary_crc != 0:
            start = footer.summary_start or footer_frame.offset
            end = footer_frame.body_offset + 16
            if start <= end <= self._file_size:
                data = self._read_at(stream, start, end - start)
                calculated = zlib.crc32(data)
                if calculated != footer.summary_crc:
                    self._emit(
                        FindingCode.SUMMARY_CRC_MISMATCH,
                        f"Footer.summary_crc is 0x{footer.summary_crc:08x}, "
                        f"calculated 0x{calculated:08x}",
                        offset=footer_frame.offset,
                        section=Section.FOOTER,
                        record="Footer",
                    )

    def _validate_section_allowed_records(self) -> None:
        for frame in self.frames:
            if (
                frame.opcode in {Opcode.HEADER, Opcode.FOOTER}
                or is_private_opcode(frame.opcode)
                or frame.opcode not in _KNOWN_OPCODES
            ):
                continue
            if frame.section == Section.DATA and frame.opcode not in _DATA_OPCODES:
                self._emit(
                    FindingCode.ILLEGAL_DATA_RECORD,
                    f"{frame.record_name} records are not allowed in the data section",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            elif frame.section == Section.SUMMARY and frame.opcode not in _SUMMARY_OPCODES:
                self._emit(
                    FindingCode.ILLEGAL_SUMMARY_RECORD,
                    f"{frame.record_name} records are not allowed in the summary section",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            elif frame.section == Section.SUMMARY_OFFSET and frame.opcode != Opcode.SUMMARY_OFFSET:
                self._emit(
                    FindingCode.ILLEGAL_SUMMARY_OFFSET_RECORD,
                    f"{frame.record_name} records are not allowed in the summary offset section",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )

    def _validate_records(self) -> None:
        data_schemas: dict[int, SchemaRecord] = {}
        data_channels: dict[int, ChannelRecord] = {}
        data_channel_shapes: dict[tuple[str, int, str], int] = {}
        metadata_names: dict[str, int] = {}
        max_log_time = 0
        has_message = False

        for frame in self.frames:
            parsed = frame.parsed
            if isinstance(parsed, HeaderRecord):
                self._validate_header(frame, parsed)
            elif isinstance(parsed, SchemaRecord):
                self._validate_schema(frame, parsed, data_schemas)
            elif isinstance(parsed, ChannelRecord):
                self._validate_channel(
                    frame,
                    parsed,
                    data_schemas,
                    data_channels,
                    data_channel_shapes,
                )
            elif isinstance(parsed, MessageRecord):
                max_log_time, has_message = self._validate_message(
                    frame, parsed, data_channels, max_log_time, has_message
                )
            elif isinstance(parsed, ChunkRecord):
                self._validate_chunk(
                    frame,
                    parsed,
                    data_schemas,
                    data_channels,
                    data_channel_shapes,
                )
            elif isinstance(parsed, MessageIndexRecord):
                self._validate_message_index(frame, parsed)
            elif isinstance(parsed, AttachmentRecord):
                self._validate_attachment(frame, parsed)
            elif isinstance(parsed, MetadataRecord):
                self._validate_metadata(frame, parsed, metadata_names)
            elif isinstance(parsed, StatisticsRecord) and parsed.channel_message_counts == {}:
                self._emit(
                    FindingCode.EMPTY_CHANNEL_MESSAGE_COUNTS,
                    "Statistics.channel_message_counts is empty; "
                    "per-channel counts are unavailable",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )

    def _validate_header(self, frame: Frame, header: HeaderRecord) -> None:
        if header.library == "":
            self._emit(
                FindingCode.EMPTY_HEADER_LIBRARY,
                "Header.library should identify the software that produced the file",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        if header.profile not in {"", "ros1", "ros2"}:
            self._emit(
                FindingCode.UNKNOWN_PROFILE,
                f"Header.profile {header.profile!r} is not a well-known profile",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )

    def _validate_schema(
        self,
        frame: Frame,
        schema: SchemaRecord,
        data_schemas: dict[int, SchemaRecord],
    ) -> None:
        if schema.schema_id == 0:
            self._emit(
                FindingCode.RESERVED_SCHEMA_ID,
                "Schema.id 0 is reserved and must not be used",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        if schema.encoding == "" and schema.data:
            self._emit(
                FindingCode.SCHEMA_DATA_WITH_EMPTY_ENCODING,
                "Schema.data must be empty when Schema.encoding is empty",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        elif schema.encoding == "" and not schema.data:
            self._emit(
                FindingCode.EMPTY_SCHEMA,
                "Schema has empty encoding and empty data",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )

        if frame.section == Section.DATA:
            self._validate_schema_advisories(frame, schema)
            previous = data_schemas.get(schema.schema_id)
            if previous is not None:
                if same_schema(previous, schema):
                    self._emit(
                        FindingCode.DUPLICATE_SCHEMA,
                        f"duplicate Schema record with id {schema.schema_id}",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                else:
                    self._emit(
                        FindingCode.CONFLICTING_SCHEMA,
                        f"Schema id {schema.schema_id} is reused with different content",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
            data_schemas[schema.schema_id] = schema

    def _validate_channel(
        self,
        frame: Frame,
        channel: ChannelRecord,
        data_schemas: dict[int, SchemaRecord],
        data_channels: dict[int, ChannelRecord],
        data_channel_shapes: dict[tuple[str, int, str], int],
    ) -> None:
        if channel.message_encoding == "":
            self._emit(
                FindingCode.EMPTY_MESSAGE_ENCODING,
                "Channel.message_encoding is empty",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        if (
            channel.schema_id != 0
            and channel.schema_id not in data_schemas
            and frame.section == Section.DATA
        ):
            self._emit(
                FindingCode.UNKNOWN_SCHEMA_ID,
                f"Channel {channel.channel_id} references unknown Schema {channel.schema_id}",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        for issue in _qos_issues(channel.metadata.values):
            self._emit(
                FindingCode.INVALID_QOS_PROFILE,
                f"Channel {channel.channel_id} ({channel.topic}): {issue}",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        if frame.section == Section.DATA:
            self._validate_channel_advisories(frame, channel, data_channel_shapes)
            previous = data_channels.get(channel.channel_id)
            if previous is not None:
                if same_channel(previous, channel):
                    self._emit(
                        FindingCode.DUPLICATE_CHANNEL,
                        f"duplicate Channel record with id {channel.channel_id}",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                else:
                    self._emit(
                        FindingCode.CONFLICTING_CHANNEL,
                        f"Channel id {channel.channel_id} is reused with different content",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
            data_channels[channel.channel_id] = channel

    def _validate_message(
        self,
        frame: Frame,
        message: MessageRecord,
        data_channels: dict[int, ChannelRecord],
        max_log_time: int,
        has_message: bool,
    ) -> tuple[int, bool]:
        channel = data_channels.get(message.channel_id)
        if frame.section == Section.DATA and channel is None:
            self._emit(
                FindingCode.MESSAGE_BEFORE_CHANNEL,
                f"Message references channel {message.channel_id} before its Channel record",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        if has_message and message.log_time < max_log_time:
            self._message_order_finding(
                f"Message.log_time {message.log_time} is less than latest log time {max_log_time}",
                frame.offset,
                frame.section,
                frame.record_name,
            )
        if message.publish_time > message.log_time:
            self._emit(
                FindingCode.MESSAGE_PUBLISH_TIME_AFTER_LOG_TIME,
                "Message.publish_time is later than Message.log_time",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        return max(max_log_time, message.log_time), True

    def _validate_chunk(
        self,
        frame: Frame,
        chunk: ChunkRecord,
        data_schemas: dict[int, SchemaRecord],
        data_channels: dict[int, ChannelRecord],
        data_channel_shapes: dict[tuple[str, int, str], int],
    ) -> None:
        inner_schemas = dict(data_schemas)
        inner_channels = dict(data_channels)
        chunk_min_time: int | None = None
        chunk_max_time: int | None = None
        chunk_max_log_time = 0
        chunk_has_message = False
        chunk_message_count = 0

        self._validate_chunk_compression_advisories(frame, chunk)

        for inner in chunk.inner_records:
            parsed = inner.parsed
            if isinstance(parsed, SchemaRecord):
                if parsed.schema_id == 0:
                    self._emit(
                        FindingCode.RESERVED_SCHEMA_ID,
                        "Schema.id 0 is reserved and must not be used",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                if parsed.encoding == "" and parsed.data:
                    self._emit(
                        FindingCode.SCHEMA_DATA_WITH_EMPTY_ENCODING,
                        "Schema.data must be empty when Schema.encoding is empty",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                elif parsed.encoding == "" and not parsed.data:
                    self._emit(
                        FindingCode.EMPTY_SCHEMA,
                        "Schema has empty encoding and empty data",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                self._validate_schema_advisories(frame, parsed)
                previous = inner_schemas.get(parsed.schema_id)
                if previous is not None and not same_schema(previous, parsed):
                    self._emit(
                        FindingCode.CONFLICTING_SCHEMA,
                        f"Schema id {parsed.schema_id} is reused with different content in a chunk",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                inner_schemas[parsed.schema_id] = parsed
                data_schemas[parsed.schema_id] = parsed
            elif isinstance(parsed, ChannelRecord):
                if parsed.message_encoding == "":
                    self._emit(
                        FindingCode.EMPTY_MESSAGE_ENCODING,
                        "Channel.message_encoding is empty",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                if parsed.schema_id != 0 and parsed.schema_id not in inner_schemas:
                    self._emit(
                        FindingCode.UNKNOWN_SCHEMA_ID,
                        f"Channel {parsed.channel_id} in chunk references unknown "
                        f"Schema {parsed.schema_id}",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                self._validate_channel_advisories(frame, parsed, data_channel_shapes)
                previous = inner_channels.get(parsed.channel_id)
                if previous is not None and not same_channel(previous, parsed):
                    self._emit(
                        FindingCode.CONFLICTING_CHANNEL,
                        f"Channel id {parsed.channel_id} is reused with different content "
                        "in a chunk",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                inner_channels[parsed.channel_id] = parsed
                data_channels[parsed.channel_id] = parsed
            elif isinstance(parsed, MessageRecord):
                if parsed.channel_id not in inner_channels:
                    self._emit(
                        FindingCode.MESSAGE_BEFORE_CHANNEL,
                        f"Message in chunk references channel {parsed.channel_id} before "
                        "its Channel record",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                if chunk_has_message and parsed.log_time < chunk_max_log_time:
                    self._message_order_finding(
                        f"Message.log_time {parsed.log_time} in chunk is less than "
                        f"latest log time {chunk_max_log_time} within the same chunk",
                        frame.offset,
                        frame.section,
                        frame.record_name,
                    )
                if parsed.publish_time > parsed.log_time:
                    self._emit(
                        FindingCode.MESSAGE_PUBLISH_TIME_AFTER_LOG_TIME,
                        "Message.publish_time is later than Message.log_time",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                chunk_max_log_time = max(chunk_max_log_time, parsed.log_time)
                chunk_has_message = True
                chunk_message_count += 1
                chunk_min_time = (
                    parsed.log_time
                    if chunk_min_time is None
                    else min(chunk_min_time, parsed.log_time)
                )
                chunk_max_time = (
                    parsed.log_time
                    if chunk_max_time is None
                    else max(chunk_max_time, parsed.log_time)
                )

        if chunk_message_count == 0:
            if chunk.message_start_time != 0 or chunk.message_end_time != 0:
                detail = (
                    f" (declared start={chunk.message_start_time} "
                    f"end={chunk.message_end_time}; both should be zero)"
                )
            else:
                detail = ""
            self._emit(
                FindingCode.EMPTY_CHUNK,
                f"Chunk contains no Message records{detail}",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        else:
            if chunk.message_start_time != chunk_min_time:
                self._emit(
                    FindingCode.CHUNK_START_TIME_MISMATCH,
                    "Chunk.message_start_time does not match earliest message log_time",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            if chunk.message_end_time != chunk_max_time:
                self._emit(
                    FindingCode.CHUNK_END_TIME_MISMATCH,
                    "Chunk.message_end_time does not match latest message log_time",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )

    def _validate_message_index(self, frame: Frame, index: MessageIndexRecord) -> None:
        if len(index.timestamps) == 0:
            self._emit(
                FindingCode.EMPTY_MESSAGE_INDEX,
                f"MessageIndex for channel {index.channel_id} contains no entries",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        if index.timestamps != sorted(index.timestamps):
            self._emit(
                FindingCode.UNSORTED_MESSAGE_INDEX,
                f"MessageIndex for channel {index.channel_id} is not sorted by log_time",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )

    def _message_order_finding(
        self, message: str, offset: int, section: Section, record_name: str
    ) -> None:
        self._emit(
            FindingCode.NON_MONOTONIC_LOG_TIME,
            message,
            offset=offset,
            section=section,
            record=record_name,
        )

    def _validate_schema_advisories(self, frame: Frame, schema: SchemaRecord) -> None:
        if schema.name == "" and (schema.encoding != "" or schema.data):
            self._emit(
                FindingCode.SCHEMA_NAME_EMPTY,
                "Schema.name is empty",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        if schema.encoding != "" and schema.encoding not in _KNOWN_SCHEMA_ENCODINGS:
            self._emit(
                FindingCode.SCHEMA_ENCODING_UNKNOWN,
                f"Schema.encoding {schema.encoding!r} is not a well-known encoding",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )

    def _validate_channel_advisories(
        self,
        frame: Frame,
        channel: ChannelRecord,
        data_channel_shapes: dict[tuple[str, int, str], int],
    ) -> None:
        if channel.topic == "":
            self._emit(
                FindingCode.CHANNEL_TOPIC_EMPTY,
                "Channel.topic is empty",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        shape = (channel.topic, channel.schema_id, channel.message_encoding)
        previous_channel_id = data_channel_shapes.get(shape)
        if previous_channel_id is not None and previous_channel_id != channel.channel_id:
            self._emit(
                FindingCode.CHANNEL_TOPIC_DUPLICATE,
                "Channel topic/schema/message_encoding duplicates another channel "
                f"with id {previous_channel_id}",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        else:
            data_channel_shapes[shape] = channel.channel_id

    def _validate_chunk_compression_advisories(self, frame: Frame, chunk: ChunkRecord) -> None:
        if (
            chunk.compression != ""
            and chunk.uncompressed_size > 0
            and chunk.compressed_size >= chunk.uncompressed_size
        ):
            self._emit(
                FindingCode.CHUNK_COMPRESSION_INEFFICIENT,
                "Compressed Chunk.records is not smaller than its uncompressed payload",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )

    def _validate_attachment(self, frame: Frame, attachment: AttachmentRecord) -> None:
        if (
            frame.section == Section.DATA
            and attachment.data_size > 0
            and attachment.media_type == ""
        ):
            self._emit(
                FindingCode.ATTACHMENT_MEDIA_TYPE_EMPTY,
                "Attachment.media_type is empty",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )

    def _validate_metadata(
        self,
        frame: Frame,
        metadata: MetadataRecord,
        metadata_names: dict[str, int],
    ) -> None:
        if frame.section != Section.DATA:
            return
        previous_offset = metadata_names.get(metadata.name)
        if previous_offset is not None:
            self._emit(
                FindingCode.METADATA_NAME_DUPLICATE,
                f"Metadata name duplicates record at offset {previous_offset}",
                offset=frame.offset,
                section=frame.section,
                record=frame.record_name,
            )
        else:
            metadata_names[metadata.name] = frame.offset

    def _validate_summary_offsets(self) -> None:
        footer_frame = self._single_footer()
        footer = (
            footer_frame.parsed
            if footer_frame and isinstance(footer_frame.parsed, FooterRecord)
            else None
        )
        summary_frames = [
            frame
            for frame in self.frames
            if frame.section == Section.SUMMARY and not is_private_opcode(frame.opcode)
        ]
        summary_offset_frames = [
            frame for frame in self.frames if frame.section == Section.SUMMARY_OFFSET
        ]

        if summary_frames:
            seen: set[int] = set()
            current_opcode: int | None = None
            for frame in summary_frames:
                if frame.opcode != current_opcode:
                    if frame.opcode in seen:
                        self._emit(
                            FindingCode.SUMMARY_NOT_GROUPED,
                            f"Summary {frame.record_name} records are not grouped by opcode",
                            offset=frame.offset,
                            section=frame.section,
                            record=frame.record_name,
                        )
                    seen.add(frame.opcode)
                    current_opcode = frame.opcode
            footer_frame = self._single_footer()
            footer = (
                footer_frame.parsed
                if footer_frame and isinstance(footer_frame.parsed, FooterRecord)
                else None
            )
            if footer is not None and footer.summary_offset_start == 0:
                self._emit(
                    FindingCode.MISSING_SUMMARY_OFFSETS,
                    "summary section has records but no SummaryOffset section",
                    offset=footer_frame.offset if footer_frame else None,
                    section=Section.FOOTER,
                    record="Footer",
                )

        if not summary_offset_frames:
            return

        groups = summary_groups(summary_frames)
        expected_by_opcode = {group[0].opcode: group for group in groups}
        seen_offsets: set[int] = set()
        for frame in summary_offset_frames:
            if not isinstance(frame.parsed, SummaryOffsetRecord):
                continue
            offset = frame.parsed
            if offset.group_opcode in seen_offsets:
                self._emit(
                    FindingCode.DUPLICATE_SUMMARY_OFFSET,
                    f"duplicate SummaryOffset for opcode {opcode_name(offset.group_opcode)}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            seen_offsets.add(offset.group_opcode)
            group = expected_by_opcode.get(offset.group_opcode)
            if group is None:
                self._emit(
                    FindingCode.SUMMARY_OFFSET_WITHOUT_GROUP,
                    f"SummaryOffset points to missing group {opcode_name(offset.group_opcode)}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
                continue
            expected_start = group[0].offset
            expected_length = group[-1].end_offset - group[0].offset
            if offset.group_start != expected_start or offset.group_length != expected_length:
                self._emit(
                    FindingCode.SUMMARY_OFFSET_RANGE_MISMATCH,
                    f"SummaryOffset for {opcode_name(offset.group_opcode)} does not "
                    "match its summary group",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )

        missing = set(expected_by_opcode) - seen_offsets
        for opcode in sorted(missing):
            self._emit(
                FindingCode.MISSING_SUMMARY_OFFSET,
                f"SummaryOffset section is missing an offset for {opcode_name(opcode)}",
                offset=summary_offset_frames[0].offset,
                section=Section.SUMMARY_OFFSET,
                record="SummaryOffset",
            )

    def _validate_indexes(self) -> None:
        chunk_sequences = self._collect_chunk_index_sequences()
        self._validate_message_index_sequences(chunk_sequences)
        self._validate_chunk_indexes(chunk_sequences)
        self._validate_attachment_indexes()
        self._validate_metadata_indexes()
        self._validate_summary_duplicates()
        self._validate_summary_duplicates_for_indexed_chunks()

    def _collect_chunk_index_sequences(self) -> dict[int, ChunkIndexSequence]:
        sequences: dict[int, ChunkIndexSequence] = {}
        pending: ChunkIndexSequence | None = None
        for frame in self.frames:
            if frame.section != Section.DATA:
                continue
            if isinstance(frame.parsed, ChunkRecord):
                if pending is not None:
                    sequences[pending.chunk.offset] = pending
                pending = ChunkIndexSequence(frame)
            elif isinstance(frame.parsed, MessageIndexRecord):
                if pending is None:
                    self._emit(
                        FindingCode.MESSAGE_INDEX_WITHOUT_CHUNK,
                        "MessageIndex must appear immediately after its Chunk",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                else:
                    pending.indexes.append(frame)
            elif pending is not None:
                sequences[pending.chunk.offset] = pending
                pending = None
        if pending is not None:
            sequences[pending.chunk.offset] = pending
        return sequences

    def _validate_message_index_sequences(self, sequences: dict[int, ChunkIndexSequence]) -> None:
        for sequence in sequences.values():
            chunk = sequence.chunk.parsed
            if not isinstance(chunk, ChunkRecord):
                continue
            expected_channels = {
                channel_id for channel_id, messages in chunk.messages_by_channel.items() if messages
            }
            if not sequence.indexes and expected_channels:
                self._emit(
                    FindingCode.MISSING_MESSAGE_INDEXES,
                    "Chunk has messages but no following MessageIndex records",
                    offset=sequence.chunk.offset,
                    section=sequence.chunk.section,
                    record=sequence.chunk.record_name,
                )
                continue

            seen_channels: set[int] = set()
            for index_frame in sequence.indexes:
                index = index_frame.parsed
                if not isinstance(index, MessageIndexRecord):
                    continue
                if index.channel_id in seen_channels:
                    self._emit(
                        FindingCode.DUPLICATE_MESSAGE_INDEX,
                        f"duplicate MessageIndex for channel {index.channel_id} after chunk",
                        offset=index_frame.offset,
                        section=index_frame.section,
                        record=index_frame.record_name,
                    )
                seen_channels.add(index.channel_id)
                self._validate_message_index_entries(sequence.chunk, chunk, index_frame, index)

            missing = expected_channels - seen_channels
            extra = seen_channels - expected_channels
            for channel_id in sorted(missing):
                self._emit(
                    FindingCode.MISSING_MESSAGE_INDEX,
                    f"Chunk is missing MessageIndex for channel {channel_id}",
                    offset=sequence.chunk.offset,
                    section=sequence.chunk.section,
                    record=sequence.chunk.record_name,
                )
            for channel_id in sorted(extra):
                self._emit(
                    FindingCode.EXTRA_MESSAGE_INDEX,
                    f"MessageIndex for channel {channel_id} has no messages in its chunk",
                    offset=sequence.chunk.offset,
                    section=sequence.chunk.section,
                    record=sequence.chunk.record_name,
                )

    def _validate_message_index_entries(
        self,
        chunk_frame: Frame,
        chunk: ChunkRecord,
        index_frame: Frame,
        index: MessageIndexRecord,
    ) -> None:
        expected_offsets = {
            offset
            for offset, message in chunk.message_by_offset.items()
            if message.channel_id == index.channel_id
        }
        seen_offsets: set[int] = set()
        for relative_offset in index.offsets:
            if relative_offset in seen_offsets:
                self._emit(
                    FindingCode.DUPLICATE_MESSAGE_INDEX_OFFSET,
                    f"MessageIndex contains duplicate offset {relative_offset}",
                    offset=index_frame.offset,
                    section=index_frame.section,
                    record=index_frame.record_name,
                )
            seen_offsets.add(relative_offset)
        actual_offsets = set(index.offsets)
        if expected_offsets and actual_offsets != expected_offsets:
            self._emit(
                FindingCode.MESSAGE_INDEX_OFFSETS_MISMATCH,
                f"MessageIndex offsets for channel {index.channel_id} do not match chunk messages",
                offset=index_frame.offset,
                section=index_frame.section,
                record=index_frame.record_name,
            )
        for timestamp, relative_offset in zip(index.timestamps, index.offsets, strict=True):
            if relative_offset >= chunk.uncompressed_size:
                self._emit(
                    FindingCode.MESSAGE_INDEX_OFFSET_OUT_OF_RANGE,
                    "MessageIndex offset is outside Chunk.uncompressed_size",
                    offset=chunk_frame.offset,
                    section=chunk_frame.section,
                    record=chunk_frame.record_name,
                )
            message = chunk.message_by_offset.get(relative_offset)
            if message is None:
                self._emit(
                    FindingCode.MESSAGE_INDEX_BAD_OFFSET,
                    f"MessageIndex offset {relative_offset} does not point to a Message record",
                    offset=index_frame.offset,
                    section=index_frame.section,
                    record=index_frame.record_name,
                )
                continue
            if message.channel_id != index.channel_id:
                self._emit(
                    FindingCode.MESSAGE_INDEX_CHANNEL_MISMATCH,
                    "MessageIndex channel_id does not match the Message at the indexed offset",
                    offset=index_frame.offset,
                    section=index_frame.section,
                    record=index_frame.record_name,
                )
            if message.log_time != timestamp:
                self._emit(
                    FindingCode.MESSAGE_INDEX_TIME_MISMATCH,
                    "MessageIndex timestamp does not match the Message at the indexed offset",
                    offset=index_frame.offset,
                    section=index_frame.section,
                    record=index_frame.record_name,
                )

    def _validate_chunk_indexes(self, sequences: dict[int, ChunkIndexSequence]) -> None:
        chunks = {
            frame.offset: frame for frame in self.frames if isinstance(frame.parsed, ChunkRecord)
        }
        chunk_index_frames = [
            frame for frame in self.frames if isinstance(frame.parsed, ChunkIndexRecord)
        ]
        if chunks and not chunk_index_frames:
            self._emit(
                FindingCode.MISSING_CHUNK_INDEXES,
                "file contains chunks but no ChunkIndex records; random access will "
                "require scanning",
            )
            return

        if chunk_index_frames:
            self._warn_if_message_records_are_not_indexed_by_chunk_indexes()

        indexed_offsets: set[int] = set()
        for frame in chunk_index_frames:
            index = frame.parsed
            if not isinstance(index, ChunkIndexRecord):
                continue
            if index.chunk_start_offset in indexed_offsets:
                self._emit(
                    FindingCode.DUPLICATE_CHUNK_INDEX,
                    "multiple ChunkIndex records reference chunk offset "
                    f"{index.chunk_start_offset}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            indexed_offsets.add(index.chunk_start_offset)
            chunk_frame = chunks.get(index.chunk_start_offset)
            if chunk_frame is None:
                self._emit(
                    FindingCode.CHUNK_INDEX_BAD_OFFSET,
                    f"ChunkIndex points to offset {index.chunk_start_offset}, which is "
                    "not a Chunk record",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
                continue
            chunk = chunk_frame.parsed
            if not isinstance(chunk, ChunkRecord):
                continue
            self._compare_chunk_index(frame, index, chunk_frame, chunk, sequences)

        missing = set(chunks) - indexed_offsets
        for offset in sorted(missing):
            self._emit(
                FindingCode.MISSING_CHUNK_INDEX,
                f"missing ChunkIndex for chunk at offset {offset}",
                offset=offset,
                section=chunks[offset].section,
                record=chunks[offset].record_name,
            )

    def _compare_chunk_index(
        self,
        index_frame: Frame,
        index: ChunkIndexRecord,
        chunk_frame: Frame,
        chunk: ChunkRecord,
        sequences: dict[int, ChunkIndexSequence],
    ) -> None:
        comparisons = [
            (index.chunk_length, chunk_frame.end_offset - chunk_frame.offset, "chunk_length"),
            (index.message_start_time, chunk.message_start_time, "message_start_time"),
            (index.message_end_time, chunk.message_end_time, "message_end_time"),
            (index.compression, chunk.compression, "compression"),
            (index.compressed_size, chunk.compressed_size, "compressed_size"),
            (index.uncompressed_size, chunk.uncompressed_size, "uncompressed_size"),
        ]
        for actual, expected, field_name in comparisons:
            if actual != expected:
                self._emit(
                    FindingCode.CHUNK_INDEX_MISMATCH,
                    f"ChunkIndex.{field_name} does not match chunk at offset {chunk_frame.offset}",
                    offset=index_frame.offset,
                    section=index_frame.section,
                    record=index_frame.record_name,
                )

        sequence = sequences.get(chunk_frame.offset)
        message_index_frames = sequence.indexes if sequence is not None else []
        actual_offsets = {
            parsed.channel_id: frame.offset
            for frame in message_index_frames
            if isinstance((parsed := frame.parsed), MessageIndexRecord)
        }
        if chunk.messages_by_channel and not index.message_index_offsets:
            self._emit(
                FindingCode.EMPTY_CHUNK_INDEX_MESSAGE_OFFSETS,
                "ChunkIndex.message_index_offsets is empty; index scans cannot locate messages",
                offset=index_frame.offset,
                section=index_frame.section,
                record=index_frame.record_name,
            )
        if index.message_index_offsets and index.message_index_offsets != actual_offsets:
            self._emit(
                FindingCode.CHUNK_INDEX_MESSAGE_INDEX_OFFSETS_MISMATCH,
                "ChunkIndex.message_index_offsets do not match following MessageIndex records",
                offset=index_frame.offset,
                section=index_frame.section,
                record=index_frame.record_name,
            )
        actual_index_length = sum(frame.end_offset - frame.offset for frame in message_index_frames)
        if index.message_index_length != actual_index_length:
            self._emit(
                FindingCode.CHUNK_INDEX_MESSAGE_INDEX_LENGTH_MISMATCH,
                "ChunkIndex.message_index_length does not match following MessageIndex "
                "record bytes",
                offset=index_frame.offset,
                section=index_frame.section,
                record=index_frame.record_name,
            )

    def _warn_if_message_records_are_not_indexed_by_chunk_indexes(self) -> None:
        for frame in self.frames:
            if frame.section == Section.DATA and isinstance(frame.parsed, MessageRecord):
                self._emit(
                    FindingCode.MESSAGE_OUTSIDE_CHUNK_WITH_CHUNK_INDEX,
                    "file has ChunkIndex records but also top-level Message records; "
                    "indexed readers can miss messages outside chunks",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
                return

    def _validate_attachment_indexes(self) -> None:
        attachments = {
            frame.offset: frame
            for frame in self.frames
            if isinstance(frame.parsed, AttachmentRecord)
        }
        indexes = [
            frame for frame in self.frames if isinstance(frame.parsed, AttachmentIndexRecord)
        ]
        if attachments and not indexes:
            self._emit(
                FindingCode.MISSING_ATTACHMENT_INDEXES,
                "file contains attachments but no AttachmentIndex records",
            )
            return
        indexed_offsets: set[int] = set()
        for frame in indexes:
            index = frame.parsed
            if not isinstance(index, AttachmentIndexRecord):
                continue
            if index.offset in indexed_offsets:
                self._emit(
                    FindingCode.DUPLICATE_ATTACHMENT_INDEX,
                    f"multiple AttachmentIndex records reference attachment offset {index.offset}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            indexed_offsets.add(index.offset)
            attachment_frame = attachments.get(index.offset)
            if attachment_frame is None:
                self._emit(
                    FindingCode.ATTACHMENT_INDEX_BAD_OFFSET,
                    f"AttachmentIndex points to offset {index.offset}, which is not an Attachment",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
                continue
            attachment = attachment_frame.parsed
            if isinstance(attachment, AttachmentRecord):
                self._compare_attachment_index(frame, index, attachment_frame, attachment)
        for offset in sorted(set(attachments) - indexed_offsets):
            self._emit(
                FindingCode.MISSING_ATTACHMENT_INDEX,
                f"missing AttachmentIndex for attachment at offset {offset}",
                offset=offset,
                section=attachments[offset].section,
                record=attachments[offset].record_name,
            )

    def _compare_attachment_index(
        self,
        index_frame: Frame,
        index: AttachmentIndexRecord,
        attachment_frame: Frame,
        attachment: AttachmentRecord,
    ) -> None:
        comparisons = [
            (index.length, attachment_frame.end_offset - attachment_frame.offset, "length"),
            (index.log_time, attachment.log_time, "log_time"),
            (index.create_time, attachment.create_time, "create_time"),
            (index.data_size, attachment.data_size, "data_size"),
            (index.name, attachment.name, "name"),
            (index.media_type, attachment.media_type, "media_type"),
        ]
        for actual, expected, field_name in comparisons:
            if actual != expected:
                self._emit(
                    FindingCode.ATTACHMENT_INDEX_MISMATCH,
                    f"AttachmentIndex.{field_name} does not match attachment at offset "
                    f"{attachment_frame.offset}",
                    offset=index_frame.offset,
                    section=index_frame.section,
                    record=index_frame.record_name,
                )

    def _validate_metadata_indexes(self) -> None:
        metadata_records = {
            frame.offset: frame for frame in self.frames if isinstance(frame.parsed, MetadataRecord)
        }
        indexes = [frame for frame in self.frames if isinstance(frame.parsed, MetadataIndexRecord)]
        if metadata_records and not indexes:
            self._emit(
                FindingCode.MISSING_METADATA_INDEXES,
                "file contains metadata but no MetadataIndex records",
            )
            return
        indexed_offsets: set[int] = set()
        for frame in indexes:
            index = frame.parsed
            if not isinstance(index, MetadataIndexRecord):
                continue
            if index.offset in indexed_offsets:
                self._emit(
                    FindingCode.DUPLICATE_METADATA_INDEX,
                    f"multiple MetadataIndex records reference metadata offset {index.offset}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            indexed_offsets.add(index.offset)
            metadata_frame = metadata_records.get(index.offset)
            if metadata_frame is None:
                self._emit(
                    FindingCode.METADATA_INDEX_BAD_OFFSET,
                    f"MetadataIndex points to offset {index.offset}, which is not Metadata",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
                continue
            metadata = metadata_frame.parsed
            if not isinstance(metadata, MetadataRecord):
                continue
            if index.length != metadata_frame.end_offset - metadata_frame.offset:
                self._emit(
                    FindingCode.METADATA_INDEX_MISMATCH,
                    "MetadataIndex.length does not match metadata at offset "
                    f"{metadata_frame.offset}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            if index.name != metadata.name:
                self._emit(
                    FindingCode.METADATA_INDEX_MISMATCH,
                    f"MetadataIndex.name does not match metadata at offset {metadata_frame.offset}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
        for offset in sorted(set(metadata_records) - indexed_offsets):
            self._emit(
                FindingCode.MISSING_METADATA_INDEX,
                f"missing MetadataIndex for metadata at offset {offset}",
                offset=offset,
                section=metadata_records[offset].section,
                record=metadata_records[offset].record_name,
            )

    def _validate_summary_duplicates(self) -> None:
        data_schemas = self._data_schemas()
        data_channels = self._data_channels()
        for frame in self.frames:
            if frame.section != Section.SUMMARY:
                continue
            parsed = frame.parsed
            if isinstance(parsed, SchemaRecord):
                data_schema = data_schemas.get(parsed.schema_id)
                if data_schema is None:
                    self._emit(
                        FindingCode.SUMMARY_SCHEMA_MISSING_IN_DATA,
                        f"Summary Schema {parsed.schema_id} has no matching data Schema",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                elif not same_schema(data_schema, parsed):
                    self._emit(
                        FindingCode.SUMMARY_SCHEMA_MISMATCH,
                        f"Summary Schema {parsed.schema_id} differs from the data Schema",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
            elif isinstance(parsed, ChannelRecord):
                data_channel = data_channels.get(parsed.channel_id)
                if data_channel is None:
                    self._emit(
                        FindingCode.SUMMARY_CHANNEL_MISSING_IN_DATA,
                        f"Summary Channel {parsed.channel_id} has no matching data Channel",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                elif not same_channel(data_channel, parsed):
                    self._emit(
                        FindingCode.SUMMARY_CHANNEL_MISMATCH,
                        f"Summary Channel {parsed.channel_id} differs from the data Channel",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )

    def _validate_summary_duplicates_for_indexed_chunks(self) -> None:
        chunk_indexes = [
            frame.parsed for frame in self.frames if isinstance(frame.parsed, ChunkIndexRecord)
        ]
        if not chunk_indexes:
            return
        summary_channels = {
            parsed.channel_id: parsed
            for frame in self.frames
            if frame.section == Section.SUMMARY
            and isinstance((parsed := frame.parsed), ChannelRecord)
        }
        summary_schemas = {
            parsed.schema_id: parsed
            for frame in self.frames
            if frame.section == Section.SUMMARY
            and isinstance((parsed := frame.parsed), SchemaRecord)
        }
        data_channels = self._data_channels()
        for chunk_frame in self.frames:
            if not isinstance(chunk_frame.parsed, ChunkRecord):
                continue
            for channel_id in sorted(chunk_frame.parsed.referenced_channels):
                channel = data_channels.get(channel_id)
                if channel_id not in summary_channels:
                    self._emit(
                        FindingCode.INDEXED_CHANNEL_MISSING_FROM_SUMMARY,
                        f"indexed chunk references channel {channel_id}, but summary "
                        "lacks its Channel record",
                        offset=chunk_frame.offset,
                        section=chunk_frame.section,
                        record=chunk_frame.record_name,
                    )
                if (
                    channel is not None
                    and channel.schema_id != 0
                    and channel.schema_id not in summary_schemas
                ):
                    self._emit(
                        FindingCode.INDEXED_SCHEMA_MISSING_FROM_SUMMARY,
                        f"indexed chunk references schema {channel.schema_id}, but "
                        "summary lacks its Schema record",
                        offset=chunk_frame.offset,
                        section=chunk_frame.section,
                        record=chunk_frame.record_name,
                    )

    def _validate_statistics(self, actual: ActualCounts) -> None:
        statistics_frames = [
            frame for frame in self.frames if isinstance(frame.parsed, StatisticsRecord)
        ]
        if len(statistics_frames) > 1:
            self._emit(
                FindingCode.MULTIPLE_STATISTICS,
                "file contains multiple Statistics records",
                offset=statistics_frames[1].offset,
                section=statistics_frames[1].section,
                record=statistics_frames[1].record_name,
            )
        if not statistics_frames:
            if actual.message_count > 0:
                self._emit(
                    FindingCode.MISSING_STATISTICS,
                    "file contains messages but no Statistics record",
                )
            return

        frame = statistics_frames[0]
        stats = frame.parsed
        if not isinstance(stats, StatisticsRecord):
            return
        comparisons = [
            (stats.message_count, actual.message_count, "message_count"),
            (stats.schema_count, actual.schema_count, "schema_count"),
            (stats.channel_count, actual.channel_count, "channel_count"),
            (stats.attachment_count, actual.attachment_count, "attachment_count"),
            (stats.metadata_count, actual.metadata_count, "metadata_count"),
            (stats.chunk_count, actual.chunk_count, "chunk_count"),
            (stats.message_start_time, actual.message_start_time, "message_start_time"),
            (stats.message_end_time, actual.message_end_time, "message_end_time"),
        ]
        for actual_value, expected_value, field_name in comparisons:
            if actual_value != expected_value:
                self._emit(
                    FindingCode.STATISTICS_MISMATCH,
                    f"Statistics.{field_name} is {actual_value}, expected {expected_value}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
        for channel_id, count in stats.channel_message_counts.items():
            expected = actual.channel_message_counts.get(channel_id)
            if expected is None:
                self._emit(
                    FindingCode.STATISTICS_UNKNOWN_CHANNEL,
                    f"Statistics.channel_message_counts references unknown channel {channel_id}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
            elif count != expected:
                self._emit(
                    FindingCode.STATISTICS_CHANNEL_COUNT_MISMATCH,
                    f"Statistics channel {channel_id} count is {count}, expected {expected}",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )

        if stats.channel_message_counts:
            for channel_id in actual.channel_message_counts:
                if channel_id not in stats.channel_message_counts:
                    self._emit(
                        FindingCode.STATISTICS_CHANNEL_COUNT_MISSING,
                        f"Statistics.channel_message_counts is missing channel {channel_id}",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
            summary_channel_offsets = {
                parsed.channel_id: summary_frame.offset
                for summary_frame in self.frames
                if summary_frame.section == Section.SUMMARY
                and isinstance((parsed := summary_frame.parsed), ChannelRecord)
            }
            for channel_id in actual.channel_message_counts:
                channel_offset = summary_channel_offsets.get(channel_id)
                if channel_offset is None:
                    self._emit(
                        FindingCode.STATISTICS_CHANNEL_MISSING_FROM_SUMMARY,
                        f"Statistics has per-channel counts but summary lacks Channel {channel_id}",
                        offset=frame.offset,
                        section=frame.section,
                        record=frame.record_name,
                    )
                elif channel_offset > frame.offset:
                    self._emit(
                        FindingCode.STATISTICS_CHANNEL_AFTER_STATISTICS,
                        f"Summary Channel {channel_id} must occur before Statistics",
                        offset=channel_offset,
                        section=Section.SUMMARY,
                        record="Channel",
                    )

    def _validate_advisory_findings(self, actual: ActualCounts) -> None:
        self._validate_absence_advisories(actual)
        self._validate_index_without_summary_advisory()
        self._validate_chunk_size_advisories()
        self._validate_message_sequence_advisory(actual)

    def _validate_absence_advisories(self, actual: ActualCounts) -> None:
        if actual.message_count == 0:
            return
        if actual.attachment_count == 0:
            self._emit(FindingCode.NO_ATTACHMENTS, "file contains no Attachment records")
        if actual.metadata_count == 0:
            self._emit(FindingCode.NO_METADATA, "file contains no Metadata records")
        if actual.channel_count > 0 and actual.schema_count == 0:
            self._emit(
                FindingCode.NO_SCHEMAS,
                "file contains channels but no non-reserved Schema records",
            )

        footer_frame = self._single_footer()
        footer = (
            footer_frame.parsed
            if footer_frame is not None and isinstance(footer_frame.parsed, FooterRecord)
            else None
        )
        if footer is not None and footer.summary_start == 0:
            self._emit(
                FindingCode.NO_SUMMARY_SECTION,
                "file contains messages but no summary section",
                offset=footer_frame.offset if footer_frame else None,
                section=Section.FOOTER,
                record="Footer",
            )

    def _validate_index_without_summary_advisory(self) -> None:
        footer_frame = self._single_footer()
        footer = (
            footer_frame.parsed
            if footer_frame is not None and isinstance(footer_frame.parsed, FooterRecord)
            else None
        )
        if footer is None or footer.summary_start != 0:
            return
        index_opcodes = {
            Opcode.ATTACHMENT_INDEX,
            Opcode.CHUNK_INDEX,
            Opcode.METADATA_INDEX,
            Opcode.STATISTICS,
            Opcode.SUMMARY_OFFSET,
        }
        for frame in self.frames:
            if frame.opcode in index_opcodes:
                self._emit(
                    FindingCode.INDEX_SECTION_PRESENT_WITHOUT_SUMMARY,
                    "file contains index records but Footer.summary_start is zero",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )
                return

    def _validate_chunk_size_advisories(self) -> None:
        chunk_frames = [frame for frame in self.frames if isinstance(frame.parsed, ChunkRecord)]
        if not chunk_frames:
            return

        compressions = {
            chunk.compression
            for frame in chunk_frames
            if isinstance((chunk := frame.parsed), ChunkRecord)
        }
        if compressions == {""}:
            self._emit(FindingCode.NO_COMPRESSION, "all chunks are uncompressed")
        elif len(compressions) > 1:
            self._emit(
                FindingCode.MIXED_COMPRESSION_TYPES,
                "file uses more than one chunk compression type",
            )

        sizes: list[int] = []
        for frame in chunk_frames:
            chunk = frame.parsed
            if not isinstance(chunk, ChunkRecord):
                continue
            sizes.append(chunk.uncompressed_size)
            if chunk.uncompressed_size > _LARGE_CHUNK_BYTES:
                self._emit(
                    FindingCode.LARGE_CHUNK,
                    "Chunk.uncompressed_size exceeds the advisory large-chunk threshold",
                    offset=frame.offset,
                    section=frame.section,
                    record=frame.record_name,
                )

        if len(sizes) >= _SMALL_CHUNK_MIN_COUNT and median_int(sizes) < _SMALL_CHUNK_MEDIAN_BYTES:
            self._emit(
                FindingCode.SMALL_CHUNKS,
                "median Chunk.uncompressed_size is below the advisory small-chunk threshold",
                offset=chunk_frames[0].offset,
                section=chunk_frames[0].section,
                record=chunk_frames[0].record_name,
            )

    def _validate_message_sequence_advisory(self, actual: ActualCounts) -> None:
        if actual.message_count == 0:
            return

        first_offset: int | None = None
        first_section = Section.UNKNOWN
        first_record = ""
        for frame in self.frames:
            if frame.section != Section.DATA:
                continue
            parsed = frame.parsed
            if isinstance(parsed, MessageRecord):
                if first_offset is None:
                    first_offset = frame.offset
                    first_section = frame.section
                    first_record = frame.record_name
                if parsed.sequence != 0:
                    return
            elif isinstance(parsed, ChunkRecord):
                for inner in parsed.inner_records:
                    if not isinstance(inner.parsed, MessageRecord):
                        continue
                    if first_offset is None:
                        first_offset = frame.offset
                        first_section = frame.section
                        first_record = frame.record_name
                    if inner.parsed.sequence != 0:
                        return

        self._emit(
            FindingCode.ZERO_MESSAGE_SEQUENCE,
            "all Message.sequence values are zero",
            offset=first_offset,
            section=first_section,
            record=first_record,
        )

    def _single_footer(self) -> Frame | None:
        footers = [frame for frame in self.frames if frame.opcode == Opcode.FOOTER]
        return footers[0] if len(footers) == 1 else None

    def _first_frame(self, opcode: int) -> Frame | None:
        for frame in self.frames:
            if frame.opcode == opcode:
                return frame
        return None

    def _data_channels(self) -> dict[int, ChannelRecord]:
        channels: dict[int, ChannelRecord] = {}
        for frame in self.frames:
            if frame.section != Section.DATA:
                continue
            if isinstance(frame.parsed, ChannelRecord):
                channels[frame.parsed.channel_id] = frame.parsed
            elif isinstance(frame.parsed, ChunkRecord):
                for inner in frame.parsed.inner_records:
                    if isinstance(inner.parsed, ChannelRecord):
                        channels[inner.parsed.channel_id] = inner.parsed
        return channels

    def _data_schemas(self) -> dict[int, SchemaRecord]:
        schemas: dict[int, SchemaRecord] = {}
        for frame in self.frames:
            if frame.section != Section.DATA:
                continue
            if isinstance(frame.parsed, SchemaRecord):
                schemas[frame.parsed.schema_id] = frame.parsed
            elif isinstance(frame.parsed, ChunkRecord):
                for inner in frame.parsed.inner_records:
                    if isinstance(inner.parsed, SchemaRecord):
                        schemas[inner.parsed.schema_id] = inner.parsed
        return schemas

    def _actual_counts(self) -> ActualCounts:
        schemas: set[int] = set()
        channels: set[int] = set()
        channel_message_counts: Counter[int] = Counter()
        attachments = 0
        metadata = 0
        chunks = 0
        message_times: list[int] = []

        for frame in self.frames:
            if frame.section != Section.DATA:
                continue
            parsed = frame.parsed
            if isinstance(parsed, SchemaRecord) and parsed.schema_id != 0:
                schemas.add(parsed.schema_id)
            elif isinstance(parsed, ChannelRecord):
                channels.add(parsed.channel_id)
            elif isinstance(parsed, MessageRecord):
                channel_message_counts[parsed.channel_id] += 1
                message_times.append(parsed.log_time)
            elif isinstance(parsed, AttachmentRecord):
                attachments += 1
            elif isinstance(parsed, MetadataRecord):
                metadata += 1
            elif isinstance(parsed, ChunkRecord):
                chunks += 1
                for inner in parsed.inner_records:
                    if isinstance(inner.parsed, SchemaRecord) and inner.parsed.schema_id != 0:
                        schemas.add(inner.parsed.schema_id)
                    elif isinstance(inner.parsed, ChannelRecord):
                        channels.add(inner.parsed.channel_id)
                    elif isinstance(inner.parsed, MessageRecord):
                        channel_message_counts[inner.parsed.channel_id] += 1
                        message_times.append(inner.parsed.log_time)

        return ActualCounts(
            message_count=sum(channel_message_counts.values()),
            schema_count=len(schemas),
            channel_count=len(channels),
            attachment_count=attachments,
            metadata_count=metadata,
            chunk_count=chunks,
            message_start_time=min(message_times) if message_times else 0,
            message_end_time=max(message_times) if message_times else 0,
            channel_message_counts=dict(channel_message_counts),
        )


@dataclass(slots=True)
class ActualCounts:
    message_count: int
    schema_count: int
    channel_count: int
    attachment_count: int
    metadata_count: int
    chunk_count: int
    message_start_time: int
    message_end_time: int
    channel_message_counts: dict[int, int]


def opcode_name(opcode: int) -> str:
    try:
        return Opcode(opcode).name.title().replace("_", "")
    except ValueError:
        if is_private_opcode(opcode):
            return f"Private(0x{opcode:02x})"
        return f"Opcode(0x{opcode:02x})"


def is_private_opcode(opcode: int) -> bool:
    return 0x80 <= opcode <= 0xFF


def same_schema(left: SchemaRecord, right: SchemaRecord) -> bool:
    return (
        left.schema_id == right.schema_id
        and left.name == right.name
        and left.encoding == right.encoding
        and left.data == right.data
    )


def same_channel(left: ChannelRecord, right: ChannelRecord) -> bool:
    return (
        left.channel_id == right.channel_id
        and left.schema_id == right.schema_id
        and left.topic == right.topic
        and left.message_encoding == right.message_encoding
        and left.metadata.values == right.metadata.values
    )


def median_int(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def summary_groups(frames: list[Frame]) -> list[list[Frame]]:
    if not frames:
        return []
    groups: list[list[Frame]] = []
    current: list[Frame] = [frames[0]]
    for frame in frames[1:]:
        if frame.opcode == current[-1].opcode:
            current.append(frame)
        else:
            groups.append(current)
            current = [frame]
    groups.append(current)
    return groups


_KNOWN_OPCODES = {opcode.value for opcode in Opcode}
_DATA_OPCODES = {
    Opcode.SCHEMA,
    Opcode.CHANNEL,
    Opcode.MESSAGE,
    Opcode.ATTACHMENT,
    Opcode.CHUNK,
    Opcode.MESSAGE_INDEX,
    Opcode.METADATA,
    Opcode.DATA_END,
}
_SUMMARY_OPCODES = {
    Opcode.SCHEMA,
    Opcode.CHANNEL,
    Opcode.CHUNK_INDEX,
    Opcode.ATTACHMENT_INDEX,
    Opcode.METADATA_INDEX,
    Opcode.STATISTICS,
}


def examine_mcap(
    stream: IO[bytes],
    size: int,
    path: str,
    *,
    strict_message_order: bool = False,
) -> DoctorReport:
    return McapDoctor(strict_message_order=strict_message_order).examine(stream, size, path)
