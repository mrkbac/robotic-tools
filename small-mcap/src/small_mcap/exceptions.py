import contextlib
from typing import TYPE_CHECKING

from small_mcap.records import Opcode

if TYPE_CHECKING:
    from small_mcap.records import McapRecord


class McapError(Exception):
    pass


class InvalidMagicError(McapError):
    def __init__(self, bad_magic: bytes | memoryview) -> None:
        super().__init__(
            f"not a valid MCAP file, invalid magic: {bytes(bad_magic).decode('utf-8', 'replace')}"
        )


class EndOfFileError(McapError):
    pass


class CRCValidationError(McapError):
    def __init__(self, expected: int, actual: int, record: "McapRecord") -> None:
        super().__init__(
            f"crc validation failed in {type(record).__name__}, "
            f"expected: {expected}, calculated: {actual}"
        )


class RecordLengthLimitExceededError(McapError):
    def __init__(self, opcode: int, length: int, limit: int) -> None:
        opcode_name = f"unknown (opcode {opcode})"
        with contextlib.suppress(ValueError):
            opcode_name = Opcode(opcode).name
        super().__init__(
            f"{opcode_name} record has length {length} that exceeds limit {limit}",
        )


class UnsupportedCompressionError(McapError):
    def __init__(self, compression: str) -> None:
        super().__init__(f"unsupported compression type {compression}")


class SchemaNotFoundError(McapError):
    def __init__(
        self, schema_id: int, *, topic: str | None = None, stream_id: int | None = None
    ) -> None:
        if topic is not None and stream_id is not None:
            msg = (
                f"Channel '{topic}' references schema_id={schema_id} "
                f"which has not been seen yet (stream_id={stream_id})"
            )
        else:
            msg = f"no schema record found with id {schema_id}"
        super().__init__(msg)


class ChannelNotFoundError(McapError):
    def __init__(self, channel_id: int, *, stream_id: int | None = None) -> None:
        if stream_id is not None:
            msg = (
                f"Message references channel_id={channel_id} which has not been seen yet "
                f"(stream_id={stream_id})"
            )
        else:
            msg = f"no channel record found with id {channel_id}"
        super().__init__(msg)


class InvalidHeaderError(McapError):
    def __init__(self, found_type: type) -> None:
        super().__init__(
            f"expected header at beginning of MCAP file, found {found_type}"
        )


class SeekRequiredError(McapError):
    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} is not supported for non-seekable streams.")
