"""Non-blocking follower for append-only local MCAP files."""

from __future__ import annotations

import stat
import time
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from small_mcap.exceptions import (
    ChannelNotFoundError,
    CRCValidationError,
    InvalidHeaderError,
    InvalidMagicError,
    McapFileReplacedError,
    McapFileTruncatedError,
    SchemaNotFoundError,
)
from small_mcap.reader import _validate_attachment_crc, breakup_chunk, try_read_record
from small_mcap.records import (
    MAGIC,
    MAGIC_SIZE,
    Attachment,
    Channel,
    Chunk,
    DataEnd,
    Footer,
    Header,
    McapRecord,
    Message,
    Opcode,
    Schema,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator
    from types import TracebackType
    from typing import IO

    from typing_extensions import Self

MessageTuple: TypeAlias = tuple[Schema | None, Channel, Message]


@dataclass(frozen=True, slots=True)
class FollowBatch:
    messages: tuple[MessageTuple, ...]
    committed_offset: int
    is_final: bool


@dataclass(frozen=True, slots=True)
class _QueuedMessage:
    value: MessageTuple
    size: int


class McapFollower:
    """Stateful parser for an append-only local regular file."""

    def __init__(
        self,
        path: Path,
        stream: IO[bytes],
        *,
        device: int,
        inode: int,
        validate_crc: bool,
    ) -> None:
        self.path = path
        self._stream = stream
        self._device = device
        self._inode = inode
        self._validate_crc = validate_crc
        self._committed_offset = 0
        self._is_magic_committed = False
        self._is_header_seen = False
        self._is_data_ended = False
        self._is_footer_seen = False
        self._is_final = False
        self._is_closed = False
        self._checksum = 0
        self._schemas: dict[int, Schema] = {}
        self._channels: dict[int, Channel] = {}
        self._pending: deque[_QueuedMessage] = deque()

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, validate_crc: bool = False) -> McapFollower:
        """Open an existing local regular file, including a zero-byte new file."""
        resolved = Path(path).resolve()
        file_stat = resolved.stat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"MCAP follower requires a local regular file: {path}")
        stream = resolved.open("rb", buffering=0)
        return cls(
            resolved,
            stream,
            device=file_stat.st_dev,
            inode=file_stat.st_ino,
            validate_crc=validate_crc,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._is_closed:
            self._stream.close()
            self._is_closed = True

    def poll_messages(
        self,
        *,
        max_messages: int = 1000,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> FollowBatch:
        """Parse currently complete records without waiting for file growth."""
        if self._is_closed:
            raise ValueError("MCAP follower is closed")
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        self._check_file_identity()

        messages: list[MessageTuple] = []
        output_bytes = self._drain_pending(messages, max_messages, max_bytes, 0)
        if self._pending or self._is_final:
            return FollowBatch(tuple(messages), self._committed_offset, self._is_final)

        read_bytes = 0
        if not self._is_magic_committed:
            magic = self._stream.read(MAGIC_SIZE)
            if len(magic) < MAGIC_SIZE:
                self._stream.seek(self._committed_offset)
                return FollowBatch(tuple(messages), self._committed_offset, False)
            if magic != MAGIC:
                raise InvalidMagicError(magic)
            self._is_magic_committed = True
            self._committed_offset = self._stream.tell()
            read_bytes += MAGIC_SIZE
            if self._validate_crc:
                self._checksum = zlib.crc32(magic, self._checksum)

        while len(messages) < max_messages and read_bytes < max_bytes:
            if self._is_footer_seen:
                self._read_trailing_magic()
                break
            result = try_read_record(self._stream)
            if result is None:
                break
            self._committed_offset = result.end_offset
            read_bytes += len(result.header) + len(result.body)
            self._process_complete_record(result.opcode, result.record, result.header, result.body)
            output_bytes = self._drain_pending(
                messages,
                max_messages,
                max_bytes,
                output_bytes,
            )
            if self._pending:
                break

        return FollowBatch(tuple(messages), self._committed_offset, self._is_final)

    def iter_messages(
        self,
        *,
        poll_interval: float = 0.1,
        idle_timeout: float | None = None,
        max_messages: int = 1000,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> Iterator[MessageTuple]:
        """Poll with sleeping until the file is final or the idle timeout expires."""
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")
        if idle_timeout is not None and idle_timeout < 0:
            raise ValueError("idle_timeout must be non-negative")
        last_activity = time.monotonic()
        previous_offset = self._committed_offset
        while True:
            batch = self.poll_messages(max_messages=max_messages, max_bytes=max_bytes)
            if batch.messages or batch.committed_offset != previous_offset:
                last_activity = time.monotonic()
            previous_offset = batch.committed_offset
            yield from batch.messages
            if batch.is_final:
                return
            if idle_timeout is not None and time.monotonic() - last_activity >= idle_timeout:
                return
            time.sleep(poll_interval)

    def _check_file_identity(self) -> None:
        try:
            current = self.path.stat()
        except FileNotFoundError as exc:
            raise McapFileReplacedError(str(self.path)) from exc
        if current.st_dev != self._device or current.st_ino != self._inode:
            raise McapFileReplacedError(str(self.path))
        if current.st_size < self._committed_offset:
            raise McapFileTruncatedError(
                str(self.path),
                current.st_size,
                self._committed_offset,
            )

    def _read_trailing_magic(self) -> None:
        start = self._committed_offset
        magic = self._stream.read(MAGIC_SIZE)
        if len(magic) < MAGIC_SIZE:
            self._stream.seek(start)
            return
        if magic != MAGIC:
            raise InvalidMagicError(magic)
        self._committed_offset = self._stream.tell()
        self._is_final = True

    def _process_complete_record(
        self,
        opcode: int,
        record: McapRecord | None,
        header: bytes,
        body: bytes,
    ) -> None:
        if not self._is_header_seen:
            if not isinstance(record, Header):
                raise InvalidHeaderError(type(record))
            self._is_header_seen = True

        if self._validate_crc:
            if isinstance(record, DataEnd):
                if record.data_section_crc not in (0, self._checksum):
                    raise CRCValidationError(record.data_section_crc, self._checksum, record)
            elif not self._is_data_ended:
                self._checksum = zlib.crc32(header, self._checksum)
                self._checksum = zlib.crc32(body, self._checksum)
            if isinstance(record, Attachment):
                _validate_attachment_crc(memoryview(body), record)

        if isinstance(record, DataEnd):
            self._is_data_ended = True
        elif isinstance(record, Footer):
            self._is_footer_seen = True
        elif isinstance(record, Chunk):
            for inner in breakup_chunk(record, validate_crc=self._validate_crc):
                self._process_content_record(inner)
        elif record is not None and opcode in (Opcode.SCHEMA, Opcode.CHANNEL, Opcode.MESSAGE):
            self._process_content_record(record)

    def _process_content_record(self, record: McapRecord) -> None:
        if isinstance(record, Schema):
            existing = self._schemas.get(record.id)
            if existing is not None and existing != record:
                raise ValueError(f"conflicting schema ID {record.id}")
            self._schemas[record.id] = record
            return
        if isinstance(record, Channel):
            if record.schema_id != 0 and record.schema_id not in self._schemas:
                raise SchemaNotFoundError(record.schema_id)
            existing = self._channels.get(record.id)
            if existing is not None and existing != record:
                raise ValueError(f"conflicting channel ID {record.id}")
            self._channels[record.id] = record
            return
        if not isinstance(record, Message):
            return
        channel = self._channels.get(record.channel_id)
        if channel is None:
            raise ChannelNotFoundError(record.channel_id)
        schema = self._schemas.get(channel.schema_id)
        self._pending.append(
            _QueuedMessage(
                (schema, channel, record),
                Message._STRUCT.size + len(record.data),  # noqa: SLF001
            )
        )

    def _drain_pending(
        self,
        messages: list[MessageTuple],
        max_messages: int,
        max_bytes: int,
        output_bytes: int,
    ) -> int:
        while self._pending and len(messages) < max_messages:
            queued = self._pending[0]
            if messages and output_bytes + queued.size > max_bytes:
                break
            self._pending.popleft()
            messages.append(queued.value)
            output_bytes += queued.size
        return output_bytes
