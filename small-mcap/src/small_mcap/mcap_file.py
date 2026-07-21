from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING

from typing_extensions import Self

from small_mcap.exceptions import EndOfFileError
from small_mcap.reader import (
    _get_chunk_data_stream,
    _LoadedChunk,
    _read_chunk_and_indexes,
    _read_message_indexed,
    _should_include_all,
    get_summary,
    read_message,
)
from small_mcap.records import Channel, ChunkIndex, Message, Schema, Summary

if TYPE_CHECKING:
    from types import TracebackType

    from _typeshed import StrPath

_DEFAULT_CHUNK_CACHE_BYTES = 64 * 1024 * 1024
_MessageIterator = Iterable[tuple[Schema | None, Channel, Message]]
_ChannelPredicate = Callable[[Channel, Schema | None], bool]
_CacheKey = tuple[int, bool]


class McapFile:
    """An opened, indexed MCAP file with independent cached message iterators."""

    def __init__(
        self,
        path: Path,
        stream: IO[bytes],
        summary: Summary | None,
        chunk_cache_bytes: int,
    ) -> None:
        self._path = path
        self._stream = stream
        self._summary = summary
        self._chunk_cache_bytes = chunk_cache_bytes
        self._chunk_cache: OrderedDict[_CacheKey, _LoadedChunk] = OrderedDict()
        self._cached_chunk_bytes = 0
        self._source_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._is_closed = False

    @classmethod
    def open(
        cls,
        path: StrPath,
        *,
        chunk_cache_bytes: int = _DEFAULT_CHUNK_CACHE_BYTES,
    ) -> Self:
        if chunk_cache_bytes < 0:
            raise ValueError("chunk_cache_bytes must be non-negative")
        resolved_path = Path(path)
        stream = resolved_path.open("rb")
        try:
            summary = get_summary(stream)
        except BaseException:
            stream.close()
            raise
        return cls(resolved_path, stream, summary, chunk_cache_bytes)

    @property
    def summary(self) -> Summary | None:
        return self._summary

    def read_message(
        self,
        should_include: _ChannelPredicate = _should_include_all,
        start_time_ns: int = 0,
        end_time_ns: int = sys.maxsize,
        validate_crc: bool = False,
        reverse: bool = False,
        num_workers: int = 0,
    ) -> _MessageIterator:
        self._ensure_open()
        summary = self._summary
        if summary is None or not summary.chunk_indexes or num_workers > 0:
            messages = self._read_fallback(
                should_include,
                start_time_ns,
                end_time_ns,
                validate_crc,
                reverse,
                num_workers,
            )
        else:
            messages = _read_message_indexed(
                summary,
                should_include,
                start_time_ns,
                end_time_ns,
                validate_crc,
                reverse,
                num_workers=0,
                load_chunk=self._load_chunk,
            )
        return self._guard(messages)

    def close(self) -> None:
        with self._source_lock:
            if self._is_closed:
                return
            self._is_closed = True
            self._stream.close()
        with self._cache_lock:
            self._chunk_cache.clear()
            self._cached_chunk_bytes = 0

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._is_closed:
            raise RuntimeError("McapFile is closed")

    def _guard(
        self, messages: _MessageIterator
    ) -> Iterator[tuple[Schema | None, Channel, Message]]:
        iterator = iter(messages)
        while True:
            self._ensure_open()
            try:
                item = next(iterator)
            except StopIteration:
                return
            self._ensure_open()
            yield item

    def _read_fallback(
        self,
        should_include: _ChannelPredicate,
        start_time_ns: int,
        end_time_ns: int,
        validate_crc: bool,
        reverse: bool,
        num_workers: int,
    ) -> Iterator[tuple[Schema | None, Channel, Message]]:
        with self._path.open("rb") as stream:
            yield from read_message(
                stream,
                should_include=should_include,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                validate_crc=validate_crc,
                reverse=reverse,
                num_workers=num_workers,
            )

    def _read_at(self, offset: int, length: int) -> bytes:
        with self._source_lock:
            self._ensure_open()
            self._stream.seek(offset)
            data = self._stream.read(length)
        if len(data) != length:
            raise EndOfFileError(f"expected {length} bytes at offset {offset}, got {len(data)}")
        return data

    def _load_chunk(self, index: ChunkIndex, validate_crc: bool) -> _LoadedChunk:
        key = (index.chunk_start_offset, validate_crc)
        with self._cache_lock:
            self._ensure_open()
            cached = self._chunk_cache.get(key)
            if cached is not None:
                self._chunk_cache.move_to_end(key)
                return cached

        raw = self._read_at(
            index.chunk_start_offset,
            index.chunk_length + index.message_index_length,
        )
        chunk, message_indexes = _read_chunk_and_indexes(raw)
        decompressed_data = bytes(_get_chunk_data_stream(chunk, validate_crc=validate_crc))
        loaded = _LoadedChunk(
            chunk=None,
            message_indexes=tuple(message_indexes),
            decompressed_data=decompressed_data,
        )
        size_bytes = len(decompressed_data)
        if size_bytes > self._chunk_cache_bytes:
            return loaded

        with self._cache_lock:
            self._ensure_open()
            existing = self._chunk_cache.get(key)
            if existing is not None:
                self._chunk_cache.move_to_end(key)
                return existing
            self._chunk_cache[key] = loaded
            self._cached_chunk_bytes += size_bytes
            while self._cached_chunk_bytes > self._chunk_cache_bytes:
                _, evicted = self._chunk_cache.popitem(last=False)
                assert evicted.decompressed_data is not None
                self._cached_chunk_bytes -= len(evicted.decompressed_data)
        return loaded
