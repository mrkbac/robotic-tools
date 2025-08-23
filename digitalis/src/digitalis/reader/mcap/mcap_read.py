from collections.abc import Callable, Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import lz4.frame
import zstandard
from cachetools import LRUCache
from mcap.data_stream import ReadDataStream
from mcap.decoder import DecoderFactory
from mcap.exceptions import DecoderNotFoundError
from mcap.opcode import Opcode
from mcap.reader import SeekingReader
from mcap.records import Channel, Chunk, ChunkIndex, Message, MessageIndex, Schema
from mcap.stream_reader import StreamReader

from digitalis.exceptions import ChannelNotFoundError, InvalidFileFormatError, MessageNotFoundError


def _get_size(data: bytes | None) -> int:
    if data is None:
        return 1
    return len(data)


class McapReader:
    def __init__(
        self,
        file_path: str | Path,
        decoder_factories: Iterable[DecoderFactory] = (),
        cache_size: int = 1000 * 1024 * 1024,  # 1GB
    ) -> None:
        self._file_path: Path = Path(file_path)
        self._file = self._file_path.open("rb")

        reader = SeekingReader(self._file)
        summary = reader.get_summary()
        if summary is None:
            raise InvalidFileFormatError("No summary found in MCAP file.")
        self.summary = summary

        self._topic_index: dict[int, list[tuple[int, MessageIndex]]] = self._build_index()

        self._cache = LRUCache(maxsize=cache_size, getsizeof=_get_size)
        self._timestamps_cache: dict[int, list[int]] = {}

        self._decoded_message_cache: LRUCache[tuple[int, int], Any] = LRUCache(maxsize=1000)

        self._decoder_factories = decoder_factories
        self._decoders: dict[int, Callable[[bytes], Any]] = {}

    def close(self) -> None:
        self._file.close()

    def _build_index(self) -> dict[int, list[tuple[int, MessageIndex]]]:
        """Reads the MCAP summary and builds an index mapping each channel ID to a list of tuples.

        Each tuple contains the chunk start offset and the associated MessageIndex.
        """
        topic_index: dict[int, list[tuple[int, MessageIndex]]] = {}

        for chunk_index in self.summary.chunk_indexes:
            self._process_chunk_index(chunk_index, topic_index)
        return topic_index

    def _process_chunk_index(
        self,
        chunk_index: ChunkIndex,
        topic_index: dict[int, list[tuple[int, MessageIndex]]],
    ) -> None:
        """Process a single chunk index and update the topic index."""
        for channel_id, offset in chunk_index.message_index_offsets.items():
            message_index = self._read_message_index(offset)
            if message_index.records:
                topic_index.setdefault(channel_id, []).append(
                    (chunk_index.chunk_start_offset, message_index)
                )

    def _read_message_index(self, offset: int) -> MessageIndex:
        """Read a MessageIndex from the given offset."""
        self._file.seek(offset)
        stream_reader = StreamReader(self._file, skip_magic=True)
        record = next(stream_reader.records)
        if not isinstance(record, MessageIndex):
            raise InvalidFileFormatError(
                f"Expected MessageIndex at offset {offset}, got {type(record).__name__}."
            )
        return record

    def _read_chunk(self, chunk_offset: int) -> bytes:
        """Seeks to the chunk start (adjusted by header size) and reads the chunk.

        Decompresses the chunk data if needed.
        """
        # Adjusting the chunk offset by header size (1 + 8 bytes)
        if cached_chunk := self._cache.get(chunk_offset):
            return cached_chunk

        self._file.seek(chunk_offset + 1 + 8)  # 1 byte OPCODE + 8 byte record length
        chunk = Chunk.read(ReadDataStream(self._file))
        if chunk.compression == "zstd":
            data: bytes = zstandard.decompress(chunk.data, chunk.uncompressed_size)
        elif chunk.compression == "lz4":
            data: bytes = lz4.frame.decompress(chunk.data)
        else:
            data = chunk.data

        self._cache[chunk_offset] = data

        return data

    def get_msg_by_timestamp(self, channel_id: int, timestamp_ns: int) -> Message | None:
        """Retrieve a message from the given channel at the specified timestamp.

        Parameters:
            channel_id (int): The channel ID to look up.
            timestamp_ns (int): The timestamp in nanoseconds to find.

        Returns:
            Message: The MCAP Message corresponding to the specified channel and timestamp.
                    Returns the message with the closest timestamp that is <= timestamp_ns.
        """
        if channel_id not in self._topic_index:
            return None

        best_message = self._find_best_message(channel_id, timestamp_ns)
        if best_message is None:
            raise MessageNotFoundError(
                f"No message found at or before timestamp {timestamp_ns} on channel {channel_id}."
            )

        return self._extract_message_from_chunk(*best_message)

    def _find_best_message(self, channel_id: int, timestamp_ns: int) -> tuple[int, int] | None:
        """Find the best message for the given timestamp."""
        best_message = None
        best_timestamp = None

        for chunk_offset, msg_index in self._topic_index[channel_id]:
            for timestamp, msg_offset in msg_index.records:
                if timestamp <= timestamp_ns:
                    # Check if this is a better (closer) match
                    if best_timestamp is None or timestamp > best_timestamp:
                        best_timestamp = timestamp
                        best_message = (chunk_offset, msg_offset)
                elif timestamp > timestamp_ns:
                    # We've gone past the target timestamp, stop searching this chunk
                    break

        return best_message

    def _extract_message_from_chunk(self, chunk_offset: int, msg_offset: int) -> Message:
        """Extract a message from the specified chunk and offset."""
        chunk_data = self._read_chunk(chunk_offset)
        if msg_offset >= len(chunk_data):
            raise InvalidFileFormatError(
                f"Message offset {msg_offset} exceeds uncompressed size {len(chunk_data)}."
            )

        # Navigate to the specific message within the chunk
        chunk_io = BytesIO(chunk_data)
        chunk_io.seek(msg_offset)
        chunk_stream = ReadDataStream(chunk_io)
        opcode = chunk_stream.read1()
        length = chunk_stream.read8()
        if opcode == Opcode.MESSAGE:
            return Message.read(chunk_stream, length)

        raise InvalidFileFormatError(f"Expected MESSAGE opcode, got {opcode:02X}.")

    def _decoded_message(self, schema: Schema | None, channel: Channel, message: Message) -> Any:
        decoder = self._decoders.get(message.channel_id)
        if decoder is not None:
            return decoder(message.data)
        for factory in self._decoder_factories:
            decoder = factory.decoder_for(channel.message_encoding, schema)
            if decoder is not None:
                self._decoders[message.channel_id] = decoder
                return decoder(message.data)

        msg = (
            f"no decoder factory supplied for message encoding {channel.message_encoding}, "
            f"schema {schema}"
        )
        raise DecoderNotFoundError(msg)

    def get_msg_decoded_by_timestamp(self, channel_id: int, timestamp_ns: int) -> Any:
        channel = self.summary.channels.get(channel_id)
        if channel is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not found in index.")
        schema = self.summary.schemas.get(channel.schema_id)

        message = self.get_msg_by_timestamp(channel_id, timestamp_ns)
        if message is None:
            raise MessageNotFoundError(
                f"No message found at or before timestamp {timestamp_ns} on channel {channel_id}."
            )

        # Check cache with actual message timestamp
        cache_key = (channel_id, message.log_time)
        if cache_key in self._decoded_message_cache:
            return self._decoded_message_cache[cache_key]

        # Decode and cache
        decoded_message = self._decoded_message(schema, channel, message)
        self._decoded_message_cache[cache_key] = decoded_message
        return decoded_message
