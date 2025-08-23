import contextlib
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from typing import IO, Any

import cachetools
import lz4.frame
import mcap
import mcap.exceptions
import zstandard
from mcap.exceptions import DecoderNotFoundError
from mcap.reader import SeekingReader
from mcap.records import Channel, Chunk, Message, Schema
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory

from digitalis.exceptions import ChannelNotFoundError, InvalidFileFormatError
from digitalis.reader.types import MessageEvent


class EndOfFileError(Exception):
    """Exception raised when the end of the file is reached."""


class McapReaderPreloading:
    def __init__(
        self,
        file_path: str | Path,
    ) -> None:
        self._decoder_factory = DecoderFactory()
        with Path(file_path).open("rb") as file:
            reader = SeekingReader(file)
            summary = reader.get_summary()
            if summary is None:
                raise InvalidFileFormatError("No summary found in MCAP file.")
            self.summary = summary
            assert self.summary.statistics
            self.statistics = self.summary.statistics

        self._file_path: Path = Path(file_path)
        self._subscribed_topics_id: set[int] = set()
        self._message_iterator: Iterator[MessageEvent] | None = None

        # Chunk cache for decompressed data (LRU with max 20 chunks)
        self._chunk_cache = cachetools.LRUCache(maxsize=20)

        self._file: IO[bytes] = self._file_path.open("rb")
        self._message_iterator = iter(self._get_message(self._file))
        self._file.seek(8)  # Skip magic bytes

    def set_subscription(self, topics: list[str]) -> None:
        """Set the topics to subscribe to."""
        new_subscribed_topics_id = {
            channel.id for channel in self.summary.channels.values() if channel.topic in topics
        }

        # Only recreate iterator if subscription actually changed
        if self._subscribed_topics_id != new_subscribed_topics_id:
            self._subscribed_topics_id = new_subscribed_topics_id
            # Create new iterator at current file position
            self._message_iterator = iter(self._get_message(self._file))

    def close(self) -> None:
        """Clean up resources."""
        if self._file and not self._file.closed:
            self._file.close()
        self._message_iterator = None
        self._chunk_cache.clear()

    def _decoded_message(self, schema: Schema | None, channel: Channel, message: Message) -> Any:
        decoder = self._decoder_factory.decoder_for(channel.message_encoding, schema)
        if decoder is not None:
            return decoder(message.data)

        msg = (
            f"no decoder factory supplied for message encoding {channel.message_encoding}, "
            f"schema {schema}"
        )
        raise DecoderNotFoundError(msg)

    def _decompress_chunk(
        self, chunk_offset: int, compression: str, data: bytes, uncompressed_size: int
    ) -> bytes:
        """Decompress chunk data with LRU caching."""
        # Check if chunk is already cached
        if chunk_offset in self._chunk_cache:
            # Move to end (most recently used)
            return self._chunk_cache[chunk_offset]

        # Decompress chunk data
        if compression == "zstd":
            decompressed_data = zstandard.decompress(data, uncompressed_size)
        elif compression == "lz4":
            decompressed_data = lz4.frame.decompress(data)
        else:
            decompressed_data = data

        # Add to cache and manage size
        self._chunk_cache[chunk_offset] = decompressed_data

        return decompressed_data

    def get_next_message(self) -> MessageEvent | None:
        """Get the next message from the stream.

        Returns:
            MessageEvent: The next message in the stream, or None if stream is exhausted.
        """
        assert self._message_iterator is not None

        try:
            return next(self._message_iterator)
        except (StopIteration, EndOfFileError):
            return None

    def _get_message(self, io: IO[bytes]) -> Iterable[MessageEvent]:
        """Generator to yield messages from the stream."""

        if not self._subscribed_topics_id:
            return iter([])

        inside_chunk = isinstance(io, BytesIO)
        stream_reader = StreamReader(io, skip_magic=True, emit_chunks=True)
        try:
            for record in stream_reader.records:
                if isinstance(record, Message):
                    channel = self.summary.channels.get(record.channel_id)
                    if channel is None:
                        raise ChannelNotFoundError(
                            f"Channel with ID {record.channel_id} not found in summary."
                        )

                    # Skip messages not in subscribed topics
                    if channel.id not in self._subscribed_topics_id:
                        continue

                    schema = self.summary.schemas.get(channel.schema_id)
                    decoded_message = self._decoded_message(schema, channel, record)

                    msg_event = MessageEvent(
                        topic=channel.topic,
                        message=decoded_message,
                        timestamp_ns=record.log_time,
                        schema_name=schema.name if schema else None,
                    )
                    yield msg_event
                elif isinstance(record, Chunk):
                    assert not inside_chunk, "Chunks should not contain chunks"

                    # Use cached decompression with chunk offset as key
                    chunk_offset = io.tell() - len(record.data) - 16  # Approximate chunk position
                    data = self._decompress_chunk(
                        chunk_offset, record.compression, record.data, record.uncompressed_size
                    )

                    # Navigate to the specific message within the chunk
                    assert len(data) == record.uncompressed_size
                    chunk_io = BytesIO(data)
                    with contextlib.suppress(mcap.exceptions.EndOfFile, EndOfFileError):
                        yield from self._get_message(chunk_io)

            # Finished processing all records
        except mcap.exceptions.EndOfFile:
            raise EndOfFileError("Reached end of MCAP file.") from None

    def seek_to_ns(self, timestamp_ns: int) -> None:
        """Seek to the specified timestamp in nanoseconds."""
        assert self._file is not None

        assert timestamp_ns >= self.statistics.message_start_time, (
            f"{timestamp_ns} is before the start time {self.statistics.message_start_time}"
        )
        assert timestamp_ns <= self.statistics.message_end_time, (
            f"{timestamp_ns} is after the end time {self.statistics.message_end_time}"
        )

        chunk = sorted(
            (
                c
                for c in self.summary.chunk_indexes
                if timestamp_ns <= c.message_end_time
                # it can happen that we seek between chunk so we select next chunk
            ),
            key=lambda c: c.chunk_start_offset,
        )

        self._file.seek(chunk[0].chunk_start_offset)
        self._message_iterator = iter(self._get_message(self._file))
