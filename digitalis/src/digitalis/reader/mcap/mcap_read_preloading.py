from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap.reader import (
    EndOfFileError,
    get_summary,
    read_message_decoded,
)

from digitalis.exceptions import InvalidFileFormatError
from digitalis.reader.types import MessageEvent


class McapReaderPreloading:
    def __init__(
        self,
        file_path: str | Path,
    ) -> None:
        self._decoder_factory = DecoderFactory()
        with Path(file_path).open("rb") as file:
            summary = get_summary(file)
            if summary is None:
                raise InvalidFileFormatError("No summary found in MCAP file.")
            self.summary = summary
            assert self.summary.statistics
            self.statistics = self.summary.statistics

        self._file_path: Path = Path(file_path)
        self._subscribed_topics_id: set[int] = set()
        self._message_iterator: Iterator[MessageEvent] | None = None

        self._file: IO[bytes] = self._file_path.open("rb")
        self._message_iterator = iter(self._get_messages(self._file, 0))
        self._file.seek(8)  # Skip magic bytes

    def set_subscription(self, topics: list[str]) -> None:
        """Set the topics to subscribe to."""
        self._subscribed_topics_id = {
            channel.id for channel in self.summary.channels.values() if channel.topic in topics
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._file and not self._file.closed:
            self._file.close()
        self._message_iterator = None

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

    def _get_messages(self, io: IO[bytes], timestamp_ns: int) -> Iterable[MessageEvent]:
        """Generator to yield messages from the stream."""

        if not self._subscribed_topics_id:
            return iter([])

        for msg in read_message_decoded(
            io,
            lambda channel, _schema: channel.id in self._subscribed_topics_id,
            decoder_factories=[self._decoder_factory],
            start_time_ns=timestamp_ns,
        ):
            msg_event = MessageEvent(
                topic=msg.channel.topic,
                message=msg.decoded_message,
                timestamp_ns=msg.message.log_time,
                schema_name=msg.schema.name if msg.schema else None,
            )
            yield msg_event

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
        self._message_iterator = iter(self._get_messages(self._file, timestamp_ns))
