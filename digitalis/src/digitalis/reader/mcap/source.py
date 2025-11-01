"""MCAP source implementation with playback control."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from typing import IO, TYPE_CHECKING

from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap import (
    EndOfFileError,
    get_summary,
    read_message_decoded,
)

from digitalis.exceptions import InvalidFileFormatError
from digitalis.reader.source import PlaybackSource, SourceStatus
from digitalis.reader.types import MessageEvent, SourceInfo, Topic

if TYPE_CHECKING:
    from small_mcap import Statistics, Summary

logger = logging.getLogger(__name__)


class McapSource(PlaybackSource):
    """MCAP file source with playback control and seeking capabilities."""

    def __init__(
        self,
        path: str,
        on_message: Callable[[MessageEvent], None],
        on_source_info: Callable[[SourceInfo], None],
        on_time: Callable[[int], None],
        on_status: Callable[[SourceStatus], None],
    ) -> None:
        super().__init__(
            on_message=on_message,
            on_source_info=on_source_info,
            on_time=on_time,
            on_status=on_status,
        )
        self.path = path
        self._file = open(path, "rb")  # noqa: PTH123, SIM115
        self._decoder_factory = DecoderFactory()
        self._summary: Summary | None = None
        self._message_iterator: Iterator[MessageEvent] | None = None
        self._stats: Statistics | None = None
        self._channels: dict[int, Topic] = {}
        self._subscribed_topics: set[str] = set()
        self._subscribed_topics_id: set[int] = set()
        self._current_time: int | None = None
        self._playback_speed = 1.0
        self._playback_task: asyncio.Task | None = None
        self._playback_condition = asyncio.Event()
        self._seek_flag = False

        # Status tracking
        self._current_status = SourceStatus.READY

    async def initialize(self) -> SourceInfo:
        """Initialize the MCAP reader and extract metadata."""
        logger.info(f"Initializing MCAP source: {self.path}")

        self._summary = get_summary(self._file)

        if self._summary is None:
            raise InvalidFileFormatError("No summary found in MCAP file.")

        stats = self._summary.statistics
        if stats is None:
            raise InvalidFileFormatError("Statistics not found in MCAP file")
        self._stats = stats

        topics = []

        for channel in self._summary.channels.values():
            schema = self._summary.schemas.get(channel.schema_id)
            topic = Topic(
                name=channel.topic,
                schema_name=schema.name if schema else None,
                message_count=stats.channel_message_counts.get(channel.id),
                topic_id=channel.id,
            )
            topics.append(topic)
            self._channels[channel.id] = topic

        self._current_time = stats.message_start_time
        self._message_iterator = iter(self._get_messages(self._file, stats.message_start_time))

        self._playback_task = asyncio.create_task(self._playback_loop())

        source_info = SourceInfo(
            topics=topics,
            start_time_ns=stats.message_start_time,
            end_time_ns=stats.message_end_time,
        )

        # Notify source info handler
        self._notify_source_info(source_info)

        # File loaded successfully, ensure status is READY
        self._set_status(SourceStatus.READY)

        return source_info

    async def subscribe(self, topic: str) -> None:
        """Subscribe to messages from a topic."""
        if topic not in {t.name for t in self._channels.values()}:
            logger.warning(f"Topic {topic} not found in MCAP file")
            return

        if topic in self._subscribed_topics:
            logger.debug(f"Already subscribed to topic {topic}")
            return

        self._subscribed_topics.add(topic)
        logger.info(f"Subscribed to topic {topic}")

        assert self._summary is not None, "McapReader not initialized"

        self._subscribed_topics_id = {
            channel.id
            for channel in self._summary.channels.values()
            if channel.topic in self._subscribed_topics
        }

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from messages from a topic."""
        if topic not in self._subscribed_topics:
            logger.warning(f"Not subscribed to topic {topic}")
            return

        self._subscribed_topics.remove(topic)
        logger.info(f"Unsubscribed from topic {topic}")
        assert self._summary is not None, "McapReader not initialized"
        self._subscribed_topics_id = {
            channel.id
            for channel in self._summary.channels.values()
            if channel.topic in self._subscribed_topics
        }

    def get_status(self) -> SourceStatus:
        """Get the current source status."""
        return self._current_status

    def _set_status(self, status: SourceStatus) -> None:
        """Update the current status and notify handler."""
        if self._current_status != status:
            self._current_status = status
            logger.debug(f"MCAP source status changed to: {status.value}")
            self._notify_status(status)

    def start_playback(self) -> None:
        """Start or resume playback."""
        self._playback_condition.set()

    def pause_playback(self) -> None:
        """Pause playback."""
        self._playback_condition.clear()

    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed multiplier."""
        self._playback_speed = speed

    async def seek_to_time(self, timestamp_ns: int) -> None:
        """Seek to a specific timestamp."""
        if not self._stats:
            logger.warning("Cannot seek: not initialized")
            return

        # Clamp timestamp to valid range
        timestamp_ns = max(
            self._stats.message_start_time, min(timestamp_ns, self._stats.message_end_time)
        )

        self._current_time = timestamp_ns
        assert self._summary is not None, "McapReader not initialized"
        chunk = sorted(
            (
                c
                for c in self._summary.chunk_indexes
                if timestamp_ns <= c.message_end_time
                # it can happen that we seek between chunk so we select next chunk
            ),
            key=lambda c: c.chunk_start_offset,
        )

        self._file.seek(chunk[0].chunk_start_offset)
        self._message_iterator = iter(self._get_messages(self._file, timestamp_ns))

        # Signal that a seek operation occurred
        self._seek_flag = True

        # Immediately fetch and display the next message at/after this timestamp
        # This provides visual feedback even when playback is paused
        msg = self.get_next_message()
        if msg is not None:
            # Send message if we're subscribed to its topic
            if msg.topic in self._subscribed_topics:
                self._notify_message(msg)

            # Update current time to the actual message timestamp
            self._current_time = msg.timestamp_ns
            self._notify_time(msg.timestamp_ns)

    @property
    def is_playing(self) -> bool:
        """Return True if playback is currently active."""
        return self._playback_condition.is_set()

    @property
    def time_range(self) -> tuple[int, int] | None:
        """Return the start and end time of the messages."""
        if not self._stats:
            return None
        return self._stats.message_start_time, self._stats.message_end_time

    @property
    def current_time(self) -> int | None:
        """Return current playback time."""
        return self._current_time

    async def close(self) -> None:
        """Clean up resources."""
        if self._playback_task:
            self.pause_playback()
            self._playback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._playback_task
        self._file.close()
        logger.info("MCAP source closed")

    async def _playback_loop(self) -> None:
        """Main playback loop."""
        if not self._stats:
            return

        last_msg_time = None
        start_time = time.perf_counter_ns()

        while True:
            # Wait for playback to be enabled
            await self._playback_condition.wait()

            # Reset timing after seek
            if self._seek_flag:
                self._seek_flag = False
                last_msg_time = None
                start_time = time.perf_counter_ns()

            msg = self.get_next_message()
            if msg is None:
                await asyncio.sleep(0.01)
                continue

            # Send message if subscribed
            if msg.topic in self._subscribed_topics:
                self._notify_message(msg)

            # Calculate and perform sleep for realistic playback timing
            if last_msg_time is not None:
                msg_delta = msg.timestamp_ns - last_msg_time
                elapsed = time.perf_counter_ns() - start_time
                target_sleep = (msg_delta / self._playback_speed) - elapsed

                if target_sleep > 0:
                    await self._smooth_sleep(target_sleep / 1_000_000_000)

            # Update state for next iteration
            last_msg_time = msg.timestamp_ns
            self._current_time = msg.timestamp_ns
            self._notify_time(msg.timestamp_ns)
            start_time = time.perf_counter_ns()

    async def _smooth_sleep(self, sleep_seconds: float) -> None:
        """Sleep with interruption checking."""
        await asyncio.sleep(sleep_seconds)

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
            return

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
