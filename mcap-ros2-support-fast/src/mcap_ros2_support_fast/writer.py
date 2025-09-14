import time
import types
from io import BufferedWriter
from typing import IO, TYPE_CHECKING, Any

import mcap
from mcap.exceptions import McapError
from mcap.records import Schema
from mcap.well_known import SchemaEncoding
from mcap.writer import CompressionType
from mcap.writer import Writer as McapWriter

from . import __version__
from ._planner import serialize_dynamic

if TYPE_CHECKING:
    from ._plans import EncoderFunction


class McapROS2WriteError(McapError):
    """Raised if a ROS2 message cannot be encoded to CDR with a given schema."""


def _library_identifier() -> str:
    mcap_version = getattr(mcap, "__version__", "<=0.0.10")
    return f"mcap-ros2-support {__version__}; mcap {mcap_version}"


class Writer:
    def __init__(
        self,
        output: str | IO[Any] | BufferedWriter,
        chunk_size: int = 1024 * 1024,
        compression: CompressionType = CompressionType.ZSTD,
        enable_crcs: bool = True,
    ) -> None:
        self._writer = McapWriter(
            output=output,
            chunk_size=chunk_size,
            compression=compression,
            enable_crcs=enable_crcs,
        )
        self._encoders: dict[int, EncoderFunction] = {}
        self._channel_ids: dict[str, int] = {}
        self._writer.start(profile="ros2", library=_library_identifier())
        self._finished = False

    def finish(self) -> None:
        """Finishes writing to the MCAP stream. This must be called before the stream is closed."""
        if not self._finished:
            self._writer.finish()
            self._finished = True

    def register_msgdef(self, datatype: str, msgdef_text: str) -> Schema:
        """Write a Schema record for a ROS2 message definition."""
        msgdef_data = msgdef_text.encode()
        schema_id = self._writer.register_schema(datatype, SchemaEncoding.ROS2, msgdef_data)
        return Schema(id=schema_id, name=datatype, encoding=SchemaEncoding.ROS2, data=msgdef_data)

    def write_message(
        self,
        topic: str,
        schema: Schema,
        message: Any,
        log_time: int | None = None,
        publish_time: int | None = None,
        sequence: int = 0,
    ) -> None:
        """
        Write a ROS2 Message record, automatically registering a channel as needed.

        :param topic: The topic of the message.
        :param message: The message to write.
        :param log_time: The time at which the message was logged as a nanosecond UNIX timestamp.
            Will default to the current time if not specified.
        :param publish_time: The time at which the message was published as a nanosecond UNIX
            timestamp. Will default to ``log_time`` if not specified.
        :param sequence: An optional sequence number.
        """
        encoder = self._encoders.get(schema.id)
        if encoder is None:
            if schema.encoding != SchemaEncoding.ROS2:
                raise McapROS2WriteError(f'can\'t parse schema with encoding "{schema.encoding}"')
            encoder = serialize_dynamic(schema.name, schema.data.decode())
            self._encoders[schema.id] = encoder

        if topic not in self._channel_ids:
            channel_id = self._writer.register_channel(
                topic=topic,
                message_encoding="cdr",
                schema_id=schema.id,
            )
            self._channel_ids[topic] = channel_id
        channel_id = self._channel_ids[topic]

        data = encoder(message)

        if log_time is None:
            log_time = time.time_ns()
        if publish_time is None:
            publish_time = log_time
        self._writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=publish_time,
            sequence=sequence,
            data=data,
        )

    def __enter__(self) -> "Writer":
        """Context manager support."""
        return self

    def __exit__(
        self,
        exc_: type[BaseException] | None,
        exc_type_: BaseException | None,
        tb_: types.TracebackType | None,
    ) -> None:
        """Call finish() on exit."""
        self.finish()
