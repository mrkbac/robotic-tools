"""High-level base classes for record-oriented data exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pymcap_cli.exporters._common import (
    DEFAULT_BLOB_SCHEMAS,
    message_timestamps_ns,
    normalize_schema_name,
    prepare_output_file,
)
from pymcap_cli.exporters.base import Exporter, TopicContext, Writer
from pymcap_cli.exporters.derived_columns import ColumnSelection, ColumnValue
from pymcap_cli.types.to_plain import to_plain

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from small_mcap import Channel, DecodedMessage, Schema, Summary

PlainScalar = bool | int | float | str | bytes | None
PlainValue = PlainScalar | list["PlainValue"] | dict[str, "PlainValue"]


@dataclass(frozen=True, slots=True)
class StructuredRecord:
    message: DecodedMessage
    columns: dict[str, ColumnValue]
    is_projection: bool

    @property
    def log_time_ns(self) -> int:
        return int(self.message.message.log_time)

    @property
    def publish_time_ns(self) -> int:
        return int(self.message.message.publish_time)

    def timestamp_fields(self) -> dict[str, int]:
        log_time_ns, publish_time_ns = message_timestamps_ns(self.message)
        return {
            "_log_time_ns": log_time_ns,
            "_publish_time_ns": publish_time_ns,
        }

    def plain_payload(self) -> PlainValue:
        return to_plain(self.message.decoded_message)


class _StructuredTopicWriter:
    def __init__(
        self,
        *,
        topic: str,
        columns: ColumnSelection,
        writer: Writer[StructuredRecord],
    ) -> None:
        self._topic = topic
        self._columns = columns
        self._writer = writer

    def write(self, msg: DecodedMessage) -> None:
        self._writer.write(
            StructuredRecord(
                message=msg,
                columns=self._columns.values_for(self._topic, msg),
                is_projection=self._columns.is_enabled,
            )
        )

    def close(self) -> None:
        self._writer.close()


class StructuredExporter(Exporter, ABC):
    """Base for CSV, JSON, Parquet, and other record-oriented formats."""

    def __init__(
        self,
        *,
        include_blobs: bool = False,
        skip_schema: list[str] | None = None,
        select: list[str] | None = None,
    ) -> None:
        self._skipped_schemas = set() if include_blobs else set(DEFAULT_BLOB_SCHEMAS)
        self._skipped_schemas.update(normalize_schema_name(schema) for schema in skip_schema or ())
        self._columns = ColumnSelection(select)

    def accepts(self, channel: Channel, schema: Schema | None) -> bool:
        if not self._columns.includes_topic(channel.topic):
            return False
        return schema is None or normalize_schema_name(schema.name) not in self._skipped_schemas

    def validate_input(self, summary: Summary | None) -> None:
        self._columns.validate_input(summary)

    def open_topic(self, ctx: TopicContext) -> Writer[DecodedMessage]:
        return _StructuredTopicWriter(
            topic=ctx.topic,
            columns=self._columns,
            writer=self.create_writer(ctx),
        )

    @abstractmethod
    def create_writer(self, ctx: TopicContext) -> Writer[StructuredRecord]:
        """Create the format-specific destination for one topic."""


class PerTopicFileExporter(StructuredExporter):
    """Structured exporter that writes one file per input topic."""

    file_suffix: ClassVar[str]
    writer_factory: ClassVar[Callable[[Path], Writer[StructuredRecord]]]

    def create_writer(self, ctx: TopicContext) -> Writer[StructuredRecord]:
        path = prepare_output_file(
            ctx.output_path / f"{ctx.safe_filename}{self.file_suffix}",
            force=ctx.force,
        )
        return self.writer_factory(path)
