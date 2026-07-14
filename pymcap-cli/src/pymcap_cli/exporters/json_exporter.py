"""JSON exporter — NDJSON per topic, or one JSON file per message."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pymcap_cli.display.message_render import format_bytes_skip
from pymcap_cli.exporters._common import prepare_topic_dir, unique_message_path
from pymcap_cli.exporters.structured import PerTopicFileExporter, StructuredRecord

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pymcap_cli.exporters.base import TopicContext, Writer


def _bytes_default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return format_bytes_skip(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_record(source: StructuredRecord) -> dict[str, Any]:
    record: dict[str, Any] = source.timestamp_fields()
    if not source.is_projection:
        record["data"] = source.plain_payload()
    record.update(source.columns)
    return record


class _NdjsonWriter:
    def __init__(self, path: Path) -> None:
        self._fh = path.open("w", encoding="utf-8")

    def write(self, record: StructuredRecord) -> None:
        self._fh.write(json.dumps(_build_record(record), default=_bytes_default))
        self._fh.write("\n")

    def close(self) -> None:
        self._fh.close()


class _PerMessageWriter:
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = dir_path
        self._used_counts: dict[int, int] = {}

    def write(self, record: StructuredRecord) -> None:
        path = unique_message_path(
            self.dir_path,
            record.log_time_ns,
            "json",
            self._used_counts,
        )
        with path.open("w", encoding="utf-8") as f:
            json.dump(_build_record(record), f, default=_bytes_default)

    def close(self) -> None:
        pass


class JsonExporter(PerTopicFileExporter):
    """JSON exporter. NDJSON per topic by default; ``per_message=True`` writes
    one ``<log_time_ns>.json`` per message under a per-topic directory."""

    name: ClassVar[str] = "json"
    file_suffix: ClassVar[str] = ".ndjson"
    writer_factory: ClassVar[Callable[[Path], Writer[StructuredRecord]]] = _NdjsonWriter

    def __init__(
        self,
        *,
        include_blobs: bool = False,
        per_message: bool = False,
        select: list[str] | None = None,
    ) -> None:
        super().__init__(
            include_blobs=include_blobs,
            select=select,
        )
        self._per_message = per_message

    def create_writer(self, ctx: TopicContext) -> Writer[StructuredRecord]:
        if self._per_message:
            dir_path = prepare_topic_dir(ctx.output_path / ctx.safe_filename, force=ctx.force)
            return _PerMessageWriter(dir_path)
        return super().create_writer(ctx)
