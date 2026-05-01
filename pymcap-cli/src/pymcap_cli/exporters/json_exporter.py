"""JSON exporter — NDJSON per topic, or one JSON file per message."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pymcap_cli.exporters._common import (
    SkipSchemaMixin,
    message_timestamps_ns,
    prepare_output_file,
    prepare_topic_dir,
    unique_message_path,
)
from pymcap_cli.exporters.base import Ros2Exporter, TopicWriter
from pymcap_cli.types.to_plain import to_plain

if TYPE_CHECKING:
    from pathlib import Path

    from small_mcap import DecodedMessage

    from pymcap_cli.exporters.base import TopicContext


def _bytes_default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_record(msg: DecodedMessage) -> tuple[int, dict[str, Any]]:
    log_time_ns, publish_time_ns = message_timestamps_ns(msg)
    record = {
        "_log_time_ns": log_time_ns,
        "_publish_time_ns": publish_time_ns,
        "data": to_plain(msg.decoded_message),
    }
    return log_time_ns, record


class _NdjsonWriter(TopicWriter):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("w", encoding="utf-8")

    def write(self, msg: DecodedMessage) -> None:
        _, record = _build_record(msg)
        self._fh.write(json.dumps(record, default=_bytes_default))
        self._fh.write("\n")

    def close(self) -> None:
        self._fh.close()


class _PerMessageWriter(TopicWriter):
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = dir_path
        self._used_counts: dict[int, int] = {}

    def write(self, msg: DecodedMessage) -> None:
        log_time_ns, record = _build_record(msg)
        path = unique_message_path(self.dir_path, log_time_ns, "json", self._used_counts)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, default=_bytes_default)

    def close(self) -> None:
        pass


class JsonExporter(SkipSchemaMixin, Ros2Exporter):
    """JSON exporter. NDJSON per topic by default; ``per_message=True`` writes
    one ``<log_time_ns>.json`` per message under a per-topic directory."""

    name: ClassVar[str] = "json"

    def __init__(self, *, include_blobs: bool = False, per_message: bool = False) -> None:
        self._per_message = per_message
        self._set_skipped_schemas(include_blobs=include_blobs)

    def open_topic(self, ctx: TopicContext) -> _NdjsonWriter | _PerMessageWriter:
        if self._per_message:
            dir_path = prepare_topic_dir(ctx.output_path / ctx.safe_filename, force=ctx.force)
            return _PerMessageWriter(dir_path)
        path = prepare_output_file(ctx.output_path / f"{ctx.safe_filename}.ndjson", force=ctx.force)
        return _NdjsonWriter(path)
