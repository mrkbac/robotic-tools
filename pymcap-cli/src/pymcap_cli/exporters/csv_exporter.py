"""CSV exporter — one ``<topic>.csv`` per topic.

Nested ROS message fields are flattened with dot-notation
(``pose.position.x``). Lists become a single column with their JSON repr —
that keeps row counts honest and CSVs trivially round-trippable in pandas
via ``ast.literal_eval`` or ``json.loads``.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any, ClassVar

from pymcap_cli.display.message_render import format_bytes_skip
from pymcap_cli.exporters._common import (
    SkipSchemaMixin,
    message_timestamps_ns,
    prepare_output_file,
)
from pymcap_cli.exporters.base import Ros2Exporter, TopicWriter
from pymcap_cli.types.to_plain import to_plain

if TYPE_CHECKING:
    from pathlib import Path

    from small_mcap import DecodedMessage

    from pymcap_cli.exporters.base import TopicContext


def _flatten(value: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten a nested dict into ``{"a.b.c": v}`` form. Lists kept as JSON."""
    if out is None:
        out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten(v, key, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value, default=str)
    elif isinstance(value, (bytes, bytearray)):
        out[prefix] = format_bytes_skip(value)
    else:
        out[prefix] = value
    return out


class _CsvWriter(TopicWriter):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._fieldnames_set: set[str] = set()

    def write(self, msg: DecodedMessage) -> None:
        plain = to_plain(msg.decoded_message)
        log_time_ns, publish_time_ns = message_timestamps_ns(msg)
        row: dict[str, Any] = {
            "_log_time_ns": log_time_ns,
            "_publish_time_ns": publish_time_ns,
        }
        if isinstance(plain, dict):
            row.update(_flatten(plain))
        else:
            row["value"] = plain

        if self._writer is None:
            fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
            self._fieldnames_set = set(fieldnames)
            self._writer.writeheader()
        else:
            missing = [k for k in row if k not in self._fieldnames_set]
            if missing:
                self._writer.fieldnames = [*self._writer.fieldnames, *missing]
                self._fieldnames_set.update(missing)
        self._writer.writerow(row)

    def close(self) -> None:
        self._fh.close()


class CsvExporter(SkipSchemaMixin, Ros2Exporter):
    """One CSV file per topic, with dot-flattened columns."""

    name: ClassVar[str] = "csv"

    def __init__(self, *, include_blobs: bool = False) -> None:
        self._set_skipped_schemas(include_blobs=include_blobs)

    def open_topic(self, ctx: TopicContext) -> _CsvWriter:
        path = prepare_output_file(ctx.output_path / f"{ctx.safe_filename}.csv", force=ctx.force)
        return _CsvWriter(path)
