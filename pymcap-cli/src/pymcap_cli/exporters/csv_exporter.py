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
from pymcap_cli.exporters.structured import PerTopicFileExporter, StructuredRecord

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pymcap_cli.exporters.base import Writer


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


class _CsvWriter:
    def __init__(self, path: Path) -> None:
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._fieldnames_set: set[str] = set()

    def write(self, record: StructuredRecord) -> None:
        row: dict[str, Any] = record.timestamp_fields()
        if not record.is_projection:
            plain = record.plain_payload()
            payload = _flatten(plain) if isinstance(plain, dict) else {"value": plain}
            row.update(payload)
        row.update(record.columns)

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


class CsvExporter(PerTopicFileExporter):
    """One CSV file per topic, with dot-flattened columns."""

    name: ClassVar[str] = "csv"
    file_suffix: ClassVar[str] = ".csv"
    writer_factory: ClassVar[Callable[[Path], Writer[StructuredRecord]]] = _CsvWriter
