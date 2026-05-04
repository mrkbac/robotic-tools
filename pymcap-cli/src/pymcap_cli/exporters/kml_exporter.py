"""KML exporter — single ``export.kml`` with one ``<Folder>`` per topic.

Tracks use Google's ``gx:Track`` extension so Google Earth animates the path
along time. Per-message points are emitted as Placemarks with ``<TimeStamp>``
elements.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar
from xml.sax.saxutils import escape

from pymcap_cli.exporters._common import prepare_output_file
from pymcap_cli.exporters.base import Ros2Exporter, TopicWriter
from pymcap_cli.exporters.geo_common import (
    GeoMode,
    Sample,
    extract_samples,
    is_no_fix,
    log_time_ns_to_rfc3339,
    schema_is_geographic,
    split_on_gaps,
    stride,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from small_mcap import DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext

logger = logging.getLogger(__name__)


def _coord_string(sample: Sample) -> str:
    """KML wants ``lon,lat[,alt]`` — note the order is lon-then-lat."""
    if sample.alt is None:
        return f"{sample.lon},{sample.lat}"
    return f"{sample.lon},{sample.lat},{sample.alt}"


class _KmlTopicWriter(TopicWriter):
    """Buffers samples; the parent :class:`KmlExporter` serialises on ``finish``."""

    def __init__(self, topic: str, include_no_fix: bool) -> None:
        self.topic = topic
        self.include_no_fix = include_no_fix
        self.samples: list[Sample] = []

    def write(self, msg: DecodedMessage) -> None:
        schema_name = msg.schema.name if msg.schema else ""
        for sample in extract_samples(schema_name, msg.decoded_message, msg.message.log_time):
            if not self.include_no_fix and is_no_fix(sample):
                continue
            self.samples.append(sample)

    def close(self) -> None:
        # Serialisation deferred to KmlExporter.finish — the per-topic writer
        # only owns the samples buffer.
        pass


class KmlExporter(Ros2Exporter):
    """Single-file KML output with a Folder per topic."""

    name: ClassVar[str] = "kml"

    def __init__(
        self,
        *,
        mode: GeoMode = "track+points",
        max_gap_ns: int = 30 * 1_000_000_000,
        stride_n: int = 1,
        include_no_fix: bool = False,
    ) -> None:
        self._mode = mode
        self._max_gap_ns = max_gap_ns
        self._stride_n = stride_n
        self._include_no_fix = include_no_fix
        self._writers: dict[int, _KmlTopicWriter] = {}
        self._force = False

    def accepts(self, schema: Schema | None) -> bool:
        return schema_is_geographic(schema)

    def open_topic(self, ctx: TopicContext) -> _KmlTopicWriter:
        writer = _KmlTopicWriter(ctx.topic, self._include_no_fix)
        self._writers[ctx.writer_key] = writer
        self._force = ctx.force
        return writer

    def finish(
        self,
        output_path: Path,
        counts: Mapping[int, int],  # noqa: ARG002 - driver already reports per-topic counts.
    ) -> None:
        if not self._writers:
            return
        path = prepare_output_file(output_path / "export.kml", force=self._force)
        with path.open("w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            fh.write(
                '<kml xmlns="http://www.opengis.net/kml/2.2" '
                'xmlns:gx="http://www.google.com/kml/ext/2.2">\n'
            )
            fh.write("  <Document>\n")
            fh.write(f"    <name>{escape('pymcap-cli export')}</name>\n")
            for writer in sorted(self._writers.values(), key=lambda w: w.topic):
                self._write_folder(fh, writer.topic, writer)
            fh.write("  </Document>\n")
            fh.write("</kml>\n")
        logger.info(f"Wrote {path}")

    def _write_folder(self, fh: Any, topic: str, writer: _KmlTopicWriter) -> None:
        samples = stride(writer.samples, self._stride_n)
        if not samples:
            return
        segments = split_on_gaps(samples, self._max_gap_ns)

        fh.write("    <Folder>\n")
        fh.write(f"      <name>{escape(topic)}</name>\n")

        if self._mode in ("track", "track+points"):
            for idx, seg in enumerate(s for s in segments if len(s) >= 2):
                self._write_gx_track(fh, topic, seg, idx)

        if self._mode in ("points", "track+points"):
            for sample in samples:
                self._write_point_placemark(fh, sample)

        fh.write("    </Folder>\n")

    def _write_gx_track(self, fh: Any, topic: str, seg: list[Sample], idx: int) -> None:
        fh.write("      <Placemark>\n")
        fh.write(f"        <name>{escape(topic)} track {idx + 1}</name>\n")
        fh.write("        <gx:Track>\n")
        for sample in seg:
            fh.write(f"          <when>{log_time_ns_to_rfc3339(sample.log_time_ns)}</when>\n")
        for sample in seg:
            fh.write(f"          <gx:coord>{_coord_string(sample).replace(',', ' ')}</gx:coord>\n")
        fh.write("        </gx:Track>\n")
        fh.write("      </Placemark>\n")

    def _write_point_placemark(self, fh: Any, sample: Sample) -> None:
        when = log_time_ns_to_rfc3339(sample.log_time_ns)
        fh.write("      <Placemark>\n")
        fh.write(f"        <TimeStamp><when>{when}</when></TimeStamp>\n")
        if sample.properties:
            extra = " ".join(
                f"{escape(str(k))}={escape(str(v))}" for k, v in sample.properties.items()
            )
            fh.write(f"        <description>{escape(extra)}</description>\n")
        fh.write(f"        <Point><coordinates>{_coord_string(sample)}</coordinates></Point>\n")
        fh.write("      </Placemark>\n")
