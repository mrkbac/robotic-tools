"""GPX exporter — single ``export.gpx`` with one ``<trk>`` per topic.

GPX 1.1 is the format consumed by JOSM, GPSBabel, Strava, Garmin Connect, etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, TextIO
from xml.sax.saxutils import escape

from pymcap_cli.exporters._common import prepare_output_file
from pymcap_cli.exporters.geo_common import (
    GeoSampleTopicWriter,
    Sample,
    SingleFileGeoExporter,
    log_time_ns_to_rfc3339,
    split_on_gaps,
    stride,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)


class GpxExporter(SingleFileGeoExporter):
    """Single-file GPX 1.1 output with a ``<trk>`` per topic."""

    name: ClassVar[str] = "gpx"

    def finish(
        self,
        output_path: Path,
        counts: Mapping[int, int],  # noqa: ARG002 - driver already reports per-topic counts.
    ) -> None:
        if not self._writers:
            return
        path = prepare_output_file(output_path / "export.gpx", force=self._force)
        with path.open("w", encoding="utf-8") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            fh.write(
                '<gpx version="1.1" creator="pymcap-cli" '
                'xmlns="http://www.topografix.com/GPX/1/1">\n'
            )
            for writer in sorted(self._writers.values(), key=lambda w: w.topic):
                self._write_topic(fh, writer.topic, writer)
            fh.write("</gpx>\n")
        logger.info(f"Wrote {path}")

    def _write_topic(self, fh: TextIO, topic: str, writer: GeoSampleTopicWriter) -> None:
        samples = stride(writer.samples, self._stride_n)
        if not samples:
            return
        segments = split_on_gaps(samples, self._max_gap_ns)

        if self._mode in ("points", "track+points"):
            for sample in samples:
                self._write_wpt(fh, sample, topic)

        if self._mode in ("track", "track+points"):
            line_segments = [seg for seg in segments if len(seg) >= 2]
            if line_segments:
                fh.write("  <trk>\n")
                fh.write(f"    <name>{escape(topic)}</name>\n")
                for seg in line_segments:
                    fh.write("    <trkseg>\n")
                    for sample in seg:
                        self._write_pt(fh, sample, "trkpt", indent="      ")
                    fh.write("    </trkseg>\n")
                fh.write("  </trk>\n")

    def _write_wpt(self, fh: TextIO, sample: Sample, topic: str) -> None:
        self._write_pt(fh, sample, "wpt", indent="  ", source_topic=topic)

    def _write_pt(
        self,
        fh: TextIO,
        sample: Sample,
        tag: str,
        *,
        indent: str,
        source_topic: str | None = None,
    ) -> None:
        fh.write(f'{indent}<{tag} lat="{sample.lat}" lon="{sample.lon}">\n')
        if sample.alt is not None:
            fh.write(f"{indent}  <ele>{sample.alt}</ele>\n")
        fh.write(f"{indent}  <time>{log_time_ns_to_rfc3339(sample.log_time_ns)}</time>\n")
        if source_topic:
            fh.write(f"{indent}  <src>{escape(source_topic)}</src>\n")
        fh.write(f"{indent}</{tag}>\n")
