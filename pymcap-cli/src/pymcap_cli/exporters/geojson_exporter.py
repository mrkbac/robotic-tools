"""GeoJSON exporter — one ``<safe_topic>.geojson`` per topic.

Each topic is one ``FeatureCollection``. Depending on ``mode``:

* ``points`` — every sample becomes a ``Point`` ``Feature`` with timestamps in
  properties.
* ``track`` — one ``LineString`` ``Feature`` (or ``MultiLineString`` when
  splitting on gaps) carrying the trajectory.
* ``track+points`` — both, in the same FeatureCollection.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pymcap_cli.exporters._common import prepare_output_file
from pymcap_cli.exporters.geo_common import (
    GeoExporter,
    GeoMode,
    GeoSampleTopicWriter,
    Sample,
    log_time_ns_to_rfc3339,
    split_on_gaps,
    stride,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymcap_cli.exporters.base import TopicContext


def _coords(sample: Sample) -> list[float]:
    """GeoJSON wants [lon, lat[, alt]] order."""
    if sample.alt is None:
        return [sample.lon, sample.lat]
    return [sample.lon, sample.lat, sample.alt]


class _GeoJsonTopicWriter(GeoSampleTopicWriter):
    """Buffers samples in memory, flushes one ``FeatureCollection`` on close."""

    def __init__(
        self,
        path: Path,
        topic: str,
        *,
        mode: GeoMode,
        max_gap_ns: int,
        stride_n: int,
        include_no_fix: bool,
    ) -> None:
        super().__init__(topic=topic, include_no_fix=include_no_fix)
        self.path = path
        self.mode = mode
        self.max_gap_ns = max_gap_ns
        self.stride_n = stride_n

    def close(self) -> None:
        samples = stride(self.samples, self.stride_n)
        segments = split_on_gaps(samples, self.max_gap_ns)
        features: list[dict[str, Any]] = []

        if self.mode in ("points", "track+points"):
            features.extend(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": _coords(sample)},
                    "properties": {
                        "topic": self.topic,
                        "log_time_ns": sample.log_time_ns,
                        "time": log_time_ns_to_rfc3339(sample.log_time_ns),
                        **sample.properties,
                    },
                }
                for sample in samples
            )

        if self.mode in ("track", "track+points"):
            line_segments = [seg for seg in segments if len(seg) >= 2]
            if line_segments:
                if len(line_segments) == 1:
                    geometry = {
                        "type": "LineString",
                        "coordinates": [_coords(s) for s in line_segments[0]],
                    }
                else:
                    geometry = {
                        "type": "MultiLineString",
                        "coordinates": [[_coords(s) for s in seg] for seg in line_segments],
                    }
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            "topic": self.topic,
                            "kind": "track",
                            "segment_count": len(line_segments),
                            "point_count": sum(len(s) for s in line_segments),
                        },
                    }
                )

        fc = {"type": "FeatureCollection", "features": features}
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(fc, fh)


class GeoJsonExporter(GeoExporter):
    """Per-topic GeoJSON ``FeatureCollection`` files."""

    name: ClassVar[str] = "geojson"

    def open_topic(self, ctx: TopicContext) -> _GeoJsonTopicWriter:
        path = prepare_output_file(
            ctx.output_path / f"{ctx.safe_filename}.geojson",
            force=ctx.force,
        )
        return _GeoJsonTopicWriter(
            path,
            topic=ctx.topic,
            mode=self._mode,
            max_gap_ns=self._max_gap_ns,
            stride_n=self._stride_n,
            include_no_fix=self._include_no_fix,
        )
