"""End-to-end tests for the GeoJSON / KML / GPX exporters."""

from __future__ import annotations

import io
import json
import math
import struct
import subprocess
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.geo_common import (
    Sample,
    extract_samples,
    log_time_ns_to_rfc3339,
    split_on_gaps,
    stride,
)
from pymcap_cli.exporters.geojson_exporter import GeoJsonExporter
from pymcap_cli.exporters.gpx_exporter import GpxExporter
from pymcap_cli.exporters.kml_exporter import KmlExporter
from pymcap_cli.utils import NS_TO_SEC
from small_mcap import CompressionType, McapWriter

if TYPE_CHECKING:
    from pathlib import Path

# NavSatFix CDR layout (after the 4-byte encapsulation header):
#   header.stamp.sec (int32) + header.stamp.nanosec (uint32)
#   header.frame_id (CDR string with trailing nul)
#   status.status (int8) + 3-byte align + status.service (uint16) + 2-byte align
#   latitude (float64) + longitude (float64) + altitude (float64)
#   position_covariance[9] (9 x float64)
#   position_covariance_type (uint8)

_NAVSATFIX_SCHEMA = b"""# Navigation Satellite fix for any Global Navigation Satellite System.
std_msgs/Header header
sensor_msgs/NavSatStatus status
float64 latitude
float64 longitude
float64 altitude
float64[9] position_covariance
uint8 position_covariance_type

uint8 COVARIANCE_TYPE_UNKNOWN=0
uint8 COVARIANCE_TYPE_APPROXIMATED=1
uint8 COVARIANCE_TYPE_DIAGONAL_KNOWN=2
uint8 COVARIANCE_TYPE_KNOWN=3

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: sensor_msgs/NavSatStatus
int8 status
uint16 service

int8 STATUS_NO_FIX=-1
int8 STATUS_FIX=0
int8 STATUS_SBAS_FIX=1
int8 STATUS_GBAS_FIX=2

uint16 SERVICE_GPS=1
"""


def _align(buf: bytearray, alignment: int, *, header_len: int = 4) -> None:
    """CDR primitives align relative to the start of the encapsulated payload.

    ``header_len`` is the size of the encapsulation header preceding the
    payload (4 bytes for ``\\x00\\x01\\x00\\x00``).
    """
    payload_offset = len(buf) - header_len
    pad = (alignment - payload_offset % alignment) % alignment
    buf += b"\x00" * pad


def _encode_navsatfix(
    *, sec: int, nanosec: int, lat: float, lon: float, alt: float, status: int
) -> bytes:
    """Hand-roll a CDR-encoded NavSatFix payload."""
    buf = bytearray()
    buf += b"\x00\x01\x00\x00"  # CDR encapsulation header (little-endian)
    # std_msgs/Header.stamp (int32 + uint32, both align 4 — already aligned)
    buf += struct.pack("<iI", sec, nanosec)
    # frame_id "" — uint32 length=1 (just the nul terminator)
    _align(buf, 4)
    buf += struct.pack("<I", 1) + b"\x00"
    # NavSatStatus.status (int8, align 1 — no pad needed)
    buf += struct.pack("<b", status)
    # NavSatStatus.service (uint16, align 2)
    _align(buf, 2)
    buf += struct.pack("<H", 1)  # SERVICE_GPS
    # latitude / longitude / altitude (float64, align 8)
    _align(buf, 8)
    buf += struct.pack("<ddd", lat, lon, alt)
    # position_covariance[9] — already 8-aligned
    buf += struct.pack("<9d", 4.0, 0, 0, 0, 4.0, 0, 0, 0, 25.0)
    # position_covariance_type (uint8) — APPROXIMATED so HDOP gets computed
    buf += struct.pack("<B", 1)
    return bytes(buf)


def _make_navsatfix_mcap(path: Path, samples: list[tuple[int, float, float, float, int]]) -> None:
    """Write a ROS2 MCAP with a single ``/gps/fix`` topic.

    ``samples``: list of ``(log_time_ns, lat, lon, alt, status)`` tuples.
    """
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=64 * 1024, compression=CompressionType.NONE)
    writer.start(profile="ros2", library="pymcap-cli-tests")
    writer.add_schema(
        schema_id=1,
        name="sensor_msgs/msg/NavSatFix",
        encoding="ros2msg",
        data=_NAVSATFIX_SCHEMA,
    )
    writer.add_channel(channel_id=1, topic="/gps/fix", message_encoding="cdr", schema_id=1)
    for log_time_ns, lat, lon, alt, status in samples:
        sec = log_time_ns // NS_TO_SEC
        nanosec = log_time_ns % NS_TO_SEC
        writer.add_message(
            channel_id=1,
            log_time=log_time_ns,
            publish_time=log_time_ns,
            data=_encode_navsatfix(
                sec=sec, nanosec=nanosec, lat=lat, lon=lon, alt=alt, status=status
            ),
        )
    writer.finish()
    path.write_bytes(buf.getvalue())


# Walk a small east-by-north line near 47.5°N, 8.5°E — Zürich-ish.
_BASE_TIME_NS = 1_700_000_000_000_000_000


def _walking_samples(n: int = 5, status: int = 0) -> list[tuple[int, float, float, float, int]]:
    return [
        (
            _BASE_TIME_NS + i * NS_TO_SEC,
            47.5 + i * 0.0001,
            8.5 + i * 0.0001,
            500.0 + i,
            status,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests for the geo_common helpers
# ---------------------------------------------------------------------------


def test_extract_navsatfix_yields_one_sample():
    schema_name = "sensor_msgs/msg/NavSatFix"

    class _Status:
        status = 0
        service = 1

    class _Msg:
        latitude = 47.5
        longitude = 8.5
        altitude = 500.0
        status = _Status()
        position_covariance = (4.0, 0, 0, 0, 4.0, 0, 0, 0, 25.0)
        position_covariance_type = 1  # APPROXIMATED

    samples = list(extract_samples(schema_name, _Msg(), 1234))
    assert len(samples) == 1
    s = samples[0]
    assert s.lat == 47.5
    assert s.lon == 8.5
    assert s.alt == 500.0
    assert s.properties["status"] == "FIX"
    assert s.properties["service"] == "GPS"
    assert "hdop" in s.properties


def test_extract_drops_nan_coords():
    class _Msg:
        latitude = float("nan")
        longitude = 8.5
        altitude = 0.0
        status = None
        position_covariance = None
        position_covariance_type = 0

    assert list(extract_samples("sensor_msgs/msg/NavSatFix", _Msg(), 1)) == []


def test_split_on_gaps():
    base = NS_TO_SEC
    samples = [
        Sample(base, 0, 0, None, {}),
        Sample(base + NS_TO_SEC, 0, 0, None, {}),
        # 60-second gap
        Sample(base + 61 * NS_TO_SEC, 0, 0, None, {}),
        Sample(base + 62 * NS_TO_SEC, 0, 0, None, {}),
    ]
    segments = split_on_gaps(samples, max_gap_ns=30 * NS_TO_SEC)
    assert len(segments) == 2
    assert len(segments[0]) == 2
    assert len(segments[1]) == 2


def test_stride_preserves_endpoints():
    samples = [Sample(i, i, i, None, {}) for i in range(10)]
    out = stride(samples, 3)
    assert out[0] is samples[0]
    assert out[-1] is samples[-1]


def test_log_time_to_rfc3339_round():
    iso = log_time_ns_to_rfc3339(0)
    assert iso == "1970-01-01T00:00:00Z"
    iso = log_time_ns_to_rfc3339(1_500_000_000)  # 1.5s
    assert iso.startswith("1970-01-01T00:00:01")


# ---------------------------------------------------------------------------
# End-to-end: each format on a synthetic NavSatFix bag
# ---------------------------------------------------------------------------


def test_geojson_exporter_round_trips_5_points(tmp_path):
    src = tmp_path / "src.mcap"
    _make_navsatfix_mcap(src, _walking_samples(5))

    out = tmp_path / "out"
    rc = run_export(
        file=str(src),
        output=out,
        exporter=GeoJsonExporter(mode="track+points"),
    )
    assert rc == 0

    files = list(out.glob("*.geojson"))
    assert len(files) == 1
    fc = json.loads(files[0].read_text())

    assert fc["type"] == "FeatureCollection"
    point_features = [f for f in fc["features"] if f["geometry"]["type"] == "Point"]
    line_features = [f for f in fc["features"] if f["geometry"]["type"] == "LineString"]
    assert len(point_features) == 5
    assert len(line_features) == 1
    coords = line_features[0]["geometry"]["coordinates"]
    assert len(coords) == 5
    # GeoJSON: [lon, lat, alt]
    assert coords[0][0] == pytest.approx(8.5)
    assert coords[0][1] == pytest.approx(47.5)


def test_export_geo_short_force_flag_keeps_format_long_only(tmp_path):
    src = tmp_path / "src.mcap"
    _make_navsatfix_mcap(src, _walking_samples(2))
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.geojson").write_text("stale", encoding="utf-8")

    result = subprocess.run(
        ["pymcap-cli", "export-geo", str(src), str(out), "-f", "--format", "geojson"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    generated = out / "gps_fix.geojson"
    assert generated.exists()
    assert generated.read_text(encoding="utf-8") != "stale"


def test_geojson_drops_no_fix_by_default(tmp_path):
    src = tmp_path / "src.mcap"
    samples = [
        *_walking_samples(3, status=0),
        (_BASE_TIME_NS + 4_000_000_000, 0.0, 0.0, 0.0, -1),  # NO_FIX, sentinel coords
    ]
    _make_navsatfix_mcap(src, samples)

    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=GeoJsonExporter(mode="points"))
    assert rc == 0

    fc = json.loads(next(out.glob("*.geojson")).read_text())
    assert len(fc["features"]) == 3  # NO_FIX dropped


def test_geojson_include_no_fix_keeps_them(tmp_path):
    src = tmp_path / "src.mcap"
    samples = [
        *_walking_samples(2, status=0),
        (_BASE_TIME_NS + 3_000_000_000, 0.0, 0.0, 0.0, -1),
    ]
    _make_navsatfix_mcap(src, samples)

    out = tmp_path / "out"
    rc = run_export(
        file=str(src),
        output=out,
        exporter=GeoJsonExporter(mode="points", include_no_fix=True),
    )
    assert rc == 0
    fc = json.loads(next(out.glob("*.geojson")).read_text())
    assert len(fc["features"]) == 3


def test_geojson_max_gap_produces_multilinestring(tmp_path):
    src = tmp_path / "src.mcap"
    samples = [
        *_walking_samples(2),
        (_BASE_TIME_NS + 100 * NS_TO_SEC, 47.6, 8.6, 510.0, 0),  # 99s gap
        (_BASE_TIME_NS + 101 * NS_TO_SEC, 47.6001, 8.6001, 511.0, 0),
    ]
    _make_navsatfix_mcap(src, samples)

    out = tmp_path / "out"
    rc = run_export(
        file=str(src),
        output=out,
        exporter=GeoJsonExporter(mode="track", max_gap_ns=30 * NS_TO_SEC),
    )
    assert rc == 0
    fc = json.loads(next(out.glob("*.geojson")).read_text())
    line_feature = next(f for f in fc["features"] if f["geometry"]["type"] == "MultiLineString")
    assert len(line_feature["geometry"]["coordinates"]) == 2


def test_kml_exporter_produces_single_file_with_track(tmp_path):
    src = tmp_path / "src.mcap"
    _make_navsatfix_mcap(src, _walking_samples(4))

    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=KmlExporter(mode="track+points"))
    assert rc == 0

    kml_path = out / "export.kml"
    assert kml_path.exists()

    # Parse the KML and find the track + placemark counts.
    tree = ET.parse(kml_path)  # noqa: S314 — file we just wrote
    ns = {
        "kml": "http://www.opengis.net/kml/2.2",
        "gx": "http://www.google.com/kml/ext/2.2",
    }
    folders = tree.findall(".//kml:Folder", ns)
    assert len(folders) == 1
    placemarks = folders[0].findall("kml:Placemark", ns)
    # 1 track placemark + 4 point placemarks
    assert len(placemarks) == 5
    when_elements = folders[0].findall(".//gx:Track/kml:when", ns)
    assert len(when_elements) == 4
    coord_elements = folders[0].findall(".//gx:Track/gx:coord", ns)
    assert len(coord_elements) == 4


def test_gpx_exporter_validates_as_xml(tmp_path):
    src = tmp_path / "src.mcap"
    _make_navsatfix_mcap(src, _walking_samples(4))

    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=GpxExporter(mode="track+points"))
    assert rc == 0

    gpx_path = out / "export.gpx"
    tree = ET.parse(gpx_path)  # noqa: S314 — file we just wrote
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    trkpts = tree.findall(".//gpx:trkpt", ns)
    assert len(trkpts) == 4
    wpts = tree.findall(".//gpx:wpt", ns)
    assert len(wpts) == 4

    first = trkpts[0]
    assert math.isclose(float(first.attrib["lat"]), 47.5, abs_tol=1e-6)
    assert math.isclose(float(first.attrib["lon"]), 8.5, abs_tol=1e-6)
    ele = first.find("gpx:ele", ns)
    assert ele is not None
    assert math.isclose(float(ele.text or ""), 500.0)


def test_unknown_schema_topic_is_skipped(tmp_path):
    """Non-geographic topic should be skipped without errors."""
    schema = b"""float64 x
"""

    src = tmp_path / "src.mcap"
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=64 * 1024, compression=CompressionType.NONE)
    writer.start(profile="ros2", library="pymcap-cli-tests")
    writer.add_schema(schema_id=1, name="test_msgs/Float", encoding="ros2msg", data=schema)
    writer.add_channel(channel_id=1, topic="/foo", message_encoding="cdr", schema_id=1)
    writer.add_message(
        channel_id=1,
        log_time=0,
        publish_time=0,
        data=b"\x00\x01\x00\x00" + struct.pack("<d", 1.0),
    )
    writer.finish()
    src.write_bytes(buf.getvalue())

    out = tmp_path / "out"
    # Driver returns 1 when no writers were created — that's the contract.
    rc = run_export(file=str(src), output=out, exporter=GeoJsonExporter())
    assert rc == 1
