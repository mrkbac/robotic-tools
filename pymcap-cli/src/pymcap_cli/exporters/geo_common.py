"""Shared helpers for the geographic exporters (GeoJSON / KML / GPX).

Extracts ``(latitude, longitude, altitude, properties)`` tuples from messages
whose schemas are inherently geographic — primarily ``sensor_msgs/NavSatFix``
and the ``geographic_msgs`` family. Local-frame poses (``Odometry``, ``Pose``)
need a datum + projection and are deliberately out of scope here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pymcap_cli.exporters._common import normalize_schema_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from small_mcap import Schema


GeoMode = Literal["points", "track", "track+points"]


# Canonical (short) schema names — compare via :func:`normalize_schema_name`.
GEO_SCHEMAS: frozenset[str] = frozenset(
    {
        "sensor_msgs/NavSatFix",
        "geographic_msgs/GeoPoint",
        "geographic_msgs/GeoPointStamped",
        "geographic_msgs/GeoPose",
        "geographic_msgs/GeoPoseStamped",
        "geographic_msgs/GeoPath",
    }
)

# NavSatStatus.STATUS_* constants (from sensor_msgs/NavSatStatus.msg).
NAVSAT_STATUS_NAMES: dict[int, str] = {
    -1: "NO_FIX",
    0: "FIX",
    1: "SBAS_FIX",
    2: "GBAS_FIX",
}
NAVSAT_SERVICE_NAMES: dict[int, str] = {
    1: "GPS",
    2: "GLONASS",
    4: "COMPASS",
    8: "GALILEO",
}


@dataclass(slots=True)
class Sample:
    """One geo-located sample. ``alt`` may be ``None`` if the message lacks it."""

    log_time_ns: int
    lat: float
    lon: float
    alt: float | None
    properties: dict[str, Any]


def schema_is_geographic(schema: Schema | None) -> bool:
    return schema is not None and normalize_schema_name(schema.name) in GEO_SCHEMAS


def _decode_navsat_status(raw: int) -> str:
    return NAVSAT_STATUS_NAMES.get(int(raw), f"UNKNOWN({raw})")


def _decode_navsat_service(raw: int) -> str:
    """Bitfield decode of ``NavSatStatus.service`` to a comma-separated name list."""
    flags = int(raw)
    if flags == 0:
        return ""
    parts = [name for bit, name in NAVSAT_SERVICE_NAMES.items() if flags & bit]
    leftover = flags & ~sum(NAVSAT_SERVICE_NAMES)
    if leftover:
        parts.append(f"BIT_{leftover:#x}")
    return "|".join(parts)


def _hdop_from_covariance(cov: Any, cov_type: int) -> float | None:
    """Approximate HDOP from ``position_covariance`` (m²) when available.

    Only meaningful when ``position_covariance_type`` is APPROXIMATED (1),
    DIAGONAL_KNOWN (2) or KNOWN (3). Returns ``sqrt(cov_xx + cov_yy) / 5`` —
    the conventional rough conversion used by GPS receivers, accurate enough
    for filtering / display purposes.
    """
    try:
        if int(cov_type) <= 0:
            return None
        cxx = float(cov[0])
        cyy = float(cov[4])
    except (TypeError, ValueError, IndexError):
        return None
    horiz_sigma_m = math.sqrt(max(cxx, 0.0) + max(cyy, 0.0))
    if horiz_sigma_m <= 0:
        return None
    return round(horiz_sigma_m / 5.0, 3)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute either by attr access or dict key — covers both
    ``__slots__`` and ``__dict__`` decoded message representations."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_navsatfix(decoded: Any) -> Iterator[tuple[float, float, float | None, dict[str, Any]]]:
    lat = float(_get(decoded, "latitude", float("nan")))
    lon = float(_get(decoded, "longitude", float("nan")))
    alt = _get(decoded, "altitude")
    alt_val: float | None
    try:
        alt_val = float(alt) if alt is not None else None
        if alt_val is not None and (math.isnan(alt_val) or math.isinf(alt_val)):
            alt_val = None
    except (TypeError, ValueError):
        alt_val = None

    props: dict[str, Any] = {}
    status = _get(decoded, "status")
    if status is not None:
        status_code = _get(status, "status")
        if status_code is not None:
            props["status"] = _decode_navsat_status(status_code)
            props["status_code"] = int(status_code)
        service = _get(status, "service")
        if service is not None:
            service_name = _decode_navsat_service(service)
            if service_name:
                props["service"] = service_name

    cov = _get(decoded, "position_covariance")
    cov_type = _get(decoded, "position_covariance_type", 0)
    if cov is not None:
        hdop = _hdop_from_covariance(cov, cov_type)
        if hdop is not None:
            props["hdop"] = hdop

    yield lat, lon, alt_val, props


def _extract_geopoint(decoded: Any) -> Iterator[tuple[float, float, float | None, dict[str, Any]]]:
    lat = float(_get(decoded, "latitude", float("nan")))
    lon = float(_get(decoded, "longitude", float("nan")))
    alt = _get(decoded, "altitude")
    try:
        alt_val: float | None = float(alt) if alt is not None else None
    except (TypeError, ValueError):
        alt_val = None
    yield lat, lon, alt_val, {}


def _extract_geopointstamped(
    decoded: Any,
) -> Iterator[tuple[float, float, float | None, dict[str, Any]]]:
    position = _get(decoded, "position", decoded)
    yield from _extract_geopoint(position)


def _extract_geopose(decoded: Any) -> Iterator[tuple[float, float, float | None, dict[str, Any]]]:
    position = _get(decoded, "position")
    if position is None:
        return
    yield from _extract_geopoint(position)


def _extract_geoposestamped(
    decoded: Any,
) -> Iterator[tuple[float, float, float | None, dict[str, Any]]]:
    pose = _get(decoded, "pose", decoded)
    yield from _extract_geopose(pose)


def _extract_geopath(decoded: Any) -> Iterator[tuple[float, float, float | None, dict[str, Any]]]:
    poses = _get(decoded, "poses") or []
    for entry in poses:
        yield from _extract_geoposestamped(entry)


_EXTRACTORS = {
    "sensor_msgs/NavSatFix": _extract_navsatfix,
    "geographic_msgs/GeoPoint": _extract_geopoint,
    "geographic_msgs/GeoPointStamped": _extract_geopointstamped,
    "geographic_msgs/GeoPose": _extract_geopose,
    "geographic_msgs/GeoPoseStamped": _extract_geoposestamped,
    "geographic_msgs/GeoPath": _extract_geopath,
}


def extract_samples(schema_name: str, decoded: Any, log_time_ns: int) -> Iterator[Sample]:
    """Yield one or more :class:`Sample` from a decoded message.

    A ``GeoPath`` produces multiple samples sharing the same ``log_time_ns``;
    every other supported schema produces exactly one.
    """
    extractor = _EXTRACTORS.get(normalize_schema_name(schema_name))
    if extractor is None:
        return
    for lat, lon, alt, props in extractor(decoded):
        if math.isnan(lat) or math.isnan(lon):
            continue
        yield Sample(log_time_ns=log_time_ns, lat=lat, lon=lon, alt=alt, properties=props)


def is_no_fix(sample: Sample) -> bool:
    """True for ``NavSatFix`` samples whose ``status.status`` is ``NO_FIX``."""
    return sample.properties.get("status_code") == -1


def split_on_gaps(samples: Iterable[Sample], max_gap_ns: int) -> list[list[Sample]]:
    """Group samples into contiguous segments separated by gaps > ``max_gap_ns``.

    A non-positive ``max_gap_ns`` disables splitting (single segment returned).
    """
    segments: list[list[Sample]] = []
    current: list[Sample] = []
    for sample in samples:
        if current and max_gap_ns > 0:
            gap = sample.log_time_ns - current[-1].log_time_ns
            if gap > max_gap_ns:
                segments.append(current)
                current = []
        current.append(sample)
    if current:
        segments.append(current)
    return segments


def stride(samples: list[Sample], n: int) -> list[Sample]:
    """Decimate a sample list, always preserving first and last."""
    if n <= 1 or len(samples) <= 2:
        return samples
    out = samples[::n]
    if out[-1] is not samples[-1]:
        out.append(samples[-1])
    return out


def log_time_ns_to_rfc3339(ns: int) -> str:
    """Convert ``log_time`` nanoseconds → RFC3339 UTC (``2026-04-30T12:34:56.789Z``).

    Microsecond precision: KML / GPX consumers don't read sub-µs digits.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    seconds = ns // 1_000_000_000
    micros = (ns // 1_000) % 1_000_000
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=micros)
    if micros:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
