"""Export geographic content (NavSatFix / geographic_msgs) to GeoJSON / KML / GPX."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.geojson_exporter import GeoJsonExporter
from pymcap_cli.exporters.gpx_exporter import GpxExporter
from pymcap_cli.exporters.kml_exporter import KmlExporter
from pymcap_cli.types.duration import parse_duration_ns
from pymcap_cli.types.types_manual import (  # noqa: TC001 — runtime for cyclopts
    ForceOverwriteOption,
    OutputPathOption,
)

console = Console()


def export_geo(
    file: str,
    output: OutputPathOption,
    *,
    format: Annotated[Literal["geojson", "kml", "gpx"], Parameter(name=["--format"])] = "geojson",
    mode: Annotated[
        Literal["points", "track", "track+points"], Parameter(name=["--mode"])
    ] = "track+points",
    topic: Annotated[list[str] | None, Parameter(name=["--topic", "-t"])] = None,
    max_gap: Annotated[str, Parameter(name=["--max-gap"])] = "30s",
    stride_n: Annotated[int, Parameter(name=["--stride"])] = 1,
    include_no_fix: Annotated[bool, Parameter(name=["--include-no-fix"])] = False,
    force: ForceOverwriteOption = False,
    num_workers: Annotated[int, Parameter(name=["--num-workers"])] = 8,
) -> int:
    """Export geographic topics (``NavSatFix``, ``geographic_msgs/*``) to a map format.

    GeoJSON writes one ``<topic>.geojson`` per topic.
    KML and GPX produce a single ``export.{kml,gpx}`` covering all topics.

    Local-frame poses (``Odometry`` / ``geometry_msgs/Pose*``) are out of
    scope: they need a datum to be georeferenced.

    Examples:
      mcap export-geo bag.mcap ./out --format kml
      mcap export-geo bag.mcap ./out --format gpx --mode track --stride 5
      mcap export-geo bag.mcap ./out --topic /gps/fix --include-no-fix

    Parameters
    ----------
    file
        Input MCAP file (local path or HTTP/HTTPS URL).
    output
        Output directory.
    format
        Output format: ``geojson`` (default), ``kml``, or ``gpx``.
    mode
        ``points`` for one feature per message, ``track`` for a single
        polyline per topic, ``track+points`` for both (default).
    topic
        Topics to include. If omitted, every supported topic is exported.
    max_gap
        Auto-split tracks when no sample is seen for this duration.
        Default ``30s``; ``0`` disables splitting.
    stride_n
        Keep every Nth sample (always preserves first and last). Default 1.
    include_no_fix
        By default, ``NavSatFix`` messages with ``status.status == NO_FIX``
        (-1) are dropped — set this to keep them.
    force
        Overwrite existing files in the output directory.
    num_workers
        MCAP chunk-decompression worker threads.
    """
    try:
        max_gap_ns = parse_duration_ns(max_gap)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    if format == "kml":
        exporter = KmlExporter(
            mode=mode,
            max_gap_ns=max_gap_ns,
            stride_n=stride_n,
            include_no_fix=include_no_fix,
        )
    elif format == "gpx":
        exporter = GpxExporter(
            mode=mode,
            max_gap_ns=max_gap_ns,
            stride_n=stride_n,
            include_no_fix=include_no_fix,
        )
    else:
        exporter = GeoJsonExporter(
            mode=mode,
            max_gap_ns=max_gap_ns,
            stride_n=stride_n,
            include_no_fix=include_no_fix,
        )

    return run_export(
        file=file,
        output=output,
        exporter=exporter,
        topics=topic,
        force=force,
        num_workers=num_workers,
        console=console,
    )
