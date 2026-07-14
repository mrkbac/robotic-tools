"""Export geographic content (NavSatFix / geographic_msgs) to GeoJSON / KML / GPX."""

import logging
from typing import Annotated, Literal

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.cmd._cli_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    ForceOverwriteOption,
    NumWorkersOption,
    OutputPathOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.geojson_exporter import GeoJsonExporter
from pymcap_cli.exporters.gpx_exporter import GpxExporter
from pymcap_cli.exporters.kml_exporter import KmlExporter
from pymcap_cli.types.duration import parse_duration_ns

logger = logging.getLogger(__name__)
console = Console()


def export_geo(
    file: str,
    output: OutputPathOption,
    *,
    format: Annotated[Literal["geojson", "kml", "gpx"], Parameter(name=["--format"])] = "geojson",
    mode: Annotated[
        Literal["points", "track", "track+points"], Parameter(name=["--mode"])
    ] = "track+points",
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    max_gap: Annotated[str, Parameter(name=["--max-gap"])] = "30s",
    stride_n: Annotated[int, Parameter(name=["--stride"])] = 1,
    include_no_fix: Annotated[bool, Parameter(name=["--include-no-fix"])] = False,
    force: ForceOverwriteOption = False,
    num_workers: NumWorkersOption = 8,
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
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
        max_gap_ns = parse_duration_ns(max_gap)
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
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
        message_filter=message_filter,
        force=force,
        num_workers=num_workers,
    )
