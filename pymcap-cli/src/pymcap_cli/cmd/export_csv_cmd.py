"""Export an MCAP file to a directory of CSV files (one per topic)."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.csv_exporter import CsvExporter
from pymcap_cli.types.types_manual import (  # noqa: TC001 — runtime for cyclopts
    ForceOverwriteOption,
    OutputPathOption,
)


def export_csv(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    topic: Annotated[list[str] | None, Parameter(name=["--topic", "-t"])] = None,
    include_blobs: Annotated[bool, Parameter(name=["--include-blobs"])] = False,
    num_workers: Annotated[int, Parameter(name=["--num-workers"])] = 8,
) -> int:
    """Export an MCAP file to a directory of CSV files (one per topic).

    Nested message fields are flattened with dot notation
    (``pose.position.x``); arrays are kept as JSON strings to preserve row
    counts. Schemas with raw media payloads (``sensor_msgs/Image``,
    ``CompressedImage`` …) are skipped unless ``--include-blobs`` is passed.
    """
    return run_export(
        file=file,
        output=output,
        exporter=CsvExporter(include_blobs=include_blobs),
        topics=topic,
        force=force,
        num_workers=num_workers,
    )
