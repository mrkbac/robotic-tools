"""Export an MCAP file to a directory of NDJSON / JSON files (one per topic)."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.json_exporter import JsonExporter
from pymcap_cli.types.types_manual import (  # noqa: TC001 — runtime for cyclopts
    ForceOverwriteOption,
    OutputPathOption,
)


def export_json(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    topic: Annotated[list[str] | None, Parameter(name=["--topic", "-t"])] = None,
    include_blobs: Annotated[bool, Parameter(name=["--include-blobs"])] = False,
    per_message: Annotated[bool, Parameter(name=["--per-message"])] = False,
    num_workers: Annotated[int, Parameter(name=["--num-workers"])] = 8,
) -> int:
    """Export an MCAP file to NDJSON (one line per message) or per-message JSON files.

    Default is one ``<topic>.ndjson`` per topic. With ``--per-message``, each
    topic gets a directory containing one ``<log_time_ns>.json`` per message —
    handy when downstream tools expect one record per file.
    """
    return run_export(
        file=file,
        output=output,
        exporter=JsonExporter(include_blobs=include_blobs, per_message=per_message),
        topics=topic,
        force=force,
        num_workers=num_workers,
    )
