"""Pluggable per-format exporters for MCAP message data.

Artifact exporters implement the low-level :class:`Exporter` API. Structured
formats extend :class:`StructuredExporter` and provide per-topic
``Writer[StructuredRecord]`` destinations.

Adding a new format = drop a new module here and register it via the CLI.
"""

from pymcap_cli.exporters.base import (
    Exporter,
    TopicContext,
    Writer,
)
from pymcap_cli.exporters.driver import run_export
from pymcap_cli.exporters.structured import (
    PerTopicFileExporter,
    StructuredExporter,
    StructuredRecord,
)

__all__ = [
    "Exporter",
    "PerTopicFileExporter",
    "StructuredExporter",
    "StructuredRecord",
    "TopicContext",
    "Writer",
    "run_export",
]
