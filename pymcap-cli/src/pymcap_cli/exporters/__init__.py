"""Pluggable per-format exporters for MCAP message data.

Each exporter implements the :class:`Exporter` protocol from
:mod:`pymcap_cli.exporters.base`. The :func:`run_export` driver iterates
decoded messages and routes them to per-topic :class:`TopicWriter` instances
created by the selected exporter.

Adding a new format = drop a new module here and register it via the CLI.
"""

from pymcap_cli.exporters.base import (
    Exporter,
    JsonRos2Exporter,
    Ros2Exporter,
    TopicContext,
    TopicWriter,
)
from pymcap_cli.exporters.driver import run_export

__all__ = [
    "Exporter",
    "JsonRos2Exporter",
    "Ros2Exporter",
    "TopicContext",
    "TopicWriter",
    "run_export",
]
