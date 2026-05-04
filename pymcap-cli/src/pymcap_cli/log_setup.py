"""Logging setup for pymcap-cli.

Two output channels:
- ``OUT`` writes results (tables, JSON, file dumps) to stdout so piping works.
- ``ERR`` writes diagnostics, status, errors, and progress bars to stderr.

`setup_logging` wires a single `RichHandler` onto the ``pymcap_cli`` package
logger; it is idempotent so test suites can call it repeatedly.

Tests should prefer ``caplog`` for log assertions and ``capsys`` for stdout
result assertions.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

OUT = Console()
ERR = Console(stderr=True)

_PACKAGE_LOGGER = "pymcap_cli"


def setup_logging(verbose: int = 0, quiet: int = 0) -> None:
    """Configure the ``pymcap_cli`` package logger with a Rich stderr handler.

    Level mapping (``quiet`` wins if both are non-zero):

    - default -> ``INFO``
    - ``verbose >= 1`` -> ``DEBUG``
    - ``quiet == 1`` -> ``WARNING``
    - ``quiet >= 2`` -> ``ERROR``
    """
    if quiet >= 2:
        level = logging.ERROR
    elif quiet == 1:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger = logging.getLogger(_PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = RichHandler(
        console=ERR,
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
