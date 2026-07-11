"""Tests for the pymcap_cli logging framework."""

from __future__ import annotations

import logging

import pytest
from pymcap_cli.core import mcap_processor
from pymcap_cli.log_setup import ERR, setup_logging
from rich.logging import RichHandler


@pytest.fixture(autouse=True)
def _reset_pymcap_logger():
    """Snapshot and restore the pymcap_cli logger between tests."""
    logger = logging.getLogger("pymcap_cli")
    saved_level = logger.level
    saved_propagate = logger.propagate
    saved_handlers = list(logger.handlers)
    logger.handlers = []
    yield
    logger.handlers = saved_handlers
    logger.level = saved_level
    logger.propagate = saved_propagate


def test_setup_logging_attaches_one_rich_handler():
    setup_logging()
    logger = logging.getLogger("pymcap_cli")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], RichHandler)


def test_setup_logging_is_idempotent():
    setup_logging()
    setup_logging()
    setup_logging()
    logger = logging.getLogger("pymcap_cli")
    assert len(logger.handlers) == 1


def test_setup_logging_sets_propagate_false():
    logger = logging.getLogger("pymcap_cli")
    logger.propagate = True
    setup_logging()
    assert logger.propagate is False


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected"),
    [
        (0, 0, logging.INFO),
        (1, 0, logging.DEBUG),
        (2, 0, logging.DEBUG),
        (0, 1, logging.WARNING),
        (0, 2, logging.ERROR),
        (0, 3, logging.ERROR),
        (1, 1, logging.WARNING),  # quiet wins over verbose
        (2, 2, logging.ERROR),
    ],
)
def test_setup_logging_level_mapping(verbose: int, quiet: int, expected: int):
    setup_logging(verbose=verbose, quiet=quiet)
    logger = logging.getLogger("pymcap_cli")
    assert logger.level == expected


def test_setup_logging_does_not_touch_root_logger():
    root_handlers_before = list(logging.getLogger().handlers)
    setup_logging()
    root_handlers_after = list(logging.getLogger().handlers)
    assert root_handlers_after == root_handlers_before


def test_logger_in_pymcap_cli_namespace_emits_at_info():
    setup_logging()
    child = logging.getLogger("pymcap_cli.foo.bar")
    captured: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    sink = _ListHandler(level=logging.DEBUG)
    pkg_logger = logging.getLogger("pymcap_cli")
    pkg_logger.addHandler(sink)
    try:
        child.info("hello")
    finally:
        pkg_logger.removeHandler(sink)

    assert any(
        record.name == "pymcap_cli.foo.bar" and record.getMessage() == "hello"
        for record in captured
    )


def test_setup_logging_clears_prior_handlers():
    logger = logging.getLogger("pymcap_cli")
    sentinel = logging.NullHandler()
    logger.addHandler(sentinel)
    setup_logging()
    assert sentinel not in logger.handlers


def test_processor_progress_uses_logging_console():
    assert mcap_processor.console is ERR
