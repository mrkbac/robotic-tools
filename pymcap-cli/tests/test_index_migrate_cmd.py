"""Tests for `pymcap-cli index migrate`."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.index.migrate_cmd import migrate_cmd
from pymcap_cli.cmd.index.scan_cmd import scan_cmd
from pymcap_cli.index.db import CURRENT_SCHEMA_VERSION

from tests.fixtures.mcap_generator import create_simple_mcap

if TYPE_CHECKING:
    from pathlib import Path


def _seed_db(tmp_path: Path) -> Path:
    """Run a scan to produce a current-schema index DB."""
    folder = tmp_path / "rec"
    folder.mkdir()
    (folder / "a.mcap").write_bytes(create_simple_mcap(num_messages=3))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(folder, db=db) == 0
    return db


def _flatten(text: str) -> str:
    """Collapse Rich's terminal line-wraps for substring assertions."""
    return " ".join(text.split())


def test_migrate_errors_when_db_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert migrate_cmd(db=tmp_path / "missing.sqlite") == 1
    out = _flatten(capsys.readouterr().out)
    assert "no index DB" in out


def test_migrate_noop_when_already_current(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = _seed_db(tmp_path)
    assert migrate_cmd(db=db) == 0
    out = _flatten(capsys.readouterr().out)
    assert f"v{CURRENT_SCHEMA_VERSION}" in out
    assert "nothing to do" in out


def test_migrate_refuses_to_downgrade(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = _seed_db(tmp_path)
    conn = sqlite3.connect(db)
    try:
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
        conn.commit()
    finally:
        conn.close()
    assert migrate_cmd(db=db) == 1
    out = _flatten(capsys.readouterr().out)
    assert "refusing to downgrade" in out


def test_migrate_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["index", "migrate", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli index migrate" in output
