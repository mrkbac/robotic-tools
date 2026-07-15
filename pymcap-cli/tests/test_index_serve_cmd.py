"""Tests for `pymcap-cli index serve` (Datasette-backed)."""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.index import serve_cmd
from pymcap_cli.cmd.index.scan_cmd import scan_cmd
from pymcap_cli.cmd.index.serve_cmd import (
    _build_datasette_argv,
    index_serve,
)

from tests.fixtures.mcap_generator import create_simple_mcap

if TYPE_CHECKING:
    from collections.abc import Sequence


def _seed_db(tmp_path: Path) -> Path:
    """Run a scan to produce a current-schema index DB with one 3-message file."""
    folder = tmp_path / "rec"
    folder.mkdir()
    (folder / "a.mcap").write_bytes(create_simple_mcap(num_messages=3))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(folder, db=db) == 0
    return db


def test_build_datasette_argv_includes_core_flags() -> None:
    argv = _build_datasette_argv(
        Path("/srv/index.sqlite"),
        Path("m.yaml"),
        Path("plugins"),
        Path("templates"),
        "localhost",
        9999,
        open_browser=False,
    )
    assert argv[:4] == [sys.executable, "-m", "datasette", "serve"]
    assert argv[4] == "/srv/index.sqlite"
    assert "-i" not in argv
    assert argv[argv.index("--metadata") + 1] == "m.yaml"
    assert argv[argv.index("--plugins-dir") + 1] == "plugins"
    assert argv[argv.index("--template-dir") + 1] == "templates"
    assert argv[argv.index("--host") + 1] == "localhost"
    assert argv[argv.index("--port") + 1] == "9999"
    settings = {argv[i + 1]: argv[i + 2] for i, value in enumerate(argv) if value == "--setting"}
    assert settings["sql_time_limit_ms"] == "10000"
    assert settings["allow_download"] == "off"
    assert settings["default_cache_ttl"] == "0"
    assert "-o" not in argv


def test_build_datasette_argv_open_browser_adds_flag() -> None:
    argv = _build_datasette_argv(
        Path("db"), Path("m"), Path("p"), Path("t"), "h", 1, open_browser=True
    )
    assert "-o" in argv


def test_serve_errors_when_db_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert index_serve(db=tmp_path / "missing.sqlite") == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "no index DB" in err


def test_serve_launches_datasette_through_stable_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _seed_db(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(serve_cmd, "_datasette_is_installed", lambda: True)

    def fake_run(argv: Sequence[str], **_kwargs: object) -> object:
        link = Path(argv[4])
        captured["argv"] = list(argv)
        captured["link_name"] = link.name
        captured["is_symlink"] = link.is_symlink()
        captured["resolves_to_db"] = link.resolve() == db.resolve()
        captured["metadata_exists"] = Path(argv[argv.index("--metadata") + 1]).is_file()
        captured["plugins_is_dir"] = Path(argv[argv.index("--plugins-dir") + 1]).is_dir()
        captured["templates_is_dir"] = Path(argv[argv.index("--template-dir") + 1]).is_dir()

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(serve_cmd.subprocess, "run", fake_run)

    assert index_serve(db=db, no_browser=True) == 0
    assert captured["link_name"] == "index.sqlite"
    assert captured["is_symlink"] is True
    assert captured["resolves_to_db"] is True
    assert captured["metadata_exists"] is True
    assert captured["plugins_is_dir"] is True
    assert captured["templates_is_dir"] is True
    assert "-o" not in captured["argv"]  # type: ignore[operator]

    uri = f"{db.resolve().as_uri()}?mode=ro"
    probe = sqlite3.connect(uri, uri=True)
    try:
        files_view = probe.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name='files_view'"
        ).fetchone()
    finally:
        probe.close()
    assert files_view is None


def test_serve_falls_back_when_stable_symlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _seed_db(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(serve_cmd, "_datasette_is_installed", lambda: True)

    def fail_symlink_to(self: Path, target: Path, target_is_directory: bool = False) -> None:
        _ = self, target, target_is_directory
        raise OSError("symlinks unavailable")

    def fake_run(argv: Sequence[str], **_kwargs: object) -> object:
        link = Path(argv[4])
        captured["link_name"] = link.name
        captured["is_symlink"] = link.is_symlink()
        captured["is_file"] = link.is_file()
        captured["same_bytes"] = link.read_bytes() == db.read_bytes()

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(serve_cmd.Path, "symlink_to", fail_symlink_to)
    monkeypatch.setattr(serve_cmd.subprocess, "run", fake_run)

    assert index_serve(db=db, no_browser=True) == 0
    assert captured["link_name"] == "index.sqlite"
    assert captured["is_symlink"] is False
    assert captured["is_file"] is True
    assert captured["same_bytes"] is True


def test_serve_reports_missing_datasette(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = _seed_db(tmp_path)

    monkeypatch.setattr(serve_cmd, "_datasette_is_installed", lambda: False)
    assert index_serve(db=db, no_browser=True) == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "serve extra" in err


def test_json_link_url_encodes_query_parameters() -> None:
    pytest.importorskip("datasette")
    pymcap_render = importlib.import_module("pymcap_cli.index.datasette.plugins.pymcap_render")

    rendered = pymcap_render._json_link(
        '{"href":"/index/file_channels","params":{"path":"/data/a&b #1.mcap"},'
        '"label":"/data/a&b #1.mcap"}'
    )

    assert rendered is not None
    html = str(rendered)
    assert 'href="/index/file_channels?path=%2Fdata%2Fa%26b+%231.mcap"' in html
    assert "/data/a&amp;b #1.mcap" in html


def test_serve_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["index", "serve", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli index serve" in output
    assert "pymcap-cli[serve]" in output
