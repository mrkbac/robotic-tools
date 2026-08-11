"""Tests for per-file MessagePath reductions over indexed recordings."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from pymcap_cli.cmd.index import index_app
from pymcap_cli.cmd.index.stats_cmd import SchemaCache, stats_cmd
from pymcap_cli.index.db import open_db
from pymcap_cli.index.scanner import scan
from small_mcap import McapWriter

if TYPE_CHECKING:
    from pathlib import Path


def _write_values(
    path: Path,
    topic: str,
    values: list[float],
    *,
    schema_encoding: str = "json",
    schema_data: bytes = b"{}",
) -> None:
    output = io.BytesIO()
    writer = McapWriter(output)
    writer.start()
    writer.add_schema(
        schema_id=1,
        name="sample",
        encoding=schema_encoding,
        data=schema_data,
    )
    writer.add_channel(channel_id=1, topic=topic, message_encoding="json", schema_id=1)
    for index, value in enumerate(values):
        writer.add_message(
            channel_id=1,
            log_time=index,
            publish_time=index,
            data=json.dumps({"value": value}).encode(),
        )
    writer.finish()
    path.write_bytes(output.getvalue())


def _scan(root: Path, db_path: Path) -> None:
    with open_db(db_path) as conn:
        scan(root, conn, pymcap_cli_version="test", jobs=1)


def test_stats_cmd_reduces_each_matching_file(tmp_path: Path, capsys) -> None:
    first = tmp_path / "first recording.mcap"
    second = tmp_path / "second.mcap"
    _write_values(first, "/preassure/front", [1.0, 4.5, 2.0])
    _write_values(second, "/preassure/front", [-3.0, -1.0])
    _write_values(tmp_path / "unrelated.mcap", "/other", [100.0])
    db_path = tmp_path / "index.sqlite"
    _scan(tmp_path, db_path)

    exit_code = stats_cmd(
        tmp_path,
        query=["maximum=/preassure/front.value.@@max"],
        format="json",
        db=db_path,
    )

    assert exit_code == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {"path": str(first), "stats": {"maximum": 4.5}},
        {"path": str(second), "stats": {"maximum": -1.0}},
    ]


def test_stats_cmd_requires_stream_reducer(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "index.sqlite"
    with open_db(db_path):
        pass

    exit_code = stats_cmd(
        query=["/preassure/front.value"],
        format="json",
        db=db_path,
    )

    assert exit_code == 1
    assert "must end in a stream reducer" in capsys.readouterr().out


def test_stats_cmd_no_matching_files_emits_empty_json(tmp_path: Path, capsys) -> None:
    _write_values(tmp_path / "recording.mcap", "/other", [1.0])
    db_path = tmp_path / "index.sqlite"
    _scan(tmp_path, db_path)

    exit_code = stats_cmd(
        query=["/preassure/front.value.@@max"],
        format="json",
        db=db_path,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == []


def test_stats_cmd_validates_shared_ros_schema_once(tmp_path: Path, capsys, monkeypatch) -> None:
    for name in ("first.mcap", "second.mcap"):
        _write_values(
            tmp_path / name,
            "/preassure/front",
            [1.0],
            schema_encoding="ros2msg",
            schema_data=b"float64 value",
        )
    db_path = tmp_path / "index.sqlite"
    _scan(tmp_path, db_path)
    calls = 0
    original = SchemaCache.validate_query

    def counting_validate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(SchemaCache, "validate_query", counting_validate)

    assert (
        stats_cmd(
            tmp_path,
            query=["/preassure/front.value.@@max"],
            format="json",
            db=db_path,
        )
        == 0
    )
    capsys.readouterr()
    assert calls == 1


def test_stats_cmd_reports_stale_corrupt_file_and_continues(tmp_path: Path, capsys) -> None:
    recording = tmp_path / "recording.mcap"
    _write_values(recording, "/preassure/front", [1.0])
    db_path = tmp_path / "index.sqlite"
    _scan(tmp_path, db_path)
    recording.write_bytes(b"not an MCAP")

    exit_code = stats_cmd(
        tmp_path,
        query=["/preassure/front.value.@@max"],
        format="json",
        db=db_path,
    )

    assert exit_code == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["path"] == str(recording)
    assert "error" in rows[0]


def test_stats_cmd_cli_roundtrip(tmp_path: Path, capsys) -> None:
    recording = tmp_path / "recording.mcap"
    _write_values(recording, "/preassure/front", [1.0, 3.0])
    db_path = tmp_path / "index.sqlite"
    _scan(tmp_path, db_path)

    try:
        index_app(
            [
                "stats",
                str(tmp_path),
                "--query",
                "maximum=/preassure/front.value.@@max",
                "--format",
                "json",
                "--db",
                str(db_path),
            ]
        )
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    else:
        exit_code = 0

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == [
        {"path": str(recording), "stats": {"maximum": 3.0}}
    ]
