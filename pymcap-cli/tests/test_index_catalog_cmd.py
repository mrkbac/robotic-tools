"""Smoke tests for the versioned catalog exported from the sidecar index."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.index import index_app, scan_cmd

from tests.fixtures.mcap_generator import create_multi_topic_mcap

if TYPE_CHECKING:
    from pathlib import Path


def test_catalog_exports_only_recordings_under_the_serving_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "recordings"
    dataset = root / "test runs"
    dataset.mkdir(parents=True)
    sibling = tmp_path / "recordings-other"
    sibling.mkdir()
    data = create_multi_topic_mcap(["/camera/front", "/imu"], messages_per_topic=2)
    (dataset / "run #1.mcap").write_bytes(data)
    (sibling / "other.mcap").write_bytes(data)
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit, match=r"^0$"):
        index_app(["catalog", str(root), "--db", str(db), "--name", "Field tests"])
    output = capsys.readouterr().out
    catalog = json.loads(output)
    assert str(tmp_path) not in output
    assert catalog["version"] == 1
    assert catalog["name"] == "Field tests"
    (recording,) = catalog["recordings"]
    assert recording["id"] == "test runs/run #1.mcap"
    assert recording["name"] == "run #1.mcap"
    assert recording["dataset"] == "test runs"
    assert recording["source"] == {"type": "remote-file", "url": "test%20runs/run%20%231.mcap"}
    assert recording["size"] == len(data)
    assert recording["messages"] == 4
    assert [(topic["name"], topic["messages"]) for topic in recording["topics"]] == [
        ("/camera/front", 2),
        ("/imu", 2),
    ]
    assert recording["start"] == "1970-01-01T00:00:00.000000000Z"
    assert recording["end"] == "1970-01-01T00:00:00.003000000Z"


def test_catalog_can_target_a_separate_recording_host(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "run.mcap").write_bytes(create_multi_topic_mcap(["/imu"], messages_per_topic=1))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit, match=r"^0$"):
        index_app(
            [
                "catalog",
                str(tmp_path),
                "--db",
                str(db),
                "--base-url",
                "https://example.com/archive",
            ]
        )
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["recordings"][0]["source"]["url"] == "https://example.com/archive/run.mcap"
