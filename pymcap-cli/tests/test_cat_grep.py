"""Tests for `pymcap-cli cat --grep` content search."""

from __future__ import annotations

import io
import json
import re
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.cat_cmd import cat
from pymcap_cli.display.message_render import message_matches_grep
from pymcap_cli.utils import NS_TO_MS
from small_mcap import CompressionType, McapWriter

if TYPE_CHECKING:
    from pathlib import Path


def _build_grep_fixture(path: Path) -> Path:
    """Write a small JSON-encoded MCAP with diverse strings/levels for grepping."""
    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=1024 * 1024, compression=CompressionType.NONE)
    writer.start()
    writer.add_schema(schema_id=1, name="diag", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/diagnostics", message_encoding="json", schema_id=1)
    writer.add_channel(channel_id=2, topic="/odom", message_encoding="json", schema_id=1)

    payloads: list[tuple[int, dict[str, object]]] = [
        (1, {"level": "OK", "name": "encoder", "msg": "all good"}),
        (1, {"level": "WARN", "name": "encoder", "msg": "timeout reached"}),
        (1, {"level": "ERROR", "name": "lidar", "msg": "stalled at startup"}),
        (2, {"pose": {"x": 1.0, "y": 2.0, "frame_id": "base_link"}}),
        (2, {"pose": {"x": 3.0, "y": 4.0, "frame_id": "odom"}}),
    ]
    for i, (channel_id, payload) in enumerate(payloads):
        writer.add_message(
            channel_id=channel_id,
            log_time=i * NS_TO_MS,
            data=json.dumps(payload).encode(),
            publish_time=i * NS_TO_MS,
        )

    writer.finish()
    path.write_bytes(buf.getvalue())
    return path


@pytest.fixture
def grep_mcap(tmp_path: Path) -> Path:
    return _build_grep_fixture(tmp_path / "grep.mcap")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_message_matches_grep_walks_dicts() -> None:
    pattern = re.compile("timeout")
    assert message_matches_grep({"level": "WARN", "msg": "timeout reached"}, pattern)
    assert not message_matches_grep({"level": "OK", "msg": "all good"}, pattern)


def test_message_matches_grep_recurses_into_nested() -> None:
    pattern = re.compile("base_link")
    payload = {"pose": {"frame_id": "base_link"}}
    assert message_matches_grep(payload, pattern)


def test_message_matches_grep_skips_bytes() -> None:
    pattern = re.compile("payload")
    # Even though the bytes contain the substring, we never decode them.
    assert not message_matches_grep({"data": b"payload-bytes"}, pattern)


def test_message_matches_grep_matches_numeric_repr() -> None:
    # Regex'ing over numbers is sometimes useful (e.g. specific status codes).
    pattern = re.compile(r"\b42\b")
    assert message_matches_grep({"status": 42}, pattern)
    assert not message_matches_grep({"status": 7}, pattern)


def test_cat_grep_emits_only_matching_messages(grep_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    rc = cat(str(grep_mcap), grep="timeout|stalled", output=out)
    assert rc == 0
    rows = _read_jsonl(out)
    assert len(rows) == 2
    msgs = [r["message"]["msg"] for r in rows]  # type: ignore[index]
    assert msgs == ["timeout reached", "stalled at startup"]


def test_cat_grep_ignore_case(grep_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    rc = cat(str(grep_mcap), grep="ERROR", grep_ignore_case=True, output=out)
    assert rc == 0
    rows = _read_jsonl(out)
    # "ERROR" appears as a level value once; case-insensitive shouldn't lose it.
    assert len(rows) == 1
    assert rows[0]["message"]["level"] == "ERROR"  # type: ignore[index]


def test_cat_grep_composes_with_topic_filter(grep_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    rc = cat(str(grep_mcap), grep="base_link", topics=["/odom"], output=out)
    assert rc == 0
    rows = _read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["topic"] == "/odom"


def test_cat_grep_invalid_regex_returns_error(grep_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    rc = cat(str(grep_mcap), grep="(unclosed", output=out)
    assert rc == 1
    assert not out.exists() or out.read_text() == ""


def test_cat_grep_no_match_writes_empty(grep_mcap: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    rc = cat(str(grep_mcap), grep="never_present_string_xyzzy", output=out)
    assert rc == 0
    assert out.read_text() == ""
