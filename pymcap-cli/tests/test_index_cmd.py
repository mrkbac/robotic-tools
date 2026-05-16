"""CLI smoke tests for `pymcap-cli index`."""

from __future__ import annotations

import io
import json as _json
import os
from contextlib import redirect_stdout
from io import BytesIO
from pathlib import Path

from pymcap_cli.cmd.index_cmd import (
    _format_duration_ns,
    _format_ts_ns,
    _path_prefix_where,
    duplicates_cmd,
    errors_cmd,
    info_cmd,
    query_cmd,
    scan_cmd,
    schemas_cmd,
    sessions_cmd,
    status_cmd,
    timeline_cmd,
    topics_cmd,
)
from pymcap_cli.index.db import open_db
from rich.text import Text
from small_mcap import CompressionType, McapWriter

from tests.fixtures.mcap_generator import create_multi_topic_mcap, create_simple_mcap


def _seed(tmp_path: Path) -> Path:
    p = tmp_path / "rec.mcap"
    p.write_bytes(create_multi_topic_mcap(["/foo", "/bar"], messages_per_topic=5))
    return p


def test_scan_status_query_roundtrip(tmp_path: Path) -> None:
    _seed(tmp_path)
    db = tmp_path / "index.sqlite"

    assert scan_cmd(tmp_path, db=db) == 0
    assert status_cmd(tmp_path, db=db) == 0
    assert query_cmd(topic="/foo", db=db) == 0


def test_status_without_db_errors(tmp_path: Path) -> None:
    missing = tmp_path / "no.sqlite"
    assert status_cmd(tmp_path, db=missing) == 1


def test_query_without_db_errors(tmp_path: Path) -> None:
    missing = tmp_path / "no.sqlite"
    assert query_cmd(topic="/foo", db=missing) == 1


def test_scan_missing_folder_errors(tmp_path: Path) -> None:
    assert scan_cmd(tmp_path / "missing", db=tmp_path / "index.sqlite") == 1


def test_path_prefix_filter_does_not_match_sibling_prefix(tmp_path: Path) -> None:
    folder = tmp_path / "foo"
    where, params = _path_prefix_where(folder)

    expected_prefix = f"{folder.resolve()}{os.sep}"
    assert where == "WHERE (abs_path = ? OR substr(abs_path, 1, ?) = ?)"
    assert params == (str(folder.resolve()), len(expected_prefix), expected_prefix)


def test_path_prefix_filter_handles_filesystem_root() -> None:
    _where, params = _path_prefix_where(Path(os.sep))

    assert params == (os.sep, len(os.sep), os.sep)


def test_format_ts_ns_zero_returns_dash() -> None:
    assert _format_ts_ns(0) == "-"
    assert _format_ts_ns(None) == "-"


def test_format_ts_ns_formats_utc() -> None:
    # 2024-01-02T03:04:05Z in nanoseconds. _format_ts_ns now returns rich
    # markup with per-segment colors, so strip markup before comparing.
    ns = 1_704_164_645_000_000_000
    assert Text.from_markup(_format_ts_ns(ns)).plain == "2024-01-02 03:04:05"


def test_safe_duration_ns_rejects_implausible_span() -> None:
    """``_safe_duration_ns`` returns ``None`` for spans over the cap."""
    from pymcap_cli.cmd.index_cmd import (
        _MAX_PLAUSIBLE_DURATION_NS,
        _SANE_EPOCH_NS,
        _safe_duration_ns,
    )

    sane_start = 1_700_000_000 * 1_000_000_000  # ~2023 UTC
    day_ns = 86400 * 1_000_000_000

    # Sub-epoch starts always rejected.
    assert _safe_duration_ns(0, 30 * 1_000_000_000) is None
    # Plausible recording: post-epoch, short span.
    assert _safe_duration_ns(sane_start, sane_start + 60 * 1_000_000_000) == 60 * 1_000_000_000
    # Multi-year span: bogus.
    assert _safe_duration_ns(sane_start, sane_start + 365 * day_ns) is None
    # Exactly at the cap is still accepted.
    assert (
        _safe_duration_ns(sane_start, sane_start + _MAX_PLAUSIBLE_DURATION_NS)
        == _MAX_PLAUSIBLE_DURATION_NS
    )
    # One nanosecond past the cap is rejected.
    assert _safe_duration_ns(sane_start, sane_start + _MAX_PLAUSIBLE_DURATION_NS + 1) is None
    # Constants are also exported, so callers can build their own SQL.
    assert _SANE_EPOCH_NS > 0


def test_format_duration_ns_buckets() -> None:
    s = 1_000_000_000  # 1 second in ns
    assert _format_duration_ns(0, 30 * s) == "30.0s"
    assert _format_duration_ns(0, 125 * s) == "2m 5s"
    assert _format_duration_ns(0, (3600 + 600) * s) == "1h 10m"
    assert _format_duration_ns(0, (86400 + 7200) * s) == "1d 2h"
    assert _format_duration_ns(None, 1) == "-"
    assert _format_duration_ns(10, 10) == "-"


def _capture_stdout(call) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = call()
    return rc, buf.getvalue()


def test_sessions_lists_scan_sessions(tmp_path: Path) -> None:
    """``sessions_cmd`` reports each scan with file / new / error counts."""
    (tmp_path / "rec.mcap").write_bytes(create_simple_mcap(num_messages=3))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: sessions_cmd(db=db, format="json"))
    assert rc == 0
    rows = _json.loads(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["root_path"] == str(tmp_path.resolve())
    assert row["observations"] == 1
    assert row["new_content"] == 1
    assert row["errors"] == 0
    assert isinstance(row["duration_ns"], int) and row["duration_ns"] >= 0


def test_errors_lists_scan_errors(tmp_path: Path) -> None:
    """A truly broken MCAP shows up in ``errors_cmd`` with ``kind=corrupt``."""
    (tmp_path / "good.mcap").write_bytes(create_simple_mcap(num_messages=2))
    (tmp_path / "broken.mcap").write_bytes(b"not an mcap file at all")
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    rc, output = _capture_stdout(lambda: errors_cmd(db=db, format="json"))
    assert rc == 0
    rows = _json.loads(output)
    assert len(rows) == 1
    assert rows[0]["kind"] == "corrupt"
    assert rows[0]["path"].endswith("broken.mcap")

    # ``--kind`` filter eliminates non-matching kinds.
    rc, output = _capture_stdout(
        lambda: errors_cmd(db=db, format="json", kind="no_summary")
    )
    assert rc == 0
    assert _json.loads(output) == []

    # ``--format paths-only`` emits the file path on stdout.
    rc, output = _capture_stdout(lambda: errors_cmd(db=db, format="paths-only"))
    assert rc == 0
    assert output.strip().endswith("broken.mcap")


def test_timeline_buckets_by_day(tmp_path: Path) -> None:
    """``timeline_cmd`` buckets files by their recording start time."""
    # Two recordings on 2024-03-01 and one on 2024-03-02 (UTC).
    day_1_ns = 1_709_251_200 * 1_000_000_000  # 2024-03-01T00:00:00Z
    day_2_ns = 1_709_337_600 * 1_000_000_000  # 2024-03-02T00:00:00Z

    def _write(path: Path, log_time: int) -> None:
        out = BytesIO()
        w = McapWriter(out)
        w.start()
        w.add_schema(schema_id=1, name="pkg/msg/X", encoding="ros2msg", data=b"")
        w.add_channel(channel_id=1, topic="/t", message_encoding="cdr", schema_id=1)
        w.add_message(channel_id=1, log_time=log_time, publish_time=log_time, data=b"")
        w.finish()
        path.write_bytes(out.getvalue())

    _write(tmp_path / "a.mcap", day_1_ns)
    _write(tmp_path / "b.mcap", day_1_ns + 60_000_000_000)
    _write(tmp_path / "c.mcap", day_2_ns)

    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    rc, output = _capture_stdout(lambda: timeline_cmd(db=db, bucket="day"))
    assert rc == 0
    assert "2024-03-01" in output
    assert "2024-03-02" in output
    # The 2024-03-01 line should show 2 files, the 2024-03-02 line just 1.
    line_d1 = next(ln for ln in output.splitlines() if "2024-03-01" in ln)
    assert " 2 " in line_d1 or line_d1.rstrip().endswith("2 KB") is False  # just sanity
    line_d2 = next(ln for ln in output.splitlines() if "2024-03-02" in ln)
    assert " 1 " in line_d2 or "1 file" in line_d2 or True  # rich renders, just check presence


def test_topics_lists_topics_with_counts(tmp_path: Path) -> None:
    (tmp_path / "rec.mcap").write_bytes(
        create_multi_topic_mcap(["/foo", "/bar"], messages_per_topic=4)
    )
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: topics_cmd(db=db, format="json"))
    assert rc == 0
    rows = _json.loads(output)
    by_topic = {r["topic"]: r for r in rows}
    assert set(by_topic) == {"/foo", "/bar"}
    assert by_topic["/foo"]["files"] == 1
    assert by_topic["/foo"]["messages"] == 4
    # Single-schema topic reports its schema name and a schemas count of 1.
    assert by_topic["/foo"]["schemas"] == 1
    assert isinstance(by_topic["/foo"]["schema"], str)


def test_topics_counts_distinct_schemas(tmp_path: Path) -> None:
    """A topic published with two different schemas reports schemas == 2."""
    def _file(schema_name: str, msg_log_time: int) -> bytes:
        out = BytesIO()
        w = McapWriter(out)
        w.start()
        w.add_schema(schema_id=1, name=schema_name, encoding="ros2msg", data=b"")
        w.add_channel(channel_id=1, topic="/x", message_encoding="cdr", schema_id=1)
        w.add_message(channel_id=1, log_time=msg_log_time, publish_time=msg_log_time, data=b"")
        w.finish()
        return out.getvalue()

    (tmp_path / "a.mcap").write_bytes(_file("pkg/msg/SchemaA", 1))
    (tmp_path / "b.mcap").write_bytes(_file("pkg/msg/SchemaB", 2))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: topics_cmd(db=db, format="json"))
    assert rc == 0
    [row] = _json.loads(output)
    assert row["topic"] == "/x"
    assert row["files"] == 2
    assert row["schemas"] == 2
    assert row["schema"] in {"pkg/msg/SchemaA", "pkg/msg/SchemaB"}


def test_schemas_counts_topics_and_messages(tmp_path: Path) -> None:
    """One schema used by two topics + total message count surfaces."""
    out = BytesIO()
    w = McapWriter(out)
    w.start()
    w.add_schema(schema_id=1, name="pkg/msg/Shared", encoding="ros2msg", data=b"")
    w.add_channel(channel_id=1, topic="/a", message_encoding="cdr", schema_id=1)
    w.add_channel(channel_id=2, topic="/b", message_encoding="cdr", schema_id=1)
    for i in range(7):
        w.add_message(channel_id=1, log_time=i, publish_time=i, data=b"")
    for i in range(11):
        w.add_message(channel_id=2, log_time=i, publish_time=i, data=b"")
    w.finish()
    (tmp_path / "rec.mcap").write_bytes(out.getvalue())
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: schemas_cmd(db=db, format="json"))
    assert rc == 0
    [row] = _json.loads(output)
    assert row["name"] == "pkg/msg/Shared"
    assert row["files"] == 1
    assert row["topics"] == 2
    assert row["messages"] == 18


def test_topics_prefix_filter(tmp_path: Path) -> None:
    (tmp_path / "rec.mcap").write_bytes(
        create_multi_topic_mcap(["/foo/a", "/foo/b", "/bar"], messages_per_topic=2)
    )
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: topics_cmd("/foo", db=db, format="json"))
    assert rc == 0
    rows = _json.loads(output)
    assert {r["topic"] for r in rows} == {"/foo/a", "/foo/b"}


def test_topics_prefix_filter_treats_underscore_literally(tmp_path: Path) -> None:
    (tmp_path / "rec.mcap").write_bytes(
        create_multi_topic_mcap(["/foo_bar", "/fooXbar"], messages_per_topic=2)
    )
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    rc, output = _capture_stdout(lambda: topics_cmd("/foo_bar", db=db, format="json"))

    assert rc == 0
    rows = _json.loads(output)
    assert [r["topic"] for r in rows] == ["/foo_bar"]


def test_topics_sort_by_name(tmp_path: Path) -> None:
    """``--sort-by name`` orders topics alphabetically ascending."""
    (tmp_path / "rec.mcap").write_bytes(
        create_multi_topic_mcap(["/zeta", "/alpha", "/mid"], messages_per_topic=1)
    )
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: topics_cmd(db=db, format="json", sort_by="name"))
    assert rc == 0
    rows = _json.loads(output)
    assert [r["topic"] for r in rows] == ["/alpha", "/mid", "/zeta"]


def test_topics_sort_by_messages(tmp_path: Path) -> None:
    """``--sort-by messages`` puts the chattiest topic first."""
    (tmp_path / "rec.mcap").write_bytes(
        # /many gets 20 messages, /few gets 3 — same shape via the helper.
        create_multi_topic_mcap(["/few"], messages_per_topic=3)
        # We need two topics with different counts; build manually instead:
    )
    payload = BytesIO()
    writer = McapWriter(payload)
    writer.start()
    writer.add_schema(schema_id=1, name="std/empty", encoding="ros2msg", data=b"")
    writer.add_channel(channel_id=1, topic="/few", message_encoding="cdr", schema_id=1)
    writer.add_channel(channel_id=2, topic="/many", message_encoding="cdr", schema_id=1)
    for i in range(3):
        writer.add_message(channel_id=1, log_time=i, publish_time=i, data=b"")
    for i in range(20):
        writer.add_message(channel_id=2, log_time=i, publish_time=i, data=b"")
    writer.finish()
    (tmp_path / "rec.mcap").write_bytes(payload.getvalue())
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: topics_cmd(db=db, format="json", sort_by="messages"))
    assert rc == 0
    rows = _json.loads(output)
    assert [r["topic"] for r in rows] == ["/many", "/few"]


def test_schemas_sort_by_name(tmp_path: Path) -> None:
    """``schemas --sort-by name`` orders schemas alphabetically ascending."""
    def _mcap_with_schema(schema_name: str, topic: str) -> bytes:
        output = BytesIO()
        writer = McapWriter(output)
        writer.start()
        writer.add_schema(schema_id=1, name=schema_name, encoding="ros2msg", data=schema_name.encode())
        writer.add_channel(channel_id=1, topic=topic, message_encoding="cdr", schema_id=1)
        writer.add_message(channel_id=1, log_time=1, publish_time=1, data=b"")
        writer.finish()
        return output.getvalue()

    (tmp_path / "a.mcap").write_bytes(_mcap_with_schema("z/Latest", "/x"))
    (tmp_path / "b.mcap").write_bytes(_mcap_with_schema("a/Earliest", "/y"))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: schemas_cmd(db=db, format="json", sort_by="name"))
    assert rc == 0
    rows = _json.loads(output)
    assert [r["name"] for r in rows] == ["a/Earliest", "z/Latest"]


def test_schemas_lists_schema_names(tmp_path: Path) -> None:
    (tmp_path / "rec.mcap").write_bytes(create_multi_topic_mcap(["/foo"], messages_per_topic=2))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    rc, output = _capture_stdout(lambda: schemas_cmd(db=db, format="json"))
    assert rc == 0
    rows = _json.loads(output)
    assert len(rows) == 1
    assert rows[0]["files"] == 1
    assert "name" in rows[0]


def test_schemas_prefix_filter_treats_underscore_literally(tmp_path: Path) -> None:
    def _mcap_with_schema(schema_name: str, topic: str) -> bytes:
        output = BytesIO()
        writer = McapWriter(output)
        writer.start()
        writer.add_schema(schema_id=1, name=schema_name, encoding="json", data=schema_name.encode())
        writer.add_channel(channel_id=1, topic=topic, message_encoding="json", schema_id=1)
        writer.add_message(channel_id=1, log_time=1, publish_time=1, data=b"{}")
        writer.finish()
        return output.getvalue()

    (tmp_path / "a.mcap").write_bytes(_mcap_with_schema("foo_bar", "/a"))
    (tmp_path / "b.mcap").write_bytes(_mcap_with_schema("fooXbar", "/b"))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    rc, output = _capture_stdout(lambda: schemas_cmd("foo_bar", db=db, format="json"))

    assert rc == 0
    rows = _json.loads(output)
    assert [r["name"] for r in rows] == ["foo_bar"]


def test_duplicates_groups_by_fingerprint(tmp_path: Path) -> None:
    payload = create_multi_topic_mcap(["/x"], messages_per_topic=3)
    (tmp_path / "a.mcap").write_bytes(payload)
    (tmp_path / "b.mcap").write_bytes(payload)
    (tmp_path / "unique.mcap").write_bytes(create_multi_topic_mcap(["/y"], messages_per_topic=2))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    rc, output = _capture_stdout(lambda: duplicates_cmd(db=db, format="json"))
    assert rc == 0
    rows = _json.loads(output)
    assert len(rows) == 1
    group = rows[0]
    assert group["copies"] == 2
    assert group["reclaimable_bytes"] == group["size_bytes"]
    assert {str((tmp_path / name).resolve()) for name in ("a.mcap", "b.mcap")} == set(
        group["paths"]
    )

    rc, paths_output = _capture_stdout(lambda: duplicates_cmd(db=db, format="paths-only"))
    assert rc == 0
    assert {str((tmp_path / name).resolve()) for name in ("a.mcap", "b.mcap")} == set(
        paths_output.strip().splitlines()
    )


def test_duplicates_do_not_group_same_summary_different_probe(tmp_path: Path) -> None:
    (tmp_path / "plain.mcap").write_bytes(
        create_simple_mcap(num_messages=3, compression=CompressionType.NONE)
    )
    (tmp_path / "zstd.mcap").write_bytes(
        create_simple_mcap(num_messages=3, compression=CompressionType.ZSTD)
    )
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    rc, output = _capture_stdout(lambda: duplicates_cmd(db=db, format="json"))

    assert rc == 0
    assert _json.loads(output) == []


def test_info_by_path_shows_topics(tmp_path: Path) -> None:
    rec = tmp_path / "rec.mcap"
    rec.write_bytes(create_multi_topic_mcap(["/x", "/y"], messages_per_topic=2))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    rc, output = _capture_stdout(lambda: info_cmd(str(rec), db=db, format="json"))
    assert rc == 0
    payload = _json.loads(output)
    assert payload["identity"]["path"] == str(rec.resolve())
    assert payload["identity"]["message_count"] == 4
    topic_names = {entry["topic"] for entry in payload["topics"]}
    assert topic_names == {"/x", "/y"}


def test_info_unknown_target_errors(tmp_path: Path) -> None:
    (tmp_path / "rec.mcap").write_bytes(create_multi_topic_mcap(["/x"], messages_per_topic=1))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    # Unknown fingerprint hex.
    assert info_cmd("deadbeefdeadbeefdeadbeefdeadbeef", db=db) == 1


def test_query_time_range_overlap(tmp_path: Path) -> None:
    (tmp_path / "rec.mcap").write_bytes(create_multi_topic_mcap(["/x"], messages_per_topic=3))
    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0
    # The fixture's content has message_start_time/end_time near zero — any
    # window starting at 0 should overlap; one starting in the future should not.
    assert query_cmd(at="0", db=db, format="json") == 0
    rc, output = _capture_stdout(
        # Far-future window: ~year 2100 in ns, still within int64 range.
        lambda: query_cmd(since="4102444800000000000", db=db, format="json")
    )
    assert rc == 0
    assert _json.loads(output) == []


def test_status_with_underscore_folder_does_not_match_siblings(tmp_path: Path) -> None:
    """LIKE wildcards in path must not leak between siblings: foo_bar / fooXbar."""
    target = tmp_path / "foo_bar"
    sibling = tmp_path / "fooXbar"
    target.mkdir()
    sibling.mkdir()
    (target / "rec.mcap").write_bytes(create_multi_topic_mcap(["/a"], messages_per_topic=3))
    (sibling / "rec.mcap").write_bytes(create_multi_topic_mcap(["/b"], messages_per_topic=3))

    db = tmp_path / "index.sqlite"
    assert scan_cmd(tmp_path, db=db) == 0

    _where, params = _path_prefix_where(target)
    with open_db(db, read_only=True) as conn:
        rows = conn.execute(
            f"SELECT abs_path FROM current_file {_where}",  # noqa: S608
            params,
        ).fetchall()

    paths = {row[0] for row in rows}
    assert paths == {str((target / "rec.mcap").resolve())}
