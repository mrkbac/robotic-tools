"""Tests for the pre-scan summary coverage hints shared by all exporters."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.exporters._summary_hints import warn_topic_coverage
from small_mcap import CompressionType, McapWriter, Summary, get_summary

if TYPE_CHECKING:
    from pathlib import Path


def _make_two_topic_mcap(path: Path, *, big: int, small: int, summary: bool = True) -> None:
    output = io.BytesIO()
    writer = McapWriter(
        output, chunk_size=4096, compression=CompressionType.ZSTD, use_statistics=summary
    )
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/big", message_encoding="json", schema_id=1)
    writer.add_channel(channel_id=2, topic="/small", message_encoding="json", schema_id=1)
    log_time = 0
    for i in range(big):
        writer.add_message(
            channel_id=1, log_time=log_time, data=f'{{"i":{i}}}'.encode(), publish_time=log_time
        )
        log_time += 1_000_000
    for i in range(small):
        writer.add_message(
            channel_id=2, log_time=log_time, data=f'{{"i":{i}}}'.encode(), publish_time=log_time
        )
        log_time += 1_000_000
    writer.finish()
    path.write_bytes(output.getvalue())


def _summary_of(path: Path) -> Summary | None:
    with path.open("rb") as stream:
        return get_summary(stream)


@pytest.fixture
def summary(tmp_path: Path) -> Summary | None:
    path = tmp_path / "rec.mcap"
    _make_two_topic_mcap(path, big=1000, small=1)
    return _summary_of(path)


def test_warns_missing_topic(summary: Summary | None, caplog):
    with caplog.at_level("WARNING"):
        warn_topic_coverage(summary, "rec.mcap", ["/big", "/ghost"])
    assert "'/ghost' has no messages" in caplog.text


def test_warns_sparse_topic(summary: Summary | None, caplog):
    with caplog.at_level("WARNING"):
        warn_topic_coverage(summary, "rec.mcap", ["/big", "/small"])
    assert "'/small' has only 1 message" in caplog.text


def test_no_warning_for_healthy_topics(summary: Summary | None, caplog):
    with caplog.at_level("WARNING"):
        warn_topic_coverage(summary, "rec.mcap", ["/big"])
    assert caplog.text == ""


def test_no_topic_check_when_topics_none(summary: Summary | None, caplog):
    """With no explicit topics (export-all), a summarised file warns nothing."""
    with caplog.at_level("WARNING"):
        warn_topic_coverage(summary, "rec.mcap", None)
    assert caplog.text == ""


def test_warns_when_no_summary(tmp_path: Path, caplog):
    path = tmp_path / "nosummary.mcap"
    _make_two_topic_mcap(path, big=5, small=5, summary=False)
    with caplog.at_level("WARNING"):
        warn_topic_coverage(_summary_of(path), str(path), ["/big"])
    assert "No usable summary" in caplog.text


def test_zero_messages_is_distinct_from_no_summary(tmp_path: Path, caplog):
    """A summarised but empty file reports 0 messages, not a missing summary."""
    path = tmp_path / "empty.mcap"
    _make_two_topic_mcap(path, big=0, small=0, summary=True)
    with caplog.at_level("WARNING"):
        warn_topic_coverage(_summary_of(path), str(path), ["/big"])
    assert "reports 0 messages" in caplog.text
    assert "No usable summary" not in caplog.text


def test_warns_when_summary_is_none(caplog):
    with caplog.at_level("WARNING"):
        warn_topic_coverage(None, "gone.mcap", ["/big"])
    assert "No usable summary" in caplog.text


def test_zero_count_channel_reads_as_blank(caplog):
    """A channel present in the stats with a 0 count is treated as blank, not missing."""
    summary = SimpleNamespace(
        statistics=SimpleNamespace(message_count=1000, channel_message_counts={1: 1000, 2: 0}),
        channels={1: SimpleNamespace(topic="/big"), 2: SimpleNamespace(topic="/zero")},
    )
    with caplog.at_level("WARNING"):
        warn_topic_coverage(summary, "rec.mcap", ["/big", "/zero"])  # type: ignore[arg-type]
    assert "'/zero' has no messages" in caplog.text
    assert "'/big'" not in caplog.text
