"""Canonicalisation invariants for the semantic summary fingerprint."""

from __future__ import annotations

import pytest
from pymcap_cli.index.summary_fingerprint import (
    SCHEME_VERSION,
    _schema_hash,
    summary_fingerprint,
)
from small_mcap.rebuild import RebuildInfo
from small_mcap.records import (
    AttachmentIndex,
    Channel,
    Header,
    MetadataIndex,
    Schema,
    Statistics,
    Summary,
)


def _info(
    *,
    profile: str = "ros2",
    library: str = "writer-A",
    schemas: list[Schema] | None = None,
    channels: list[Channel] | None = None,
    message_counts: dict[int, int] | None = None,
    msg_start: int = 100,
    msg_end: int = 200,
    attachments: list[AttachmentIndex] | None = None,
    metadata_indexes: list[MetadataIndex] | None = None,
) -> RebuildInfo:
    schemas = schemas or [Schema(id=1, name="pkg/msg/T", encoding="ros2msg", data=b"defT")]
    channels = channels or [
        Channel(id=1, schema_id=1, topic="/t", message_encoding="cdr", metadata={})
    ]
    message_counts = message_counts if message_counts is not None else {1: 5}
    stats = Statistics(
        message_count=sum(message_counts.values()),
        schema_count=len(schemas),
        channel_count=len(channels),
        attachment_count=len(attachments or []),
        metadata_count=len(metadata_indexes or []),
        chunk_count=0,
        message_start_time=msg_start,
        message_end_time=msg_end,
        channel_message_counts=dict(message_counts),
    )
    summary = Summary(
        statistics=stats,
        schemas={s.id: s for s in schemas},
        channels={c.id: c for c in channels},
        chunk_indexes=[],
        attachment_indexes=list(attachments or []),
        metadata_indexes=list(metadata_indexes or []),
    )
    return RebuildInfo(header=Header(profile=profile, library=library), summary=summary)


def test_fingerprint_has_scheme_prefix() -> None:
    fp = summary_fingerprint(_info())
    assert fp.startswith(f"{SCHEME_VERSION}:")
    # Remainder is xxh3_128 hex => 32 chars.
    assert len(fp.split(":", 1)[1]) == 32


def test_library_string_does_not_affect_fingerprint() -> None:
    assert summary_fingerprint(_info(library="writer-A")) == summary_fingerprint(
        _info(library="writer-B vendored")
    )


def test_profile_change_changes_fingerprint() -> None:
    assert summary_fingerprint(_info(profile="ros2")) != summary_fingerprint(_info(profile="ros1"))


def test_schema_permutation_yields_same_fingerprint() -> None:
    s1 = Schema(id=1, name="pkg/msg/A", encoding="ros2msg", data=b"A")
    s2 = Schema(id=2, name="pkg/msg/B", encoding="ros2msg", data=b"B")
    ch_a = Channel(id=1, schema_id=1, topic="/a", message_encoding="cdr", metadata={})
    ch_b = Channel(id=2, schema_id=2, topic="/b", message_encoding="cdr", metadata={})
    fp1 = summary_fingerprint(
        _info(schemas=[s1, s2], channels=[ch_a, ch_b], message_counts={1: 1, 2: 2})
    )
    fp2 = summary_fingerprint(
        _info(schemas=[s2, s1], channels=[ch_b, ch_a], message_counts={1: 1, 2: 2})
    )
    assert fp1 == fp2


def test_schema_id_reassignment_yields_same_fingerprint() -> None:
    """Same schema content under different writer-assigned IDs => same fingerprint."""
    a1 = Schema(id=7, name="pkg/msg/A", encoding="ros2msg", data=b"AAA")
    ch1 = Channel(id=1, schema_id=7, topic="/a", message_encoding="cdr", metadata={})
    a2 = Schema(id=42, name="pkg/msg/A", encoding="ros2msg", data=b"AAA")
    ch2 = Channel(id=1, schema_id=42, topic="/a", message_encoding="cdr", metadata={})
    assert summary_fingerprint(_info(schemas=[a1], channels=[ch1])) == summary_fingerprint(
        _info(schemas=[a2], channels=[ch2])
    )


def test_schema_rename_changes_fingerprint() -> None:
    a = Schema(id=1, name="pkg/msg/A", encoding="ros2msg", data=b"AAA")
    b = Schema(id=1, name="pkg/msg/B", encoding="ros2msg", data=b"AAA")
    assert summary_fingerprint(_info(schemas=[a])) != summary_fingerprint(_info(schemas=[b]))


def test_non_ros_schema_hash_includes_name_and_encoding() -> None:
    assert _schema_hash("json", "pkg/A", b"{}") != _schema_hash("json", "pkg/B", b"{}")
    assert _schema_hash("json", "pkg/A", b"{}") != _schema_hash("protobuf", "pkg/A", b"{}")


def test_per_channel_count_change_changes_fingerprint() -> None:
    base = summary_fingerprint(_info(message_counts={1: 5}))
    bumped = summary_fingerprint(_info(message_counts={1: 6}))
    assert base != bumped


def test_attachment_permutation_yields_same_fingerprint() -> None:
    a = AttachmentIndex(
        offset=0, length=0, log_time=0, create_time=0, data_size=10, name="a", media_type="x"
    )
    b = AttachmentIndex(
        offset=0, length=0, log_time=0, create_time=0, data_size=20, name="b", media_type="y"
    )
    fp1 = summary_fingerprint(_info(attachments=[a, b]))
    fp2 = summary_fingerprint(_info(attachments=[b, a]))
    assert fp1 == fp2


def test_attachment_layout_fields_do_not_affect_fingerprint() -> None:
    """Offset/length/log_time/create_time vary on re-emit and must not matter."""
    a1 = AttachmentIndex(
        offset=100, length=200, log_time=1, create_time=2, data_size=10, name="a", media_type="x"
    )
    a2 = AttachmentIndex(
        offset=999,
        length=888,
        log_time=777,
        create_time=666,
        data_size=10,
        name="a",
        media_type="x",
    )
    assert summary_fingerprint(_info(attachments=[a1])) == summary_fingerprint(
        _info(attachments=[a2])
    )


def test_attachment_identity_fields_do_affect_fingerprint() -> None:
    a = AttachmentIndex(
        offset=0, length=0, log_time=0, create_time=0, data_size=10, name="a", media_type="x"
    )
    renamed = AttachmentIndex(
        offset=0, length=0, log_time=0, create_time=0, data_size=10, name="A", media_type="x"
    )
    different_type = AttachmentIndex(
        offset=0, length=0, log_time=0, create_time=0, data_size=10, name="a", media_type="y"
    )
    different_size = AttachmentIndex(
        offset=0, length=0, log_time=0, create_time=0, data_size=11, name="a", media_type="x"
    )
    base = summary_fingerprint(_info(attachments=[a]))
    assert base != summary_fingerprint(_info(attachments=[renamed]))
    assert base != summary_fingerprint(_info(attachments=[different_type]))
    assert base != summary_fingerprint(_info(attachments=[different_size]))


def test_metadata_index_name_changes_fingerprint() -> None:
    m1 = MetadataIndex(offset=0, length=0, name="cal")
    m2 = MetadataIndex(offset=0, length=0, name="calibration")
    assert summary_fingerprint(_info(metadata_indexes=[m1])) != summary_fingerprint(
        _info(metadata_indexes=[m2])
    )


def test_metadata_index_permutation_yields_same_fingerprint() -> None:
    m1 = MetadataIndex(offset=0, length=0, name="a")
    m2 = MetadataIndex(offset=0, length=0, name="b")
    assert summary_fingerprint(_info(metadata_indexes=[m1, m2])) == summary_fingerprint(
        _info(metadata_indexes=[m2, m1])
    )


def test_channel_metadata_canonicalised() -> None:
    ch_ordered = Channel(
        id=1, schema_id=1, topic="/t", message_encoding="cdr", metadata={"a": "1", "b": "2"}
    )
    ch_reversed = Channel(
        id=1, schema_id=1, topic="/t", message_encoding="cdr", metadata={"b": "2", "a": "1"}
    )
    assert summary_fingerprint(_info(channels=[ch_ordered])) == summary_fingerprint(
        _info(channels=[ch_reversed])
    )


def test_missing_statistics_raises() -> None:
    info = _info()
    info.summary.statistics = None
    with pytest.raises(ValueError, match="Statistics"):
        summary_fingerprint(info)


def test_time_range_change_changes_fingerprint() -> None:
    base = summary_fingerprint(_info(msg_start=100, msg_end=200))
    different = summary_fingerprint(_info(msg_start=100, msg_end=300))
    assert base != different
