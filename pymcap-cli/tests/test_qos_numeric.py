"""Tests for converting ROS 2 QoS channel metadata to Humble's numeric form."""

from __future__ import annotations

import io
import zlib

import pytest
import yaml
from pymcap_cli.cmd import process_cmd
from pymcap_cli.core.processors.base import Action, ChunkDecision
from pymcap_cli.core.processors.qos_metadata import QosMetadataProcessor
from pymcap_cli.core.qos import (
    parse_qos_override_yaml,
    parse_qos_set_rule,
    qos_profiles_to_numeric,
)
from small_mcap import (
    Channel,
    Chunk,
    CompressionType,
    McapWriter,
    Message,
    MessageIndex,
    breakup_chunk,
    get_summary,
    read_message,
    stream_reader,
)

from tests.helpers import channel_context, chunk_context, lazy_chunk

_MODERN_QOS = """\
- history: keep_last
  depth: 5
  reliability: best_effort
  durability: volatile
  deadline:
    sec: 0
    nsec: 500000000
  lifespan:
    sec: 0
    nsec: 0
  liveliness: automatic
  liveliness_lease_duration:
    sec: 0
    nsec: 0
  avoid_ros_namespace_conventions: false
  vendor_extension: preserved
"""

_NUMERIC_QOS = """\
- history: 1
  depth: 5
  reliability: 2
  durability: 2
  deadline: {sec: 0, nsec: 500000000}
  lifespan: {sec: 0, nsec: 0}
  liveliness: 1
  liveliness_lease_duration: {sec: 0, nsec: 0}
  avoid_ros_namespace_conventions: false
"""


def test_qos_profiles_to_numeric_converts_policy_names_only() -> None:
    converted = yaml.safe_load(qos_profiles_to_numeric(_MODERN_QOS))

    assert converted == [
        {
            "history": 1,
            "depth": 5,
            "reliability": 2,
            "durability": 2,
            "deadline": {"sec": 0, "nsec": 500_000_000},
            "lifespan": {"sec": 0, "nsec": 0},
            "liveliness": 1,
            "liveliness_lease_duration": {"sec": 0, "nsec": 0},
            "avoid_ros_namespace_conventions": False,
            "vendor_extension": "preserved",
        }
    ]


@pytest.mark.parametrize(
    ("field", "name", "code"),
    [
        ("history", "system_default", 0),
        ("history", "keep_last", 1),
        ("history", "keep_all", 2),
        ("history", "unknown", 3),
        ("reliability", "system_default", 0),
        ("reliability", "reliable", 1),
        ("reliability", "best_effort", 2),
        ("reliability", "unknown", 3),
        ("durability", "system_default", 0),
        ("durability", "transient_local", 1),
        ("durability", "volatile", 2),
        ("durability", "unknown", 3),
        ("liveliness", "system_default", 0),
        ("liveliness", "automatic", 1),
        ("liveliness", "manual_by_node", 2),
        ("liveliness", "manual_by_topic", 3),
        ("liveliness", "unknown", 4),
    ],
)
def test_qos_profiles_to_numeric_maps_every_humble_policy(field: str, name: str, code: int) -> None:
    profile: dict[str, str | int] = {
        "history": 1,
        "reliability": 1,
        "durability": 2,
        "liveliness": 1,
    }
    profile[field] = name

    converted = yaml.safe_load(qos_profiles_to_numeric(yaml.safe_dump([profile])))[0]

    assert converted[field] == code


def test_qos_profiles_to_numeric_accepts_mixed_profiles() -> None:
    raw = yaml.safe_dump(
        [
            {
                "history": "keep_all",
                "reliability": 1,
                "durability": "transient_local",
                "liveliness": 3,
            },
            {
                "history": 0,
                "reliability": "system_default",
                "durability": 3,
                "liveliness": "unknown",
            },
        ],
        sort_keys=False,
    )

    converted = yaml.safe_load(qos_profiles_to_numeric(raw))

    assert converted[0] == {
        "history": 2,
        "reliability": 1,
        "durability": 1,
        "liveliness": 3,
    }
    assert converted[1] == {
        "history": 0,
        "reliability": 0,
        "durability": 3,
        "liveliness": 4,
    }


def test_qos_profiles_to_numeric_leaves_numeric_yaml_byte_identical() -> None:
    raw = "- history: 1\n  reliability: 2\n  durability: 2\n  liveliness: 1\n"

    assert qos_profiles_to_numeric(raw) == raw


@pytest.mark.parametrize("raw", ["", "[]"])
def test_qos_profiles_to_numeric_accepts_empty_profiles(raw: str) -> None:
    assert qos_profiles_to_numeric(raw) == raw


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (": not valid yaml ::", "malformed"),
        ("history: keep_last", "YAML list"),
        ("- keep_last", "profile 0"),
        (
            yaml.safe_dump(
                [
                    {
                        "history": "keep_last",
                        "reliability": "best_available",
                        "durability": "volatile",
                        "liveliness": "automatic",
                    }
                ]
            ),
            "best_available",
        ),
        (
            yaml.safe_dump(
                [
                    {
                        "history": True,
                        "reliability": 1,
                        "durability": 2,
                        "liveliness": 1,
                    }
                ]
            ),
            "history",
        ),
        (
            yaml.safe_dump(
                [
                    {
                        "history": 1,
                        "reliability": 4,
                        "durability": 2,
                        "liveliness": 1,
                    }
                ]
            ),
            "reliability",
        ),
    ],
)
def test_qos_profiles_to_numeric_rejects_incompatible_values(raw: str, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        qos_profiles_to_numeric(raw)


def _channel(channel_id: int, metadata: dict[str, str]) -> Channel:
    return Channel(
        id=channel_id,
        schema_id=1,
        topic="/camera",
        message_encoding="cdr",
        metadata=metadata,
    )


def test_qos_metadata_processor_rewrites_only_offered_profiles() -> None:
    processor = QosMetadataProcessor(qos_format="numeric")
    channel = _channel(
        1,
        {
            "offered_qos_profiles": _MODERN_QOS,
            "subscribed_qos_profiles": _MODERN_QOS,
            "custom": "value",
        },
    )

    assert processor.on_channel(channel_context(channel), channel, None) is Action.CONTINUE

    converted = yaml.safe_load(channel.metadata["offered_qos_profiles"])[0]
    assert converted["history"] == 1
    assert channel.metadata["subscribed_qos_profiles"] == _MODERN_QOS
    assert channel.metadata["custom"] == "value"


def test_qos_metadata_processor_verifies_every_chunk() -> None:
    processor = QosMetadataProcessor(qos_format="numeric")
    changed = _channel(1, {"offered_qos_profiles": _MODERN_QOS})
    numeric = _channel(
        2,
        {
            "offered_qos_profiles": (
                "- history: 1\n  reliability: 1\n  durability: 2\n  liveliness: 1\n"
            )
        },
    )

    processor.on_channel(channel_context(changed, stream_id=2), changed, None)
    processor.on_channel(channel_context(numeric, stream_id=3), numeric, None)

    chunk = lazy_chunk(0, 100)
    assert processor.on_chunk(chunk_context(stream_id=2), chunk) is ChunkDecision.DECODE_VERIFY
    assert processor.on_chunk(chunk_context(stream_id=3), chunk) is ChunkDecision.DECODE_VERIFY
    assert processor.on_chunk(chunk_context(stream_id=4), chunk) is ChunkDecision.DECODE_VERIFY


def test_qos_metadata_processor_adds_topic_context_to_errors() -> None:
    processor = QosMetadataProcessor(qos_format="numeric")
    channel = _channel(1, {"offered_qos_profiles": "- history: best_available"})

    with pytest.raises(ValueError, match="/camera"):
        processor.on_channel(channel_context(channel), channel, None)


def test_qos_override_yaml_uses_ros_topic_mapping_shape() -> None:
    overrides = parse_qos_override_yaml(
        """\
/camera/image:
  reliability: best_effort
  history: keep_last
  depth: 3
/tf_static:
  durability: transient_local
"""
    )

    assert overrides == {
        "/camera/image": {
            "reliability": "best_effort",
            "history": "keep_last",
            "depth": 3,
        },
        "/tf_static": {"durability": "transient_local"},
    }


def test_qos_set_rule_parses_regex_field_and_yaml_scalar() -> None:
    assert parse_qos_set_rule(r"/(?:camera|lidar)/.*:reliability=best_effort") == (
        r"/(?:camera|lidar)/.*",
        "reliability",
        "best_effort",
    )
    assert parse_qos_set_rule(r"/camera/.*:depth=3") == (r"/camera/.*", "depth", 3)


def test_qos_metadata_processor_overrides_every_profile_with_ordered_precedence() -> None:
    raw = yaml.safe_dump(
        [
            {
                "history": "keep_last",
                "depth": 10,
                "reliability": "reliable",
                "durability": "volatile",
                "liveliness": "automatic",
                "vendor_extension": "first",
            },
            {
                "history": "keep_all",
                "depth": 0,
                "reliability": "reliable",
                "durability": "transient_local",
                "liveliness": "manual_by_topic",
                "vendor_extension": "second",
            },
        ],
        sort_keys=False,
    )
    processor = QosMetadataProcessor(
        qos_format="preserve",
        topic_overrides={"/camera/image": {"reliability": "reliable", "depth": 5}},
        set_rules=[
            (r"/camera/.*", "reliability", "best_effort"),
            (r"/camera/image", "depth", 3),
        ],
    )
    channel = Channel(
        1,
        1,
        "/camera/image",
        "cdr",
        {"offered_qos_profiles": raw, "custom": "value"},
    )

    processor.on_channel(channel_context(channel), channel, None)

    profiles = yaml.safe_load(channel.metadata["offered_qos_profiles"])
    assert [profile["reliability"] for profile in profiles] == ["best_effort"] * 2
    assert [profile["depth"] for profile in profiles] == [3, 3]
    assert [profile["vendor_extension"] for profile in profiles] == ["first", "second"]
    assert channel.metadata["custom"] == "value"


def test_qos_metadata_processor_set_regex_uses_full_match() -> None:
    processor = QosMetadataProcessor(
        qos_format="preserve",
        set_rules=[(r"/camera", "reliability", "best_effort")],
    )
    channel = Channel(
        1,
        1,
        "/camera/image",
        "cdr",
        {"offered_qos_profiles": _MODERN_QOS},
    )

    processor.on_channel(channel_context(channel), channel, None)

    assert channel.metadata["offered_qos_profiles"] == _MODERN_QOS


def test_qos_metadata_processor_applies_override_before_numeric_conversion() -> None:
    processor = QosMetadataProcessor(
        qos_format="numeric",
        set_rules=[(r"/camera", "reliability", "reliable")],
    )
    channel = _channel(1, {"offered_qos_profiles": _MODERN_QOS})

    processor.on_channel(channel_context(channel), channel, None)

    profile = yaml.safe_load(channel.metadata["offered_qos_profiles"])[0]
    assert profile["reliability"] == 1
    assert profile["history"] == 1


def test_qos_metadata_processor_rejects_override_without_recorded_profile() -> None:
    processor = QosMetadataProcessor(
        qos_format="preserve",
        set_rules=[(r"/camera", "reliability", "best_effort")],
    )
    channel = _channel(1, {})

    with pytest.raises(ValueError, match="no offered_qos_profiles"):
        processor.on_channel(channel_context(channel), channel, None)


def test_process_numeric_qos_preserves_messages_and_other_metadata(tmp_path) -> None:
    source = tmp_path / "jazzy.mcap"
    output = tmp_path / "humble.mcap"
    payloads = [b"first", b"second"]
    with source.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=1024)
        writer.start(profile="ros2", library="qos-test")
        writer.add_schema(1, "std_msgs/msg/String", "ros2msg", b"string data")
        writer.add_channel(
            1,
            "/camera",
            "cdr",
            1,
            metadata={"offered_qos_profiles": _MODERN_QOS, "custom": "value"},
        )
        for timestamp, payload in enumerate(payloads, start=10):
            writer.add_message(1, timestamp, payload, publish_time=timestamp + 100)
        writer.finish()

    assert (
        process_cmd.process(
            file=[str(source)],
            output=output,
            qos_format="numeric",
            no_clobber=True,
        )
        == 0
    )

    with output.open("rb") as stream:
        summary = get_summary(stream)
    assert summary is not None
    channel = next(iter(summary.channels.values()))
    converted = yaml.safe_load(channel.metadata["offered_qos_profiles"])[0]
    assert converted["history"] == 1
    assert converted["reliability"] == 2
    assert converted["durability"] == 2
    assert converted["liveliness"] == 1
    assert channel.metadata["custom"] == "value"

    with output.open("rb") as stream:
        messages = list(read_message(stream))
    assert [message.data for _schema, _channel, message in messages] == payloads
    assert [message.log_time for _schema, _channel, message in messages] == [10, 11]
    assert [message.publish_time for _schema, _channel, message in messages] == [110, 111]


def test_process_embeds_file_and_repeatable_qos_overrides(tmp_path) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "overridden.mcap"
    override = tmp_path / "qos.yaml"
    override.write_text(
        """\
/camera/front:
  reliability: reliable
  depth: 3
"""
    )
    with source.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=1024)
        writer.start(profile="ros2", library="qos-override-test")
        writer.add_schema(1, "std_msgs/msg/String", "ros2msg", b"string data")
        writer.add_channel(
            1,
            "/camera/front",
            "cdr",
            1,
            metadata={"offered_qos_profiles": _MODERN_QOS},
        )
        writer.add_channel(
            2,
            "/imu",
            "cdr",
            1,
            metadata={"offered_qos_profiles": _MODERN_QOS},
        )
        writer.add_message(1, 10, b"camera", publish_time=10)
        writer.add_message(2, 11, b"imu", publish_time=11)
        writer.finish()

    assert (
        process_cmd.process(
            file=[str(source)],
            output=output,
            qos_override=override,
            qos_set=[
                r"/camera/.*:reliability=best_effort",
                r"/camera/front:durability=transient_local",
            ],
            qos_format="numeric",
            no_clobber=True,
        )
        == 0
    )

    with output.open("rb") as stream:
        summary = get_summary(stream)
    assert summary is not None
    channels = {channel.topic: channel for channel in summary.channels.values()}
    camera_profile = yaml.safe_load(channels["/camera/front"].metadata["offered_qos_profiles"])[0]
    imu_profile = yaml.safe_load(channels["/imu"].metadata["offered_qos_profiles"])[0]
    assert camera_profile["reliability"] == 2
    assert camera_profile["durability"] == 1
    assert camera_profile["depth"] == 3
    assert imu_profile["reliability"] == 2
    assert imu_profile["durability"] == 2

    with output.open("rb") as stream:
        messages = list(read_message(stream))
    assert [message.data for _schema, _channel, message in messages] == [b"camera", b"imu"]


def test_process_numeric_qos_rewrites_channel_after_message_inside_chunk(tmp_path) -> None:
    source = tmp_path / "interleaved.mcap"
    output = tmp_path / "numeric.mcap"
    channel = Channel(1, 1, "/camera", "cdr", {"offered_qos_profiles": _MODERN_QOS})
    message = Message(1, 0, 10, 20, b"payload")
    chunk_buffer = io.BytesIO()
    message.write_record_to(chunk_buffer)
    channel.write_record_to(chunk_buffer)
    chunk_data = chunk_buffer.getvalue()

    with source.open("wb") as stream:
        writer = McapWriter(
            stream,
            use_chunking=False,
            compression=CompressionType.NONE,
        )
        writer.start(profile="ros2", library="interleaved-test")
        writer.add_schema(1, "std_msgs/msg/String", "ros2msg", b"string data")
        writer.add_channel(
            channel.id,
            channel.topic,
            channel.message_encoding,
            channel.schema_id,
            channel.metadata,
        )
        writer.add_chunk(
            Chunk(10, 10, len(chunk_data), zlib.crc32(chunk_data), "", chunk_data),
            {1: MessageIndex(1, [10], [0])},
        )
        writer.finish()

    assert (
        process_cmd.process(
            file=[str(source)],
            output=output,
            qos_format="numeric",
            no_clobber=True,
        )
        == 0
    )

    with output.open("rb") as stream:
        chunks = [
            record
            for record in stream_reader(stream, emit_chunks=True)
            if isinstance(record, Chunk)
        ]
    assert len(chunks) == 1
    embedded_channels = [
        record for record in breakup_chunk(chunks[0]) if isinstance(record, Channel)
    ]
    assert embedded_channels == []

    with output.open("rb") as stream:
        messages = list(read_message(stream))
    assert len(messages) == 1
    assert messages[0][2].data == b"payload"


def test_process_numeric_qos_finds_unreferenced_channel_inside_chunk(tmp_path) -> None:
    source = tmp_path / "hidden-channel.mcap"
    output = tmp_path / "numeric.mcap"
    visible_channel = Channel(1, 1, "/visible", "cdr", {"offered_qos_profiles": _NUMERIC_QOS})
    hidden_channel = Channel(2, 1, "/hidden", "cdr", {"offered_qos_profiles": _MODERN_QOS})
    message = Message(1, 0, 10, 20, b"payload")
    chunk_buffer = io.BytesIO()
    message.write_record_to(chunk_buffer)
    hidden_channel.write_record_to(chunk_buffer)
    chunk_data = chunk_buffer.getvalue()

    with source.open("wb") as stream:
        writer = McapWriter(stream, use_chunking=False, compression=CompressionType.NONE)
        writer.start(profile="ros2", library="hidden-channel-test")
        writer.add_schema(1, "std_msgs/msg/String", "ros2msg", b"string data")
        writer.add_channel(
            visible_channel.id,
            visible_channel.topic,
            visible_channel.message_encoding,
            visible_channel.schema_id,
            visible_channel.metadata,
        )
        writer.add_chunk(
            Chunk(10, 10, len(chunk_data), zlib.crc32(chunk_data), "", chunk_data),
            {1: MessageIndex(1, [10], [0])},
        )
        writer.finish()

    assert (
        process_cmd.process(
            file=[str(source)],
            output=output,
            qos_format="numeric",
            no_clobber=True,
        )
        == 0
    )

    with output.open("rb") as stream:
        chunks = [
            record
            for record in stream_reader(stream, emit_chunks=True)
            if isinstance(record, Chunk)
        ]
    assert len(chunks) == 1
    assert not any(isinstance(record, Channel) for record in breakup_chunk(chunks[0]))
