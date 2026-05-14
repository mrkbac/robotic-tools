"""Generate test MCAP files for testing and benchmarking."""

import io
from pathlib import Path

from mcap_ros2_support_fast import ROS2EncoderFactory
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, NS_TO_MS, NS_TO_SEC
from small_mcap import CompressionType, McapWriter


def create_simple_mcap(
    num_messages: int = 200,
    chunk_size: int = 1024 * 1024,  # 1MB
    compression: CompressionType = CompressionType.ZSTD,
) -> bytes:
    """Create a simple MCAP file with one topic."""
    output = io.BytesIO()
    writer = McapWriter(output, chunk_size=chunk_size, compression=compression)
    writer.start()

    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=1)

    for i in range(num_messages):
        writer.add_message(
            channel_id=1,
            log_time=i * NS_TO_MS,
            data=f'{{"i": {i}}}'.encode(),
            publish_time=i * NS_TO_MS,
        )

    writer.finish()
    return output.getvalue()


def create_multi_topic_mcap(
    topics: list[str],
    messages_per_topic: int = 100,
    chunk_size: int = 1024 * 1024,
    compression: CompressionType = CompressionType.ZSTD,
) -> bytes:
    """Create an MCAP file with multiple topics."""
    output = io.BytesIO()
    writer = McapWriter(output, chunk_size=chunk_size, compression=compression)
    writer.start()

    # Add schema and channels for each topic
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")

    for i, topic in enumerate(topics, start=1):
        writer.add_channel(channel_id=i, topic=topic, message_encoding="json", schema_id=1)

    # Interleave messages from different topics
    for msg_idx in range(messages_per_topic):
        for channel_id in range(1, len(topics) + 1):
            log_time = (msg_idx * len(topics) + channel_id - 1) * NS_TO_MS
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=f'{{"msg": {msg_idx}}}'.encode(),
                publish_time=log_time,
            )

    writer.finish()
    return output.getvalue()


_TRANSIENT_LOCAL_QOS_YAML = """\
- history: 3
  depth: 0
  reliability: 1
  durability: transient_local
  deadline:
    sec: 0
    nsec: 0
"""


def create_latched_topic_mcap(
    *,
    latched_topic: str = "/tf_static",
    latched_log_time: int = 0,
    latched_update_times: list[int] | None = None,
    other_topic: str = "/scan",
    other_messages: int = 10,
    other_step_ns: int = NS_TO_SEC,
    chunk_size: int = 1024,
    compression: CompressionType = CompressionType.NONE,
) -> bytes:
    """MCAP with a latched topic published once at ``latched_log_time`` and a
    second non-latched topic publishing ``other_messages`` messages at fixed
    intervals starting at ``latched_log_time`` (one message per step).

    The latched channel carries the standard ROS 2 ``offered_qos_profiles``
    metadata blob with ``durability: transient_local`` so opt-in
    ``--latch-from-metadata`` autodetection picks it up.
    """
    output = io.BytesIO()
    writer = McapWriter(output, chunk_size=chunk_size, compression=compression)
    writer.start()

    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(
        channel_id=1,
        topic=latched_topic,
        message_encoding="json",
        schema_id=1,
        metadata={"offered_qos_profiles": _TRANSIENT_LOCAL_QOS_YAML},
    )
    writer.add_channel(channel_id=2, topic=other_topic, message_encoding="json", schema_id=1)

    events: list[tuple[int, int, bytes]] = [
        (latched_log_time, 1, b'{"latched": true}'),
    ]
    for i, ts in enumerate(latched_update_times or []):
        events.append((ts, 1, f'{{"latched": true, "update": {i}}}'.encode()))

    for i in range(other_messages):
        ts = latched_log_time + (i + 1) * other_step_ns
        events.append((ts, 2, f'{{"i": {i}}}'.encode()))

    for ts, channel_id, data in sorted(events, key=lambda event: event[0]):
        writer.add_message(
            channel_id=channel_id,
            log_time=ts,
            publish_time=ts,
            data=data,
        )

    writer.finish()
    return output.getvalue()


_TF_MESSAGE_SCHEMA = """\
geometry_msgs/TransformStamped[] transforms

================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation

================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w"""


def _tf_message(
    transforms: list[tuple[str, str, tuple[float, float, float]]],
    *,
    stamp_ns: int = 0,
) -> dict:
    sec = stamp_ns // NS_TO_SEC
    nanosec = stamp_ns % NS_TO_SEC
    return {
        "transforms": [
            {
                "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": parent},
                "child_frame_id": child,
                "transform": {
                    "translation": {"x": tx, "y": ty, "z": tz},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            }
            for parent, child, (tx, ty, tz) in transforms
        ]
    }


def create_tf_mcap(
    *,
    static_edges: list[tuple[str, str, tuple[float, float, float]]] | None = None,
    dynamic_edges: list[tuple[str, str, tuple[float, float, float]]] | None = None,
    dynamic_samples: int = 0,
    dynamic_stamp_offset_ns: int = 0,
    compression: CompressionType = CompressionType.NONE,
) -> bytes:
    """Write a small MCAP with /tf_static (and optionally /tf) ROS 2 messages.

    `static_edges` and `dynamic_edges` each list `(parent, child, (tx, ty, tz))`
    tuples; rotation is identity. Dynamic edges are repeated `dynamic_samples`
    times at successive 100 ms steps.
    """
    static_edges = static_edges or []
    dynamic_edges = dynamic_edges or []

    output = io.BytesIO()
    writer = McapWriter(
        output,
        chunk_size=1024,
        compression=compression,
        encoder_factory=ROS2EncoderFactory(),
    )
    writer.start()

    writer.add_schema(
        schema_id=1,
        name="tf2_msgs/msg/TFMessage",
        encoding="ros2msg",
        data=_TF_MESSAGE_SCHEMA.encode(),
    )

    if static_edges:
        writer.add_channel(
            channel_id=1,
            topic="/tf_static",
            message_encoding="cdr",
            schema_id=1,
        )
        writer.add_message_encode(
            channel_id=1,
            log_time=0,
            publish_time=0,
            data=_tf_message(static_edges),
        )

    if dynamic_edges and dynamic_samples > 0:
        writer.add_channel(
            channel_id=2,
            topic="/tf",
            message_encoding="cdr",
            schema_id=1,
        )
        step_ns = 100 * NS_TO_MS
        for i in range(dynamic_samples):
            log_time = (i + 1) * step_ns
            writer.add_message_encode(
                channel_id=2,
                log_time=log_time,
                publish_time=log_time,
                data=_tf_message(
                    dynamic_edges,
                    stamp_ns=dynamic_stamp_offset_ns + log_time,
                ),
            )

    writer.finish()
    return output.getvalue()


def create_corrupt_mcap(corruption_type: str = "truncated") -> bytes:
    """Create a corrupt MCAP file for testing recovery."""
    # Create a file with multiple chunks to test recovery
    # Use large messages and small chunk size to ensure chunking happens
    output = io.BytesIO()
    writer = McapWriter(output, chunk_size=8192, compression=CompressionType.ZSTD)  # 8KB chunks
    writer.start()

    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=1)

    # Create messages with larger data to exceed chunk size
    # Each message ~1KB, so we'll get multiple chunks with 100 messages
    large_data = b"x" * 1000
    for i in range(100):
        writer.add_message(
            channel_id=1,
            log_time=i * NS_TO_MS,
            data=f'{{"i": {i}, "data": "{large_data.decode()}"}}'.encode(),
            publish_time=i * NS_TO_MS,
        )

    writer.finish()
    data = output.getvalue()

    if corruption_type == "truncated":
        # Truncate the file to cut off some chunks and the footer
        return data[: len(data) // 2]
    if corruption_type == "bad_crc":
        # Corrupt a byte in the middle
        data_array = bytearray(data)
        data_array[len(data) // 2] = (data_array[len(data) // 2] + 1) % 256
        return bytes(data_array)
    return data


def create_large_mcap(
    target_size_mb: int = 10,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compression: CompressionType = CompressionType.ZSTD,
) -> bytes:
    """Create a large MCAP file for performance testing."""
    # Estimate messages needed (rough approximation)
    message_data_size = 100  # Average message size in bytes
    estimated_messages = (target_size_mb * 1024 * 1024) // message_data_size

    return create_multi_topic_mcap(
        topics=["/camera/image", "/lidar/points", "/gps/position", "/imu/data"],
        messages_per_topic=estimated_messages // 4,
        chunk_size=chunk_size,
        compression=compression,
    )


def save_fixture(name: str, data: bytes, fixtures_dir: Path | None = None) -> Path:
    """Save MCAP data to fixtures directory."""
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent
    filepath = fixtures_dir / f"{name}.mcap"
    filepath.write_bytes(data)
    return filepath


def ensure_fixtures(fixtures_dir: Path | None = None) -> dict[str, Path]:
    """Ensure all test fixtures exist and return their paths."""
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent

    fixtures = {}

    # Simple file
    fixtures["simple"] = save_fixture("simple", create_simple_mcap(num_messages=100), fixtures_dir)

    # Multi-topic file
    fixtures["multi_topic"] = save_fixture(
        "multi_topic",
        create_multi_topic_mcap(
            topics=["/camera/front", "/camera/back", "/lidar/points", "/debug/log"],
            messages_per_topic=50,
        ),
        fixtures_dir,
    )

    # Corrupt files
    fixtures["truncated"] = save_fixture(
        "corrupt_truncated", create_corrupt_mcap("truncated"), fixtures_dir
    )

    fixtures["bad_crc"] = save_fixture(
        "corrupt_bad_crc", create_corrupt_mcap("bad_crc"), fixtures_dir
    )

    # Uncompressed file
    fixtures["uncompressed"] = save_fixture(
        "uncompressed",
        create_simple_mcap(num_messages=100, compression=CompressionType.NONE),
        fixtures_dir,
    )

    # LZ4 compressed file
    fixtures["lz4_compressed"] = save_fixture(
        "lz4_compressed",
        create_simple_mcap(num_messages=100, compression=CompressionType.LZ4),
        fixtures_dir,
    )

    # Large file for benchmarking (1MB)
    fixtures["large_1mb"] = save_fixture(
        "large_1mb", create_large_mcap(target_size_mb=1), fixtures_dir
    )

    # Even larger file for benchmarking (10MB)
    fixtures["large_10mb"] = save_fixture(
        "large_10mb", create_large_mcap(target_size_mb=10), fixtures_dir
    )

    return fixtures


if __name__ == "__main__":
    # Generate all fixtures when run directly
    fixtures_dir = Path(__file__).parent
    fixtures = ensure_fixtures(fixtures_dir)
    print("Generated test fixtures:")
    for name, path in fixtures.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:20s} -> {path.name:30s} ({size_kb:.1f} KB)")
