"""Generate test MCAP files for testing and benchmarking."""

import io
from pathlib import Path

from small_mcap.writer import CompressionType, McapWriter


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
            log_time=i * 1_000_000,
            data=f'{{"i": {i}}}'.encode(),
            publish_time=i * 1_000_000,
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
            log_time = (msg_idx * len(topics) + channel_id - 1) * 1_000_000
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=f'{{"msg": {msg_idx}}}'.encode(),
                publish_time=log_time,
            )

    writer.finish()
    return output.getvalue()


def create_corrupt_mcap(corruption_type: str = "truncated") -> bytes:
    """Create a corrupt MCAP file for testing recovery."""
    # First create a valid file
    data = create_simple_mcap(num_messages=100)

    if corruption_type == "truncated":
        # Truncate the file (remove footer)
        return data[: len(data) // 2]
    if corruption_type == "bad_crc":
        # Corrupt a byte in the middle
        data_array = bytearray(data)
        data_array[len(data) // 2] = (data_array[len(data) // 2] + 1) % 256
        return bytes(data_array)
    return data


def create_large_mcap(
    target_size_mb: int = 10,
    chunk_size: int = 4 * 1024 * 1024,
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
    print("Generated test fixtures:")  # noqa: T201
    for name, path in fixtures.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:20s} -> {path.name:30s} ({size_kb:.1f} KB)")  # noqa: T201
