"""E2E tests for roscompress and rosdecompress roundtrip.

Verifies that compress → decompress preserves:
- Exact message count (per topic and total)
- Monotonically increasing timestamps
- Correct timestamp order matching input
- Topic names
- Non-image messages passed through unchanged
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from small_mcap import McapWriter, get_summary
from small_mcap.reader import read_message

from pymcap_cli.cmd.roscompress_cmd import roscompress
from pymcap_cli.cmd.rosdecompress_cmd import rosdecompress
from pymcap_cli.encoding.encoder_common import EncoderMode
from tests.fixtures.image_mcap_generator import (
    SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA,
    SENSOR_MSGS_IMAGE_SCHEMA,
    create_jpeg_frame,
    create_simple_rgb_frame,
)

# Schema for a non-image topic (pass-through).
_STRING_SCHEMA = "string data"


def _read_messages(path: Path) -> list[dict[str, Any]]:
    """Read all messages (raw), returning dicts with topic, log_time, publish_time, schema."""
    results: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for schema, channel, message in read_message(f):
            results.append(
                {
                    "topic": channel.topic,
                    "log_time": message.log_time,
                    "publish_time": message.publish_time,
                    "schema": schema.name if schema else "",
                }
            )
    return results


def _get_message_count(path: Path) -> int:
    with path.open("rb") as f:
        s = get_summary(f)
        if s and s.statistics and s.statistics.channel_message_counts:
            return sum(s.statistics.channel_message_counts.values())
    return 0


def _get_topic_counts(path: Path) -> dict[str, int]:
    """Return {topic: message_count} from MCAP summary."""
    with path.open("rb") as f:
        s = get_summary(f)
        if s and s.statistics and s.statistics.channel_message_counts:
            return {
                s.channels[ch_id].topic: count
                for ch_id, count in s.statistics.channel_message_counts.items()
            }
    return {}


def _roundtrip(
    input_path: Path,
    tmp_path: Path,
    backend: EncoderMode,
) -> tuple[Path, Path]:
    """Compress then decompress, return (compressed_path, decompressed_path)."""
    compressed = tmp_path / "compressed.mcap"
    decompressed = tmp_path / "decompressed.mcap"

    roscompress(
        file=str(input_path),
        output=compressed,
        force=True,
        encoder="libx264",
        backend=backend,
        pointcloud=False,
    )
    rosdecompress(
        file=str(compressed),
        output=decompressed,
        force=True,
        video_format="raw",
        backend=backend,
        pointcloud=False,
    )
    return compressed, decompressed


def _assert_timestamps_monotonic(messages: list[dict[str, Any]]) -> None:
    log_times = [m["log_time"] for m in messages]
    for i in range(1, len(log_times)):
        assert log_times[i] >= log_times[i - 1], (
            f"log_time not monotonic at index {i}: {log_times[i - 1]} > {log_times[i]}"
        )


# ---------------------------------------------------------------------------
# Mixed-input fixture: CompressedImage + raw Image + non-image topic
# ---------------------------------------------------------------------------


def _create_mixed_mcap(num_frames: int = 20) -> bytes:
    """Create an MCAP with 3 topics: CompressedImage, raw Image, and a string topic."""
    output = io.BytesIO()
    writer = McapWriter(
        output,
        chunk_size=1024 * 1024,
        encoder_factory=ROS2EncoderFactory(),
    )
    writer.start()

    # Register schemas and channels.
    writer.add_schema(1, "sensor_msgs/msg/CompressedImage", "ros2msg", SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode())
    writer.add_schema(2, "sensor_msgs/msg/Image", "ros2msg", SENSOR_MSGS_IMAGE_SCHEMA.encode())
    writer.add_schema(3, "std_msgs/msg/String", "ros2msg", _STRING_SCHEMA.encode())

    writer.add_channel(1, "/camera/left", "cdr", 1)
    writer.add_channel(2, "/camera/right", "cdr", 2)
    writer.add_channel(3, "/status", "cdr", 3)

    fps = 30
    time_step_ns = int(1e9 / fps)

    for i in range(num_frames):
        log_time = i * time_step_ns
        sec = log_time // int(1e9)
        nanosec = log_time % int(1e9)
        header = {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": "camera"}

        # CompressedImage on /camera/left
        writer.add_message_encode(
            channel_id=1,
            log_time=log_time,
            publish_time=log_time,
            data={"header": header, "format": "jpeg", "data": create_jpeg_frame(160, 120, i)},
        )

        # Raw Image on /camera/right
        writer.add_message_encode(
            channel_id=2,
            log_time=log_time + 1,  # +1ns to maintain strict ordering
            publish_time=log_time + 1,
            data={
                "header": header,
                "height": 160,
                "width": 120,
                "encoding": "rgb8",
                "is_bigendian": False,
                "step": 120 * 3,
                "data": create_simple_rgb_frame(120, 160, i),
            },
        )

        # String message on /status (every 5 frames)
        if i % 5 == 0:
            writer.add_message_encode(
                channel_id=3,
                log_time=log_time + 2,
                publish_time=log_time + 2,
                data={"data": f"frame_{i}"},
            )

    writer.finish()
    return output.getvalue()


@pytest.fixture
def mixed_mcap(tmp_path: Path) -> Path:
    """MCAP with CompressedImage + raw Image + non-image topics."""
    path = tmp_path / "mixed.mcap"
    path.write_bytes(_create_mixed_mcap(num_frames=20))
    return path


# ---------------------------------------------------------------------------
# Single-topic tests
# ---------------------------------------------------------------------------


class TestRoscompressRoundtrip:
    """Test that roscompress → rosdecompress preserves message count and timing."""

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_compressed_image_roundtrip(
        self, image_compressed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        input_count = _get_message_count(image_compressed_mcap)
        compressed, decompressed = _roundtrip(image_compressed_mcap, tmp_path, backend)

        assert _get_message_count(compressed) == input_count
        assert _get_message_count(decompressed) == input_count

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_raw_image_roundtrip(
        self, image_rgb_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        input_count = _get_message_count(image_rgb_mcap)
        compressed, decompressed = _roundtrip(image_rgb_mcap, tmp_path, backend)

        assert _get_message_count(compressed) == input_count
        assert _get_message_count(decompressed) == input_count

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_timestamps_match_input(
        self, image_compressed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        input_times = [m["log_time"] for m in _read_messages(image_compressed_mcap)]
        compressed, decompressed = _roundtrip(image_compressed_mcap, tmp_path, backend)

        compressed_times = [m["log_time"] for m in _read_messages(compressed)]
        assert compressed_times == input_times

        decompressed_times = [m["log_time"] for m in _read_messages(decompressed)]
        assert decompressed_times == input_times

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_timestamps_monotonic(
        self, image_compressed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        compressed, decompressed = _roundtrip(image_compressed_mcap, tmp_path, backend)
        _assert_timestamps_monotonic(_read_messages(compressed))
        _assert_timestamps_monotonic(_read_messages(decompressed))

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_topic_preserved(
        self, image_compressed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        input_topics = {m["topic"] for m in _read_messages(image_compressed_mcap)}
        compressed, decompressed = _roundtrip(image_compressed_mcap, tmp_path, backend)

        assert {m["topic"] for m in _read_messages(compressed)} == input_topics
        assert {m["topic"] for m in _read_messages(decompressed)} == input_topics


# ---------------------------------------------------------------------------
# Mixed-input tests
# ---------------------------------------------------------------------------


class TestRoscompressMixedInput:
    """Test with multiple image topics + non-image pass-through."""

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_mixed_total_message_count(
        self, mixed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        """Total message count must be preserved through roundtrip."""
        input_count = _get_message_count(mixed_mcap)
        compressed, decompressed = _roundtrip(mixed_mcap, tmp_path, backend)

        assert _get_message_count(compressed) == input_count
        assert _get_message_count(decompressed) == input_count

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_mixed_per_topic_message_count(
        self, mixed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        """Per-topic message count must be preserved."""
        input_counts = _get_topic_counts(mixed_mcap)
        compressed, _ = _roundtrip(mixed_mcap, tmp_path, backend)
        compressed_counts = _get_topic_counts(compressed)

        for topic, expected in input_counts.items():
            assert compressed_counts.get(topic, 0) == expected, (
                f"Topic {topic}: {compressed_counts.get(topic, 0)} != {expected}"
            )

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_mixed_passthrough_preserved(
        self, mixed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        """Non-image messages must be passed through unchanged."""
        compressed, _ = _roundtrip(mixed_mcap, tmp_path, backend)
        compressed_msgs = _read_messages(compressed)

        status_msgs = [m for m in compressed_msgs if m["topic"] == "/status"]
        # 20 frames, string every 5 = 4 messages
        assert len(status_msgs) == 4

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_mixed_timestamps_monotonic(
        self, mixed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        compressed, decompressed = _roundtrip(mixed_mcap, tmp_path, backend)
        _assert_timestamps_monotonic(_read_messages(compressed))
        _assert_timestamps_monotonic(_read_messages(decompressed))

    @pytest.mark.parametrize("backend", [EncoderMode.PYAV, EncoderMode.FFMPEG_CLI])
    def test_mixed_all_topics_present(
        self, mixed_mcap: Path, tmp_path: Path, backend: EncoderMode
    ):
        """All topics from input must appear in output."""
        input_topics = {m["topic"] for m in _read_messages(mixed_mcap)}
        compressed, decompressed = _roundtrip(mixed_mcap, tmp_path, backend)

        assert {m["topic"] for m in _read_messages(compressed)} == input_topics
        assert {m["topic"] for m in _read_messages(decompressed)} == input_topics
