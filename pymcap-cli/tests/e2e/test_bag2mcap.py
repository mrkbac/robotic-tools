"""End-to-end tests for the bag2mcap command."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.bag2mcap_cmd import Bag2McapOptions, convert_bag_to_mcap
from pymcap_cli.rosbag_reader import read_bag_messages
from small_mcap.reader import read_message

if TYPE_CHECKING:
    from small_mcap.records import Channel, Message, Schema

from ..fixtures.bag_generator import (
    generate_multi_topic_bag,
    generate_simple_bag,
)

_NSEC_PER_SEC = 1_000_000_000

# Real bag fixtures from data/mcap/ (written by actual ROS tooling)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEMO_BAG = _REPO_ROOT / "data" / "mcap" / "testdata" / "bags" / "demo.bag"
_MARKERS_BAG = _REPO_ROOT / "data" / "mcap" / "go" / "ros" / "testdata" / "markers.bag"
_MARKERS_BZ2_BAG = _REPO_ROOT / "data" / "mcap" / "go" / "ros" / "testdata" / "markers.bz2.bag"


@pytest.fixture
def simple_bag(tmp_path: Path) -> Path:
    bag_path = tmp_path / "simple.bag"
    bag_path.write_bytes(generate_simple_bag())
    return bag_path


@pytest.fixture
def multi_topic_bag(tmp_path: Path) -> Path:
    bag_path = tmp_path / "multi.bag"
    bag_path.write_bytes(generate_multi_topic_bag())
    return bag_path


def _convert(bag_path: Path, output_path: Path, **kwargs) -> None:
    options = Bag2McapOptions(**kwargs)
    with output_path.open("wb") as f:
        convert_bag_to_mcap(bag_path, f, options)


def _read_mcap_messages(
    mcap_path: Path,
) -> list[tuple[Schema | None, Channel, Message]]:
    with mcap_path.open("rb") as f:
        return list(read_message(f))


@pytest.mark.e2e
class TestBag2McapBasic:
    def test_produces_valid_mcap(self, simple_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(simple_bag, output)
        assert output.exists()
        assert output.stat().st_size > 0

        messages = _read_mcap_messages(output)
        assert len(messages) == 5

    def test_correct_message_count(self, multi_topic_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(multi_topic_bag, output)
        messages = _read_mcap_messages(output)
        assert len(messages) == 10

    def test_returns_correct_statistics(self, simple_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        options = Bag2McapOptions()
        with output.open("wb") as f:
            stats = convert_bag_to_mcap(simple_bag, f, options)

        assert stats.topic_count == 1
        assert stats.message_count == 5
        assert stats.schema_count == 1

    def test_multi_topic_statistics(self, multi_topic_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        options = Bag2McapOptions()
        with output.open("wb") as f:
            stats = convert_bag_to_mcap(multi_topic_bag, f, options)

        assert stats.topic_count == 2
        assert stats.message_count == 10
        assert stats.schema_count == 2


@pytest.mark.e2e
class TestBag2McapSchemaAndEncoding:
    def test_ros1_profile(self, simple_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(simple_bag, output)

        messages = _read_mcap_messages(output)
        for schema, channel, _msg in messages:
            assert channel.message_encoding == "ros1"
            assert schema is not None
            assert schema.encoding == "ros1msg"

    def test_schema_contains_message_definition(self, simple_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(simple_bag, output)

        messages = _read_mcap_messages(output)
        schema, _channel, _msg = messages[0]
        assert schema is not None
        assert schema.name == "std_msgs/String"
        definition = schema.data.decode("utf-8")
        assert "string data" in definition


@pytest.mark.e2e
class TestBag2McapDataPassthrough:
    def test_message_data_preserved(self, simple_bag: Path, tmp_path: Path) -> None:
        """Verify raw message bytes match between bag and MCAP."""
        output = tmp_path / "output.mcap"
        _convert(simple_bag, output)

        with simple_bag.open("rb") as f:
            bag_messages = list(read_bag_messages(f))

        mcap_messages = _read_mcap_messages(output)

        assert len(bag_messages) == len(mcap_messages)
        for bag_msg, (_schema, _channel, mcap_msg) in zip(bag_messages, mcap_messages, strict=True):
            assert bag_msg.data == mcap_msg.data

    def test_message_content_correct(self, simple_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(simple_bag, output)

        messages = _read_mcap_messages(output)
        for i, (_schema, _channel, msg) in enumerate(messages):
            data = bytes(msg.data)
            str_len = struct.unpack("<I", data[:4])[0]
            text = data[4 : 4 + str_len].decode("utf-8")
            assert text == f"hello {i}"


@pytest.mark.e2e
class TestBag2McapTimestamps:
    def test_timestamps_preserved(self, simple_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(simple_bag, output)

        mcap_messages = _read_mcap_messages(output)
        for i, (_schema, _channel, msg) in enumerate(mcap_messages):
            expected_ns = 1000 * _NSEC_PER_SEC + i * 100_000_000
            assert msg.log_time == expected_ns
            assert msg.publish_time == expected_ns

    def test_timestamps_monotonic(self, simple_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(simple_bag, output)

        mcap_messages = _read_mcap_messages(output)
        for i in range(1, len(mcap_messages)):
            assert mcap_messages[i][2].log_time >= mcap_messages[i - 1][2].log_time


@pytest.mark.e2e
class TestBag2McapMultiTopic:
    def test_all_topics_present(self, multi_topic_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(multi_topic_bag, output)

        mcap_messages = _read_mcap_messages(output)
        topics = {channel.topic for _schema, channel, _msg in mcap_messages}
        assert topics == {"/chatter", "/counter"}

    def test_per_topic_message_counts(self, multi_topic_bag: Path, tmp_path: Path) -> None:
        output = tmp_path / "output.mcap"
        _convert(multi_topic_bag, output)

        mcap_messages = _read_mcap_messages(output)
        topic_counts: dict[str, int] = {}
        for _schema, channel, _msg in mcap_messages:
            topic_counts[channel.topic] = topic_counts.get(channel.topic, 0) + 1

        assert topic_counts["/chatter"] == 5
        assert topic_counts["/counter"] == 5


@pytest.mark.e2e
class TestBag2McapCompression:
    @pytest.mark.parametrize("compression", ["none", "lz4", "zstd"])
    def test_compression_options(self, simple_bag: Path, tmp_path: Path, compression: str) -> None:
        from small_mcap.writer import CompressionType  # noqa: PLC0415

        compression_map = {
            "none": CompressionType.NONE,
            "lz4": CompressionType.LZ4,
            "zstd": CompressionType.ZSTD,
        }
        output = tmp_path / f"output_{compression}.mcap"
        _convert(simple_bag, output, compression=compression_map[compression])

        messages = _read_mcap_messages(output)
        assert len(messages) == 5

    def test_compressed_bag_input(self, tmp_path: Path) -> None:
        """Test converting a bz2-compressed bag file."""
        bag_path = tmp_path / "compressed.bag"
        bag_path.write_bytes(generate_simple_bag(compression="bz2"))
        output = tmp_path / "output.mcap"
        _convert(bag_path, output)

        messages = _read_mcap_messages(output)
        assert len(messages) == 5


@pytest.mark.e2e
class TestBag2McapRealBags:
    """Tests against real bag files written by ROS tooling."""

    def test_demo_bag(self, tmp_path: Path) -> None:
        """Convert demo.bag (2 topics, 3 messages, uncompressed)."""
        output = tmp_path / "demo.mcap"
        _convert(_DEMO_BAG, output)

        mcap_messages = _read_mcap_messages(output)
        assert len(mcap_messages) == 3

        topics = {channel.topic for _schema, channel, _msg in mcap_messages}
        assert topics == {"/chatter", "/diagnostics"}

        for schema, channel, _msg in mcap_messages:
            assert channel.message_encoding == "ros1"
            assert schema is not None
            assert schema.encoding == "ros1msg"
            assert schema.name == "std_msgs/String"

    def test_demo_bag_data_passthrough(self, tmp_path: Path) -> None:
        """Verify raw bytes are preserved from demo.bag."""
        output = tmp_path / "demo.mcap"
        _convert(_DEMO_BAG, output)

        with _DEMO_BAG.open("rb") as f:
            bag_messages = list(read_bag_messages(f))

        mcap_messages = _read_mcap_messages(output)
        assert len(bag_messages) == len(mcap_messages)
        for bag_msg, (_schema, _channel, mcap_msg) in zip(bag_messages, mcap_messages, strict=True):
            assert bag_msg.data == mcap_msg.data

    def test_markers_bag(self, tmp_path: Path) -> None:
        """Convert markers.bag (1 topic, 10 MarkerArray messages, uncompressed)."""
        output = tmp_path / "markers.mcap"
        _convert(_MARKERS_BAG, output)

        mcap_messages = _read_mcap_messages(output)
        assert len(mcap_messages) == 10

        topics = {channel.topic for _schema, channel, _msg in mcap_messages}
        assert topics == {"/example_markers"}

        schema = mcap_messages[0][0]
        assert schema is not None
        assert schema.name == "visualization_msgs/MarkerArray"

    def test_markers_bz2_bag(self, tmp_path: Path) -> None:
        """Convert markers.bz2.bag (bz2-compressed, 10 chunks)."""
        output = tmp_path / "markers_bz2.mcap"
        _convert(_MARKERS_BZ2_BAG, output)

        mcap_messages = _read_mcap_messages(output)
        assert len(mcap_messages) == 10

        # Verify same data as uncompressed version
        output_uncompressed = tmp_path / "markers.mcap"
        _convert(_MARKERS_BAG, output_uncompressed)
        mcap_uncompressed = _read_mcap_messages(output_uncompressed)

        for compressed, uncompressed in zip(mcap_messages, mcap_uncompressed, strict=True):
            assert compressed[2].data == uncompressed[2].data
