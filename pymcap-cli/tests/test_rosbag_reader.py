"""Unit tests for the rosbag_reader package."""

from __future__ import annotations

import io
import struct

import pytest
from pymcap_cli.rosbag_reader import read_bag_info, read_bag_messages

from .fixtures.bag_generator import (
    generate_bag,
    generate_empty_bag,
    generate_multi_topic_bag,
    generate_simple_bag,
    make_string_connection,
    make_string_message,
)

_NSEC_PER_SEC = 1_000_000_000


class TestMagicValidation:
    def test_rejects_invalid_magic(self) -> None:
        with pytest.raises(ValueError, match=r"Not a ROS bag v2\.0 file"):
            read_bag_info(io.BytesIO(b"not a bag file"))

    def test_rejects_empty_file(self) -> None:
        with pytest.raises(ValueError, match=r"Not a ROS bag v2\.0 file"):
            read_bag_info(io.BytesIO(b""))

    def test_rejects_wrong_version(self) -> None:
        with pytest.raises(ValueError, match=r"Not a ROS bag v2\.0 file"):
            read_bag_info(io.BytesIO(b"#ROSBAG V1.2\n"))


class TestReadBagInfo:
    def test_simple_bag_connections(self) -> None:
        bag_data = generate_simple_bag()
        info = read_bag_info(io.BytesIO(bag_data))

        assert info.conn_count == 1
        assert info.chunk_count == 1
        assert len(info.connections) == 1

        conn = info.connections[0]
        assert conn.topic == "/chatter"
        assert conn.msg_type == "std_msgs/String"
        assert conn.md5sum == "992ce8a1687cec8c8bd883ec73ca41d1"
        assert conn.message_definition == "string data\n"

    def test_simple_bag_message_count(self) -> None:
        bag_data = generate_simple_bag()
        info = read_bag_info(io.BytesIO(bag_data))
        assert info.message_count == 5

    def test_simple_bag_time_range(self) -> None:
        bag_data = generate_simple_bag()
        info = read_bag_info(io.BytesIO(bag_data))

        assert info.start_time_ns == 1000 * _NSEC_PER_SEC + 0
        assert info.end_time_ns == 1000 * _NSEC_PER_SEC + 4 * 100_000_000

    def test_multi_topic_connections(self) -> None:
        bag_data = generate_multi_topic_bag()
        info = read_bag_info(io.BytesIO(bag_data))

        assert info.conn_count == 2
        assert len(info.connections) == 2

        topics = {c.topic for c in info.connections.values()}
        assert topics == {"/chatter", "/counter"}

        types = {c.msg_type for c in info.connections.values()}
        assert types == {"std_msgs/String", "std_msgs/Int32"}

    def test_multi_topic_message_count(self) -> None:
        bag_data = generate_multi_topic_bag()
        info = read_bag_info(io.BytesIO(bag_data))
        assert info.message_count == 10  # 5 string + 5 int

    def test_empty_bag(self) -> None:
        bag_data = generate_empty_bag()
        info = read_bag_info(io.BytesIO(bag_data))

        assert info.conn_count == 0
        assert info.chunk_count == 1
        assert info.message_count == 0
        assert len(info.connections) == 0

    def test_connection_with_callerid(self) -> None:
        conn = make_string_connection()
        conn.callerid = "/my_node"
        conn.latching = True
        bag_data = generate_bag([conn], [])
        info = read_bag_info(io.BytesIO(bag_data))

        parsed = info.connections[0]
        assert parsed.callerid == "/my_node"
        assert parsed.latching is True


class TestReadBagMessages:
    def test_simple_bag_message_count(self) -> None:
        bag_data = generate_simple_bag()
        messages = list(read_bag_messages(io.BytesIO(bag_data)))
        assert len(messages) == 5

    def test_simple_bag_message_data(self) -> None:
        bag_data = generate_simple_bag()
        messages = list(read_bag_messages(io.BytesIO(bag_data)))

        for i, msg in enumerate(messages):
            assert msg.conn_id == 0
            # Decode ROS1 string: [4B len][string bytes]
            str_len = struct.unpack("<I", msg.data[:4])[0]
            text = msg.data[4 : 4 + str_len].decode("utf-8")
            assert text == f"hello {i}"

    def test_simple_bag_timestamps(self) -> None:
        bag_data = generate_simple_bag()
        messages = list(read_bag_messages(io.BytesIO(bag_data)))

        for i, msg in enumerate(messages):
            expected_ns = 1000 * _NSEC_PER_SEC + i * 100_000_000
            assert msg.time_ns == expected_ns

    def test_timestamps_monotonic(self) -> None:
        bag_data = generate_simple_bag()
        messages = list(read_bag_messages(io.BytesIO(bag_data)))

        for i in range(1, len(messages)):
            assert messages[i].time_ns >= messages[i - 1].time_ns

    def test_multi_topic_messages(self) -> None:
        bag_data = generate_multi_topic_bag()
        messages = list(read_bag_messages(io.BytesIO(bag_data)))

        assert len(messages) == 10
        string_msgs = [m for m in messages if m.conn_id == 0]
        int_msgs = [m for m in messages if m.conn_id == 1]
        assert len(string_msgs) == 5
        assert len(int_msgs) == 5

    def test_multi_topic_int_data(self) -> None:
        bag_data = generate_multi_topic_bag()
        messages = list(read_bag_messages(io.BytesIO(bag_data)))

        int_msgs = [m for m in messages if m.conn_id == 1]
        for i, msg in enumerate(int_msgs):
            value = struct.unpack("<i", msg.data)[0]
            assert value == i * 10

    def test_empty_bag_no_messages(self) -> None:
        bag_data = generate_empty_bag()
        messages = list(read_bag_messages(io.BytesIO(bag_data)))
        assert len(messages) == 0

    def test_bz2_compressed(self) -> None:
        bag_data = generate_simple_bag(compression="bz2")
        info = read_bag_info(io.BytesIO(bag_data))
        assert info.message_count == 5

        messages = list(read_bag_messages(io.BytesIO(bag_data)))
        assert len(messages) == 5

        # Verify data integrity
        for i, msg in enumerate(messages):
            str_len = struct.unpack("<I", msg.data[:4])[0]
            text = msg.data[4 : 4 + str_len].decode("utf-8")
            assert text == f"hello {i}"

    def test_lz4_compressed(self) -> None:
        bag_data = generate_simple_bag(compression="lz4")
        info = read_bag_info(io.BytesIO(bag_data))
        assert info.message_count == 5

        messages = list(read_bag_messages(io.BytesIO(bag_data)))
        assert len(messages) == 5

    def test_data_passthrough(self) -> None:
        """Verify raw message bytes are preserved exactly."""
        conn = make_string_connection()
        original_data = b"\x05\x00\x00\x00world"  # ROS1 string "world"
        msg = make_string_message(0, 1000, 0, "world")
        bag_data = generate_bag([conn], [msg])

        messages = list(read_bag_messages(io.BytesIO(bag_data)))
        assert len(messages) == 1
        assert messages[0].data == original_data
