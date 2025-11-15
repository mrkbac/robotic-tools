"""Test PointCloud2 serialization to verify padding handling in primitive groups."""

from io import BytesIO

import numpy as np
import pytest
from mcap_ros2_support_fast import ROS2EncoderFactory
from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap.reader import read_message_decoded
from small_mcap.writer import McapWriter


def read_ros2_messages(stream: BytesIO):
    """Read ROS2 messages from MCAP stream."""
    return list(read_message_decoded(stream, decoder_factories=[DecoderFactory()]))


# PointCloud2 schema with nested PointField type
POINTCLOUD2_SCHEMA = """std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: sensor_msgs/PointField
string name
uint32 offset
uint8 datatype
uint32 count"""


def create_pointcloud2_xyz(num_points: int) -> dict:
    """Create a PointCloud2 message with XYZ points.

    This matches the structure created by pointcloud_voxel.py transformer.

    Args:
        num_points: Number of points in the cloud

    Returns:
        PointCloud2 message dict
    """
    # Create XYZ points
    rng = np.random.default_rng()
    points = rng.standard_normal((num_points, 3)).astype(np.float32)

    # Create PointField definitions for XYZ
    fields = [
        {
            "name": "x",
            "offset": 0,
            "datatype": 7,  # FLOAT32
            "count": 1,
        },
        {
            "name": "y",
            "offset": 4,
            "datatype": 7,  # FLOAT32
            "count": 1,
        },
        {
            "name": "z",
            "offset": 8,
            "datatype": 7,  # FLOAT32
            "count": 1,
        },
    ]

    # Create structured array
    dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
    structured = np.zeros(num_points, dtype=dtype)
    if num_points > 0:
        structured["x"] = points[:, 0]
        structured["y"] = points[:, 1]
        structured["z"] = points[:, 2]

    data_bytes = structured.tobytes()
    point_step = dtype.itemsize  # 12 bytes (3 * 4)
    row_step = point_step * num_points

    return {
        "header": {
            "stamp": {"sec": 1762166808, "nanosec": 100356658},
            "frame_id": "test_frame",
        },
        "height": 1,
        "width": num_points,
        "fields": fields,
        "is_bigendian": False,
        "point_step": point_step,
        "row_step": row_step,
        "data": list(data_bytes),
        "is_dense": True,
    }, points


def test_pointcloud2_basic_roundtrip():
    """Test basic PointCloud2 serialization and deserialization."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    # Register schema and channel
    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2_SCHEMA.encode())
    ros_writer.add_channel(channel_id, "/points", "cdr", schema_id)

    # Create a simple point cloud with 100 points
    msg, original_points = create_pointcloud2_xyz(100)

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data=msg,
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    # Read back and verify
    output.seek(0)
    messages = read_ros2_messages(output)
    assert len(messages) == 1

    decoded = messages[0].decoded_message

    # Verify header
    assert decoded.header.stamp.sec == 1762166808
    assert decoded.header.stamp.nanosec == 100356658
    assert decoded.header.frame_id == "test_frame"

    # Verify dimensions
    assert decoded.height == 1
    assert decoded.width == 100

    # Verify fields - this tests the PointField array with padding
    assert len(decoded.fields) == 3
    assert decoded.fields[0].name == "x"
    assert decoded.fields[0].offset == 0
    assert decoded.fields[0].datatype == 7
    assert decoded.fields[0].count == 1

    assert decoded.fields[1].name == "y"
    assert decoded.fields[1].offset == 4
    assert decoded.fields[1].datatype == 7
    assert decoded.fields[1].count == 1

    assert decoded.fields[2].name == "z"
    assert decoded.fields[2].offset == 8
    assert decoded.fields[2].datatype == 7
    assert decoded.fields[2].count == 1

    # CRITICAL: Verify the fields after the complex array
    # This is where the bug manifests - these fields get corrupted
    assert decoded.is_bigendian is False, "is_bigendian should be False"
    assert decoded.point_step == 12, f"point_step should be 12, got {decoded.point_step}"
    assert decoded.row_step == 1200, f"row_step should be 1200, got {decoded.row_step}"

    # Verify data is not empty
    assert len(decoded.data) == 1200, f"data should be 1200 bytes, got {len(decoded.data)}"

    # Verify the actual point data
    decoded_data = bytes(decoded.data)
    dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
    decoded_points = np.frombuffer(decoded_data, dtype=dtype)

    assert decoded_points.shape[0] == 100
    np.testing.assert_array_almost_equal(decoded_points["x"], original_points[:, 0])
    np.testing.assert_array_almost_equal(decoded_points["y"], original_points[:, 1])
    np.testing.assert_array_almost_equal(decoded_points["z"], original_points[:, 2])

    # Verify is_dense
    assert decoded.is_dense is True


def test_pointcloud2_empty():
    """Test PointCloud2 with no points."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    # Register schema and channel
    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2_SCHEMA.encode())
    ros_writer.add_channel(channel_id, "/points", "cdr", schema_id)

    msg, _ = create_pointcloud2_xyz(0)

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data=msg,
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = read_ros2_messages(output)
    assert len(messages) == 1

    decoded = messages[0].decoded_message
    assert decoded.width == 0
    assert decoded.point_step == 12
    assert decoded.row_step == 0
    assert len(decoded.data) == 0
    assert decoded.is_bigendian is False


def test_pointcloud2_large():
    """Test PointCloud2 with many points (like from voxel downsampling)."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    # Register schema and channel
    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2_SCHEMA.encode())
    ros_writer.add_channel(channel_id, "/points", "cdr", schema_id)

    # Use 4071 points like in the bug report
    msg, _original_points = create_pointcloud2_xyz(4071)

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data=msg,
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = read_ros2_messages(output)
    assert len(messages) == 1

    decoded = messages[0].decoded_message
    assert decoded.width == 4071
    assert decoded.point_step == 12
    assert decoded.row_step == 48852
    assert len(decoded.data) == 48852
    assert decoded.is_bigendian is False
    assert decoded.is_dense is True


def test_pointfield_padding():
    """Test that PointField struct correctly handles padding between uint8 and uint32.

    The PointField struct has:
    - string name
    - uint32 offset
    - uint8 datatype
    - uint32 count

    Between datatype (uint8) and count (uint32), there must be 3 padding bytes
    for proper CDR alignment.
    """
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    # Create message with specific field values to test padding
    msg = {
        "header": {
            "stamp": {"sec": 0, "nanosec": 0},
            "frame_id": "",
        },
        "height": 1,
        "width": 0,
        "fields": [
            {
                "name": "test",
                "offset": 123,  # Specific value to detect misalignment
                "datatype": 7,  # uint8
                "count": 456,  # Should be aligned after 3 bytes of padding
            }
        ],
        "is_bigendian": False,
        "point_step": 0,
        "row_step": 0,
        "data": [],
        "is_dense": False,
    }

    # Register schema and channel
    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2_SCHEMA.encode())
    ros_writer.add_channel(channel_id, "/points", "cdr", schema_id)

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data=msg,
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = read_ros2_messages(output)
    decoded = messages[0].decoded_message

    # Verify the PointField was correctly encoded and decoded
    assert len(decoded.fields) == 1
    assert decoded.fields[0].name == "test"
    assert decoded.fields[0].offset == 123, f"offset should be 123, got {decoded.fields[0].offset}"
    assert decoded.fields[0].datatype == 7, (
        f"datatype should be 7, got {decoded.fields[0].datatype}"
    )
    assert decoded.fields[0].count == 456, f"count should be 456, got {decoded.fields[0].count}"


@pytest.mark.parametrize("num_points", [0, 1, 10, 100, 1000, 4071])
def test_pointcloud2_various_sizes(num_points):
    """Test PointCloud2 with various point counts."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    # Register schema and channel
    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2_SCHEMA.encode())
    ros_writer.add_channel(channel_id, "/points", "cdr", schema_id)

    msg, _ = create_pointcloud2_xyz(num_points)

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data=msg,
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = read_ros2_messages(output)
    decoded = messages[0].decoded_message

    assert decoded.width == num_points
    assert decoded.point_step == 12
    assert decoded.row_step == 12 * num_points
    assert len(decoded.data) == 12 * num_points
    assert decoded.is_bigendian is False
