"""
Test that the fast implementation accepts the same input types as the reference implementation.

The reference implementation (data/mcap/python/mcap-ros2-support/mcap_ros2/_dynamic.py:427-436)
accepts these types for array fields:
- list
- tuple
- bytes (for uint8/byte/char arrays)
- array.array (from the array module)

This test suite ensures mcap-ros2-support-fast has identical input type support.
"""

from array import array
from io import BytesIO

from mcap_ros2_support_fast import ROS2EncoderFactory
from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap.reader import read_message_decoded
from small_mcap.writer import McapWriter


def read_ros2_messages(stream: BytesIO):
    return read_message_decoded(stream, decoder_factories=[DecoderFactory()])


def test_primitive_arrays_with_tuples() -> None:
    """Test that primitive arrays accept tuple inputs (int32[], float64[], etc.)."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    schema_name = "test_msgs/PrimitiveArrays"
    schema_data = b"int32[] ints\nfloat64[] floats\nuint16[] uints"

    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    ros_writer.add_channel(channel_id, "/test", "cdr", schema_id)

    # Use tuples instead of lists
    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data={
            "ints": (1, 2, 3, -4, 5),  # tuple of ints
            "floats": (1.5, 2.5, 3.5, -4.5),  # tuple of floats
            "uints": (100, 200, 300),  # tuple of unsigned ints
        },
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = list(read_ros2_messages(output))
    assert len(messages) == 1

    msg = messages[0].decoded_message
    assert list(msg.ints) == [1, 2, 3, -4, 5]
    assert list(msg.floats) == [1.5, 2.5, 3.5, -4.5]
    assert list(msg.uints) == [100, 200, 300]


def test_primitive_arrays_with_array_module() -> None:
    """Test that primitive arrays accept array.array inputs for various types."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    schema_name = "test_msgs/TypedArrays"
    schema_data = b"int16[] int16s\nint32[] int32s\nfloat32[] float32s\nfloat64[] float64s"

    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    ros_writer.add_channel(channel_id, "/test", "cdr", schema_id)

    # Use array.array with appropriate type codes
    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data={
            "int16s": array("h", [-100, -200, 300, 400]),  # signed short
            "int32s": array("i", [1000, -2000, 3000, -4000]),  # signed int
            "float32s": array("f", [1.5, 2.5, -3.5, 4.5]),  # float
            "float64s": array("d", [10.5, -20.5, 30.5, -40.5]),  # double
        },
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = list(read_ros2_messages(output))
    assert len(messages) == 1

    msg = messages[0].decoded_message
    assert list(msg.int16s) == [-100, -200, 300, 400]
    assert list(msg.int32s) == [1000, -2000, 3000, -4000]
    # Float comparison with tolerance
    assert len(msg.float32s) == 4
    assert abs(msg.float32s[0] - 1.5) < 0.001
    assert abs(msg.float32s[1] - 2.5) < 0.001
    assert abs(msg.float32s[2] - (-3.5)) < 0.001
    assert abs(msg.float32s[3] - 4.5) < 0.001
    assert list(msg.float64s) == [10.5, -20.5, 30.5, -40.5]


def test_byte_arrays_with_bytes_and_bytearray() -> None:
    """Test that uint8[], byte[], and char[] arrays accept bytes and bytearray objects."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    schema_name = "test_msgs/ByteTypes"
    schema_data = b"uint8[] uint8s\nbyte[] bytes\nchar[] chars"

    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    ros_writer.add_channel(channel_id, "/test", "cdr", schema_id)

    # Test with bytes object
    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data={
            "uint8s": b"\x01\x02\x03\x04\x05",  # bytes object
            "bytes": bytes([10, 20, 30, 40]),  # bytes constructor
            "chars": b"hello",  # bytes literal
        },
        publish_time=0,
        sequence=0,
    )

    # Test with bytearray object
    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=1,
        data={
            "uint8s": bytearray([100, 101, 102]),  # bytearray
            "bytes": bytearray(b"\xff\xfe\xfd"),  # bytearray from bytes
            "chars": bytearray([65, 66, 67]),  # ABC in ASCII
        },
        publish_time=1,
        sequence=1,
    )

    ros_writer.finish()

    output.seek(0)
    messages = list(read_ros2_messages(output))
    assert len(messages) == 2

    # Check first message (bytes object)
    msg0 = messages[0].decoded_message
    assert list(msg0.uint8s) == [1, 2, 3, 4, 5]
    assert list(msg0.bytes) == [10, 20, 30, 40]
    assert list(msg0.chars) == list(b"hello")

    # Check second message (bytearray object)
    msg1 = messages[1].decoded_message
    assert list(msg1.uint8s) == [100, 101, 102]
    assert list(msg1.bytes) == [255, 254, 253]
    assert list(msg1.chars) == [65, 66, 67]


def test_complex_message_arrays_with_tuples() -> None:
    """Test that complex message arrays accept tuple inputs."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    schema_name = "test_msgs/NestedArrays"
    schema_data = b"""Point[] points
================================================================================
MSG: test_msgs/Point
float32 x
float32 y
float32 z"""

    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    ros_writer.add_channel(channel_id, "/test", "cdr", schema_id)

    # Use a tuple of dicts for the complex message array
    points_tuple = (
        {"x": 1.0, "y": 2.0, "z": 3.0},
        {"x": 4.0, "y": 5.0, "z": 6.0},
        {"x": 7.0, "y": 8.0, "z": 9.0},
    )

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data={"points": points_tuple},
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = list(read_ros2_messages(output))
    assert len(messages) == 1

    msg = messages[0].decoded_message
    assert len(msg.points) == 3
    assert msg.points[0].x == 1.0
    assert msg.points[0].y == 2.0
    assert msg.points[0].z == 3.0
    assert msg.points[1].x == 4.0
    assert msg.points[1].y == 5.0
    assert msg.points[1].z == 6.0
    assert msg.points[2].x == 7.0
    assert msg.points[2].y == 8.0
    assert msg.points[2].z == 9.0


def test_mixed_input_types() -> None:
    """Test that a single message can use different input types for different fields."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    schema_name = "test_msgs/MixedTypes"
    schema_data = b"uint8[] bytes\nint32[] ints\nfloat64[] floats\nstring[] strings"

    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    ros_writer.add_channel(channel_id, "/test", "cdr", schema_id)

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data={
            "bytes": b"\x01\x02\x03",  # bytes
            "ints": (10, 20, 30),  # tuple
            "floats": array("d", [1.1, 2.2, 3.3]),  # array.array
            "strings": ["hello", "world"],  # list
        },
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = list(read_ros2_messages(output))
    assert len(messages) == 1

    msg = messages[0].decoded_message
    assert list(msg.bytes) == [1, 2, 3]
    assert list(msg.ints) == [10, 20, 30]
    assert abs(msg.floats[0] - 1.1) < 0.001
    assert abs(msg.floats[1] - 2.2) < 0.001
    assert abs(msg.floats[2] - 3.3) < 0.001
    assert list(msg.strings) == ["hello", "world"]


def test_fixed_arrays_with_various_types() -> None:
    """Test that fixed-size arrays work with tuple/array.array/bytes inputs."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    schema_name = "test_msgs/FixedArrays"
    schema_data = b"uint8[5] fixed_bytes\nint32[3] fixed_ints\nfloat32[4] fixed_floats"

    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    ros_writer.add_channel(channel_id, "/test", "cdr", schema_id)

    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data={
            "fixed_bytes": b"\x0a\x0b\x0c\x0d\x0e",  # bytes (exactly 5)
            "fixed_ints": (100, 200, 300),  # tuple (exactly 3)
            "fixed_floats": array("f", [1.0, 2.0, 3.0, 4.0]),  # array.array (exactly 4)
        },
        publish_time=0,
        sequence=0,
    )
    ros_writer.finish()

    output.seek(0)
    messages = list(read_ros2_messages(output))
    assert len(messages) == 1

    msg = messages[0].decoded_message
    assert list(msg.fixed_bytes) == [10, 11, 12, 13, 14]
    assert list(msg.fixed_ints) == [100, 200, 300]
    assert len(msg.fixed_floats) == 4


def test_bounded_arrays_with_various_types() -> None:
    """Test that bounded arrays (<=N) work with different input types and truncate correctly."""
    output = BytesIO()
    ros_writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    ros_writer.start()

    schema_name = "test_msgs/BoundedArrays"
    schema_data = b"uint8[<=5] bounded_bytes\nint32[<=3] bounded_ints"

    schema_id = 1
    channel_id = 1
    ros_writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    ros_writer.add_channel(channel_id, "/test", "cdr", schema_id)

    # Test with inputs that exceed bounds (should truncate)
    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=0,
        data={
            "bounded_bytes": b"\x01\x02\x03\x04\x05\x06\x07",  # 7 bytes, should truncate to 5
            "bounded_ints": (10, 20, 30, 40, 50),  # 5 ints, should truncate to 3
        },
        publish_time=0,
        sequence=0,
    )

    # Test with inputs within bounds
    ros_writer.add_message_encode(
        channel_id=channel_id,
        log_time=1,
        data={
            "bounded_bytes": array("B", [100, 101]),  # 2 bytes, within bounds
            "bounded_ints": [1, 2],  # 2 ints, within bounds
        },
        publish_time=1,
        sequence=1,
    )

    ros_writer.finish()

    output.seek(0)
    messages = list(read_ros2_messages(output))
    assert len(messages) == 2

    # First message (truncated)
    msg0 = messages[0].decoded_message
    assert list(msg0.bounded_bytes) == [1, 2, 3, 4, 5]  # Truncated to 5
    assert list(msg0.bounded_ints) == [10, 20, 30]  # Truncated to 3

    # Second message (within bounds)
    msg1 = messages[1].decoded_message
    assert list(msg1.bounded_bytes) == [100, 101]
    assert list(msg1.bounded_ints) == [1, 2]
