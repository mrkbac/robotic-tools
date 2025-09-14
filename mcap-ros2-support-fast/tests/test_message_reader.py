"""Tests translated from TypeScript MessageReader.test.ts"""

import struct
from collections.abc import Iterable

import pytest
from mcap_ros2_support_fast._dynamic_codegen import (
    create_decoder as create_codegen_decoder,
)
from mcap_ros2_support_fast._planner import generate_dynamic


def get_decoder_parser():
    """Helper function to get the correct parser based on decoder type."""
    return create_dynamic_decoder if decoder_type == "dynamic" else create_codegen_decoder


def serialize_string(s: str) -> bytes:
    """Python equivalent of TypeScript serializeString function."""
    data = s.encode("utf-8")
    length = struct.pack("<I", len(data) + 1)  # +1 for null terminator
    return length + data + b"\x00"


def float32_buffer(floats: list[float]) -> bytes:
    """Python equivalent of TypeScript float32Buffer function."""
    return struct.pack(f"<{len(floats)}f", *floats)


@pytest.mark.parametrize(
    ("msg_def", "data", "expected"),
    [
        # int8 tests
        ("int8 sample # lowest", [0x80], {"sample": -128}),
        ("int8 sample # highest", [0x7F], {"sample": 127}),
        # uint8 tests
        ("uint8 sample # lowest", [0x00], {"sample": 0}),
        ("uint8 sample # highest", [0xFF], {"sample": 255}),
        # int16 tests
        ("int16 sample # lowest", [0x00, 0x80], {"sample": -32768}),
        ("int16 sample # highest", [0xFF, 0x7F], {"sample": 32767}),
        # uint16 tests
        ("uint16 sample # lowest", [0x00, 0x00], {"sample": 0}),
        ("uint16 sample # highest", [0xFF, 0xFF], {"sample": 65535}),
        # int32 tests
        ("int32 sample # lowest", [0x00, 0x00, 0x00, 0x80], {"sample": -2147483648}),
        ("int32 sample # highest", [0xFF, 0xFF, 0xFF, 0x7F], {"sample": 2147483647}),
        # uint32 tests
        ("uint32 sample # lowest", [0x00, 0x00, 0x00, 0x00], {"sample": 0}),
        ("uint32 sample # highest", [0xFF, 0xFF, 0xFF, 0xFF], {"sample": 4294967295}),
        # int64 tests
        (
            "int64 sample # lowest",
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80],
            {"sample": -9223372036854775808},
        ),
        (
            "int64 sample # highest",
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F],
            {"sample": 9223372036854775807},
        ),
        # uint64 tests
        (
            "uint64 sample # lowest",
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            {"sample": 0},
        ),
        (
            "uint64 sample # highest",
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            {"sample": 18446744073709551615},
        ),
        # float32 tests
        ("float32 sample", list(float32_buffer([5.5])), {"sample": 5.5}),
        # float64 tests
        (
            "float64 sample",
            list(struct.pack("<d", 0.123456789121212121212)),
            {"sample": 0.123456789121212121212},
        ),
        # int32[] array tests
        (
            "int32[] arr",
            [
                *[0x02, 0x00, 0x00, 0x00],  # length
                *list(struct.pack("<ii", 3, 7)),
            ],
            {"arr": [3, 7]},  # Python uses regular lists instead of typed arrays
        ),
        # unaligned access test
        (
            "uint8 blank\nint32[] arr",
            [
                0x00,
                *[0x00, 0x00, 0x00],  # alignment
                *[0x02, 0x00, 0x00, 0x00],  # length
                *list(struct.pack("<ii", 3, 7)),
            ],
            {"blank": 0, "arr": [3, 7]},
        ),
        # float32[2] fixed array tests
        ("float32[2] arr", list(float32_buffer([5.5, 6.5])), {"arr": [5.5, 6.5]}),
        # unaligned float32[2] array test
        (
            "uint8 blank\nfloat32[2] arr",
            [
                0x00,
                *[0x00, 0x00, 0x00],  # alignment
                *list(float32_buffer([5.5, 6.5])),
            ],
            {"blank": 0, "arr": [5.5, 6.5]},
        ),
        # float32[] dynamic array tests
        (
            "float32[] arr",
            [
                *[0x02, 0x00, 0x00, 0x00],  # length
                *list(float32_buffer([5.5, 6.5])),
            ],
            {"arr": [5.5, 6.5]},
        ),
        # unaligned float32[] array test
        (
            "uint8 blank\nfloat32[] arr",
            [
                0x00,
                *[0x00, 0x00, 0x00],  # alignment
                *[0x02, 0x00, 0x00, 0x00],
                *list(float32_buffer([5.5, 6.5])),
            ],
            {"blank": 0, "arr": [5.5, 6.5]},
        ),
        # multiple dynamic arrays test
        (
            "float32[] first\nfloat32[] second",
            [
                *[0x02, 0x00, 0x00, 0x00],  # length
                *list(float32_buffer([5.5, 6.5])),
                *[0x02, 0x00, 0x00, 0x00],  # length
                *list(float32_buffer([5.5, 6.5])),
            ],
            {
                "first": [5.5, 6.5],
                "second": [5.5, 6.5],
            },
        ),
        # string tests
        ("string sample # empty string", list(serialize_string("")), {"sample": ""}),
        (
            "string sample # some string",
            list(serialize_string("some string")),
            {"sample": "some string"},
        ),
        # int8[4] fixed array tests
        ("int8[4] first", [0x00, 0xFF, 0x80, 0x7F], {"first": [0, -1, -128, 127]}),
        # int8[] dynamic array tests
        (
            "int8[] first",
            [
                *[0x04, 0x00, 0x00, 0x00],  # length
                0x00,
                0xFF,
                0x80,
                0x7F,
            ],
            {"first": [0, -1, -128, 127]},
        ),
        # uint8[4] fixed array tests
        ("uint8[4] first", [0x00, 0xFF, 0x80, 0x7F], {"first": b"\x00\xff\x80\x7f"}),
        # string[2] fixed array tests
        (
            "string[2] first",
            [*list(serialize_string("one")), *list(serialize_string("longer string"))],
            {"first": ["one", "longer string"]},
        ),
        # string[] dynamic array tests
        (
            "string[] first",
            [
                *[0x02, 0x00, 0x00, 0x00],  # length
                *list(serialize_string("one")),
                *list(serialize_string("longer string")),
            ],
            {"first": ["one", "longer string"]},
        ),
        # multiple fields test
        ("int8 first\nint8 second", [0x80, 0x7F], {"first": -128, "second": 127}),
        # string followed by int8 test
        (
            "string first\nint8 second",
            [*list(serialize_string("some string")), 0x80],
            {"first": "some string", "second": -128},
        ),
        # custom type tests
        (
            "CustomType custom\n============\nMSG: custom_type/CustomType\nuint8 first",
            [0x02],
            {"custom": {"first": 0x02}},
        ),
        # custom type[3] fixed array tests
        (
            "CustomType[3] custom\n============\nMSG: custom_type/CustomType\nuint8 first",
            [0x02, 0x03, 0x04],
            {"custom": [{"first": 0x02}, {"first": 0x03}, {"first": 0x04}]},
        ),
        # custom type[] dynamic array tests
        (
            "CustomType[] custom\n============\nMSG: custom_type/CustomType\nuint8 first",
            [
                *[0x03, 0x00, 0x00, 0x00],  # length
                0x02,
                0x03,
                0x04,
            ],
            {"custom": [{"first": 0x02}, {"first": 0x03}, {"first": 0x04}]},
        ),
        # constants test (should be ignored)
        (
            "int8 STATUS_ONE = 1\nint8 STATUS_TWO = 2\nint8 status",
            [0x02],
            {"status": 2},
        ),
        # nested custom types test
        (
            "CustomType[] custom\n"
            "============\n"
            "MSG: custom_type/CustomType\n"
            "MoreCustom another\n"
            "============\n"
            "MSG: custom_type/MoreCustom\n"
            "uint8 field",
            [
                *[0x03, 0x00, 0x00, 0x00],  # length
                0x02,
                0x03,
                0x04,
            ],
            {
                "custom": [
                    {"another": {"field": 0x02}},
                    {"another": {"field": 0x03}},
                    {"another": {"field": 0x04}},
                ]
            },
        ),
    ],
)
def test_deserialize_basic_types(msg_def: str, data: Iterable[int], expected: dict) -> None:
    """Test deserialization of basic types."""
    buffer = bytes([0, 1, 0, 0, *list(data)])

    # Choose decoder based on parameter
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)

    # Convert SimpleNamespace to dict for comparison
    result_dict = _convert_to_dict(result)
    assert result_dict == expected


@pytest.mark.parametrize(
    ("msg_def", "data", "expected"),
    [
        (
            "builtin_interfaces/Time stamp",
            [0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00],
            {"stamp": {"sec": 0, "nanosec": 1}},
        ),
        (
            "builtin_interfaces/Duration stamp",
            [0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00],
            {"stamp": {"sec": 0, "nanosec": 1}},
        ),
        (
            "builtin_interfaces/Time[1] arr",
            [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00],
            {"arr": [{"sec": 1, "nanosec": 2}]},
        ),
        (
            "builtin_interfaces/Duration[1] arr",
            [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00],
            {"arr": [{"sec": 1, "nanosec": 2}]},
        ),
        (
            "builtin_interfaces/Time[] arr",
            [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00],
            {"arr": [{"sec": 2, "nanosec": 3}]},
        ),
        (
            "builtin_interfaces/Duration[] arr",
            [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00],
            {"arr": [{"sec": 2, "nanosec": 3}]},
        ),
        # unaligned access
        (
            "uint8 blank\nbuiltin_interfaces/Time[] arr",
            [
                0x00,
                *[0x00, 0x00, 0x00],  # alignment
                *[0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00],
            ],
            {"blank": 0, "arr": [{"sec": 2, "nanosec": 3}]},
        ),
    ],
)
def test_deserialize_time_types(msg_def: str, data: Iterable[int], expected: dict) -> None:
    """Test deserialization of ROS2 time and duration types (nanosec format)."""
    buffer = bytes([0, 1, 0, 0, *list(data)])

    # Test ROS2 format (nanosec)
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)
    result_dict = _convert_to_dict(result)
    assert result_dict == expected


def test_deserialize_ros2_log_message() -> None:
    """Test deserialization of a ROS 2 log message."""
    buffer = bytes.fromhex(
        "00010000fb65865e80faae0614000000120000006d696e696d616c5f7075626c69736865720000001e0000005075626c697368696e673a202748656c6c6f2c20776f726c64212030270000004c0000002f6f70742f726f73325f77732f656c6f7175656e742f7372632f726f73322f6578616d706c65732f72636c6370702f6d696e696d616c5f7075626c69736865722f6c616d6264612e637070000b0000006f70657261746f722829007326000000"
    )
    msg_def = """
byte DEBUG=10
byte INFO=20
byte WARN=30
byte ERROR=40
byte FATAL=50
##
## Fields
##
builtin_interfaces/Time stamp
uint8 level
string name # name of the node
string msg # message
string file # file the message came from
string function # function the message came from
uint32 line # line the message came from
"""
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)
    result_dict = _convert_to_dict(result)

    expected = {
        "stamp": {"sec": 1585866235, "nanosec": 112130688},
        "level": 20,
        "name": "minimal_publisher",
        "msg": "Publishing: 'Hello, world! 0'",
        "file": "/opt/ros2_ws/eloquent/src/ros2/examples/rclcpp/minimal_publisher/lambda.cpp",
        "function": "operator()",
        "line": 38,
    }
    assert result_dict == expected


def test_deserialize_ros2_tf_message() -> None:
    """Test deserialization of a ROS 2 tf2_msgs/TFMessage."""
    buffer = bytes.fromhex(
        "0001000001000000286fae6169ddd73108000000747572746c6531000e000000747572746c65315f616865616400000000000000000000000000f03f00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000f03f"
    )
    msg_def = """
geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id # the frame id of the child frame
geometry_msgs/Transform transform
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
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
float64 w
"""
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)
    result_dict = _convert_to_dict(result)

    expected = {
        "transforms": [
            {
                "header": {
                    "stamp": {"sec": 1638821672, "nanosec": 836230505},
                    "frame_id": "turtle1",
                },
                "child_frame_id": "turtle1_ahead",
                "transform": {
                    "translation": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        ],
    }
    assert result_dict == expected


def test_deserialize_empty_ros2_message() -> None:
    """Test deserialization of an empty ROS 2 message (e.g. std_msgs/msg/Empty)."""
    buffer = bytes.fromhex("0001000000")
    msg_def = ""
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)
    result_dict = _convert_to_dict(result)

    assert result_dict == {}


def test_deserialize_empty_field_followed_by_uint8() -> None:
    """Test deserialization of a custom msg with a std_msgs/msg/Empty field followed by uint8."""
    # Note: ROS/FastDDS seems to add 2 extra padding bytes at the end
    buffer = bytes.fromhex("00010000007b0000")
    msg_def = """
std_msgs/Empty empty
uint8 uint_8_field
================================================================================
MSG: std_msgs/Empty
# This message has no fields
"""
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)
    result_dict = _convert_to_dict(result)

    assert result_dict == {"empty": {}, "uint_8_field": 123}


def test_deserialize_empty_field_followed_by_int32() -> None:
    """Test deserialization of a custom msg with a std_msgs/msg/Empty field followed by int32."""
    buffer = bytes.fromhex("00010000000000007b000001")
    msg_def = """
std_msgs/Empty empty
int32 int_32_field
================================================================================
MSG: std_msgs/Empty
# This message has no fields
"""
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)
    result_dict = _convert_to_dict(result)

    assert result_dict == {"empty": {}, "int_32_field": 16777339}


def test_deserialize_empty_message_with_constants() -> None:
    """Deserialization of a custom msg with an empty message (with constants) followed by int32."""
    buffer = bytes.fromhex("00010000000000007b000001")
    msg_def = """
custom_msgs/Nothing empty
int32 int_32_field
================================================================================
MSG: custom_msgs/Nothing
int32 EXAMPLE=123
"""
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)
    result = decoder(buffer)
    result_dict = _convert_to_dict(result)

    assert result_dict == {"empty": {}, "int_32_field": 16777339}


@pytest.mark.parametrize(
    "msg_def",
    [
        "wstring field",
        "wstring[] field",
    ],
)
def test_wstring_throws_exception(msg_def: str) -> None:
    """Test that wstring fields throw an exception."""
    buffer = bytes.fromhex("00010000000000007b000000")
    decoder = generate_dynamic("custom_type/TestMsg", msg_def, parser=create_codegen_decoder)

    with pytest.raises(NotImplementedError, match="wstring.*not.*implemented"):
        decoder(buffer)


def _convert_to_dict(obj) -> dict:
    """Convert SimpleNamespace objects to dictionaries recursively."""
    # Handle bytes objects directly (they should not be converted)
    if isinstance(obj, bytes):
        return obj

    result = {}
    if hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            value = getattr(obj, slot)
            result[slot] = _convert_to_dict(value)
    elif isinstance(obj, (list, tuple)):
        result = [_convert_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        result = {key: _convert_to_dict(value) for key, value in obj.items()}
    else:
        return obj
    return result
