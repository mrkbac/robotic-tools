"""Tests for the get_fields_and_field_types() method on generated message classes."""

from mcap_ros2_support_fast._planner import generate_plans


def test_primitive_types():
    """Test get_fields_and_field_types() with various primitive types."""
    msg_def = """
bool bool_field
byte byte_field
char char_field
float32 float32_field
float64 float64_field
int8 int8_field
uint8 uint8_field
int16 int16_field
uint16 uint16_field
int32 int32_field
uint32 uint32_field
int64 int64_field
uint64 uint64_field
string string_field
wstring wstring_field
"""
    plan = generate_plans("test_msgs/PrimitiveTypes", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "bool_field": "bool",
        "byte_field": "byte",
        "char_field": "char",
        "float32_field": "float32",
        "float64_field": "float64",
        "int8_field": "int8",
        "uint8_field": "uint8",
        "int16_field": "int16",
        "uint16_field": "uint16",
        "int32_field": "int32",
        "uint32_field": "uint32",
        "int64_field": "int64",
        "uint64_field": "uint64",
        "string_field": "string",
        "wstring_field": "wstring",
    }

    assert fields == expected


def test_bounded_strings():
    """Test get_fields_and_field_types() with bounded strings."""
    msg_def = """
string<=10 bounded_string
string unbounded_string
"""
    plan = generate_plans("test_msgs/BoundedStrings", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "bounded_string": "string<=10",
        "unbounded_string": "string",
    }

    assert fields == expected


def test_unbounded_arrays():
    """Test get_fields_and_field_types() with unbounded arrays."""
    msg_def = """
int32[] int_array
string[] string_array
float64[] float_array
"""
    plan = generate_plans("test_msgs/UnboundedArrays", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "int_array": "sequence<int32>",
        "string_array": "sequence<string>",
        "float_array": "sequence<float64>",
    }

    assert fields == expected


def test_bounded_arrays():
    """Test get_fields_and_field_types() with bounded arrays."""
    msg_def = """
int32[<=5] bounded_int_array
string[<=10] bounded_string_array
"""
    plan = generate_plans("test_msgs/BoundedArrays", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "bounded_int_array": "sequence<int32, 5>",
        "bounded_string_array": "sequence<string, 10>",
    }

    assert fields == expected


def test_fixed_arrays():
    """Test get_fields_and_field_types() with fixed-size arrays."""
    msg_def = """
int32[5] fixed_int_array
float64[3] fixed_float_array
"""
    plan = generate_plans("test_msgs/FixedArrays", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "fixed_int_array": "int32[5]",
        "fixed_float_array": "float64[3]",
    }

    assert fields == expected


def test_complex_types():
    """Test get_fields_and_field_types() with complex/nested types."""
    msg_def = """
std_msgs/Header header
geometry_msgs/Point position
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
    plan = generate_plans("test_msgs/ComplexTypes", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "header": "std_msgs/Header",
        "position": "geometry_msgs/Point",
    }

    assert fields == expected


def test_complex_array_types():
    """Test get_fields_and_field_types() with arrays of complex types."""
    msg_def = """
geometry_msgs/Point[] points
geometry_msgs/Point[5] fixed_points
geometry_msgs/Point[<=10] bounded_points
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
"""
    plan = generate_plans("test_msgs/ComplexArrays", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "points": "sequence<geometry_msgs/Point>",
        "fixed_points": "geometry_msgs/Point[5]",
        "bounded_points": "sequence<geometry_msgs/Point, 10>",
    }

    assert fields == expected


def test_mixed_types():
    """Test get_fields_and_field_types() with a mix of different types."""
    msg_def = """
# Example from user's request
string name
uint32 offset
uint8 datatype
uint32 count
"""
    plan = generate_plans("test_msgs/MultiArrayDimension", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "name": "string",
        "offset": "uint32",
        "datatype": "uint8",
        "count": "uint32",
    }

    assert fields == expected


def test_return_is_copy():
    """Test that get_fields_and_field_types() returns a copy, not the original."""
    msg_def = """
int32 x
int32 y
"""
    plan = generate_plans("test_msgs/Point2D", msg_def)
    msg_class = plan[0]

    fields1 = msg_class.get_fields_and_field_types()
    fields2 = msg_class.get_fields_and_field_types()

    # Should be equal but not the same object
    assert fields1 == fields2
    assert fields1 is not fields2

    # Modifying one shouldn't affect the other
    fields1["new_field"] = "int32"
    assert "new_field" not in fields2
    assert "new_field" not in msg_class.get_fields_and_field_types()


def test_empty_message():
    """Test get_fields_and_field_types() with a message that has no fields."""
    msg_def = """
# Empty message with no fields
"""
    plan = generate_plans("test_msgs/Empty", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    assert fields == {}


def test_ros2_standard_example():
    """Test with a standard ROS2 message example."""
    msg_def = """
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
    plan = generate_plans("sensor_msgs/CompressedImage", msg_def)
    msg_class = plan[0]

    fields = msg_class.get_fields_and_field_types()

    expected = {
        "header": "std_msgs/Header",
        "format": "string",
        "data": "sequence<uint8>",
    }

    assert fields == expected
