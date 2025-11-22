"""Tests for ROS1 array types."""

import pytest
from ros_parser._lark_standalone_runtime import UnexpectedToken
from ros_parser.ros1_msg import parse_message_string

from .reference_parser import parse_message_string as reference_parse


def test_unbounded_array():
    """Test parsing unbounded array types."""
    msg = parse_message_string("int32[] values")
    assert len(msg.fields) == 1
    assert msg.fields[0].name == "values"
    assert msg.fields[0].type.type_name == "int32"
    assert msg.fields[0].type.is_array
    assert msg.fields[0].type.is_dynamic_array
    assert msg.fields[0].type.array_size is None
    assert not msg.fields[0].type.is_upper_bound


def test_fixed_array():
    """Test parsing fixed-size array types."""
    msg = parse_message_string("float64[10] matrix")
    assert len(msg.fields) == 1
    assert msg.fields[0].name == "matrix"
    assert msg.fields[0].type.type_name == "float64"
    assert msg.fields[0].type.is_array
    assert msg.fields[0].type.is_fixed_array
    assert msg.fields[0].type.array_size == 10
    assert not msg.fields[0].type.is_upper_bound


def test_string_array():
    """Test parsing string array types."""
    msg = parse_message_string("string[] names\nstring[5] labels")
    assert len(msg.fields) == 2
    assert msg.fields[0].type.is_dynamic_array
    assert msg.fields[1].type.is_fixed_array
    assert msg.fields[1].type.array_size == 5


def test_various_array_sizes():
    """Test arrays with various sizes."""
    definition = """
    int32[1] single
    int32[3] triple
    int32[100] hundred
    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 3
    assert msg.fields[0].type.array_size == 1
    assert msg.fields[1].type.array_size == 3
    assert msg.fields[2].type.array_size == 100


def test_complex_type_arrays():
    """Test arrays of complex types."""
    msg = parse_message_string("geometry_msgs/Point[] points\nstd_msgs/String[10] messages")
    assert len(msg.fields) == 2

    # Unbounded array of complex type
    assert msg.fields[0].name == "points"
    assert msg.fields[0].type.type_name == "Point"
    assert msg.fields[0].type.package_name == "geometry_msgs"
    assert msg.fields[0].type.is_dynamic_array

    # Fixed array of complex type
    assert msg.fields[1].name == "messages"
    assert msg.fields[1].type.type_name == "String"
    assert msg.fields[1].type.package_name == "std_msgs"
    assert msg.fields[1].type.is_fixed_array
    assert msg.fields[1].type.array_size == 10


def test_no_bounded_arrays():
    """Test that ROS1 doesn't support bounded arrays (<=N syntax)."""
    # ROS1 doesn't have bounded arrays, so this should fail
    # The grammar doesn't include bounded_array rule
    with pytest.raises(UnexpectedToken):
        parse_message_string("int32[<=10] values")


def test_comparison_with_reference_parser():
    """Test that our array parsing matches the reference parser."""
    definition = """
    int32[] dynamic
    float64[5] fixed
    string[] names
    """

    our_msg = parse_message_string(definition)
    ref_msg = reference_parse(definition)

    assert len(our_msg.fields) == len(ref_msg.fields)

    for our_field, ref_field in zip(our_msg.fields, ref_msg.fields, strict=True):
        assert our_field.name == ref_field.name
        # Reference parser includes array syntax in field_type
        assert our_field.type.type_name in ref_field.field_type


def test_mixed_arrays_and_scalars():
    """Test message with mix of arrays and scalar fields."""
    definition = """
    int32 count
    float64[] values
    string name
    bool[10] flags
    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 4
    assert not msg.fields[0].type.is_array  # scalar
    assert msg.fields[1].type.is_dynamic_array
    assert not msg.fields[2].type.is_array  # scalar
    assert msg.fields[3].type.is_fixed_array
