"""Tests for ROS1 constant definitions."""

import pytest
from ros_parser.ros1_msg import parse_message_string

from .reference_parser import parse_message_string as reference_parse


def test_int_constant():
    """Test parsing integer constants."""
    msg = parse_message_string("int32 MAX_VALUE=100")
    assert len(msg.constants) == 1
    assert msg.constants[0].name == "MAX_VALUE"
    assert msg.constants[0].type.type_name == "int32"
    assert msg.constants[0].value == 100


def test_string_constant():
    """Test parsing string constants."""
    msg = parse_message_string("string NAME=robot")
    assert len(msg.constants) == 1
    assert msg.constants[0].name == "NAME"
    assert msg.constants[0].type.type_name == "string"
    assert msg.constants[0].value == "robot"


def test_string_constant_with_spaces():
    """Test that string constants preserve everything after =."""
    msg = parse_message_string("string MSG=Hello World")
    assert msg.constants[0].value == "Hello World"


def test_bool_constants():
    """Test parsing boolean constants."""
    definition = """
    bool ENABLED=true
    bool DISABLED=false
    bool ON=1
    bool OFF=0
    """
    msg = parse_message_string(definition)
    assert len(msg.constants) == 4
    assert msg.constants[0].value is True
    assert msg.constants[1].value is False
    # Note: 1 and 0 are parsed as integers, not converted to bool
    # This matches ROS1 genmsg behavior
    assert msg.constants[2].value == 1
    assert msg.constants[3].value == 0


def test_float_constants():
    """Test parsing float constants."""
    definition = """
    float32 PI=3.14159
    float64 E=2.71828
    """
    msg = parse_message_string(definition)
    assert len(msg.constants) == 2
    assert msg.constants[0].name == "PI"
    assert abs(msg.constants[0].value - 3.14159) < 0.00001
    assert msg.constants[1].name == "E"
    assert abs(msg.constants[1].value - 2.71828) < 0.00001


def test_hex_constant():
    """Test parsing hexadecimal constants."""
    msg = parse_message_string("int32 HEX_VAL=0xFF")
    assert msg.constants[0].value == 255


def test_binary_constant():
    """Test parsing binary constants."""
    msg = parse_message_string("int32 BIN_VAL=0b1010")
    assert msg.constants[0].value == 10


def test_octal_constant():
    """Test parsing octal constants."""
    msg = parse_message_string("int32 OCT_VAL=0o77")
    assert msg.constants[0].value == 63


def test_multiple_constants():
    """Test message with multiple constants."""
    definition = """
    int32 TYPE_A=1
    int32 TYPE_B=2
    int32 TYPE_C=3
    string VERSION=v1.0
    """
    msg = parse_message_string(definition)
    assert len(msg.constants) == 4
    assert msg.constants[0].value == 1
    assert msg.constants[1].value == 2
    assert msg.constants[2].value == 3
    assert msg.constants[3].value == "v1.0"


def test_fields_and_constants_mixed():
    """Test message with both fields and constants."""
    definition = """
    int32 MAX_SIZE=100
    string data
    int32 size
    bool ENABLED=true
    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 2
    assert len(msg.constants) == 2
    assert len(msg.fields_all) == 4  # total
    # Check order is preserved
    assert msg.fields_all[0].name == "MAX_SIZE"
    assert msg.fields_all[1].name == "data"
    assert msg.fields_all[2].name == "size"
    assert msg.fields_all[3].name == "ENABLED"


def test_constant_no_array():
    """Test that constants cannot be arrays."""
    with pytest.raises(ValueError, match="Constants must use primitive types"):
        parse_message_string("int32[] VALUES=1")


def test_constant_no_complex_type():
    """Test that constants cannot use complex types."""
    with pytest.raises(ValueError, match="Constants must use primitive types"):
        parse_message_string("geometry_msgs/Point ORIGIN=0")


def test_comparison_with_reference_parser():
    """Test that constant parsing matches the reference parser."""
    definition = """
int32 MAX=100
bool FLAG=true
float64 RATE=10.5
    """

    our_msg = parse_message_string(definition)
    ref_msg = reference_parse(definition)

    assert len(our_msg.constants) == len(ref_msg.constants)

    for our_const, ref_const in zip(our_msg.constants, ref_msg.constants, strict=True):
        assert our_const.name == ref_const.name
        assert our_const.type.type_name == ref_const.field_type
        assert our_const.value == ref_const.value


def test_constant_with_comment():
    """Test constants with inline comments."""
    msg = parse_message_string("int32 MAX=100  # maximum value")
    assert len(msg.constants) == 1
    assert msg.constants[0].value == 100


def test_negative_constants():
    """Test negative numeric constants."""
    definition = """
    int32 NEG_INT=-42
    float64 NEG_FLOAT=-3.14
    """
    msg = parse_message_string(definition)
    assert msg.constants[0].value == -42
    assert msg.constants[1].value == -3.14
