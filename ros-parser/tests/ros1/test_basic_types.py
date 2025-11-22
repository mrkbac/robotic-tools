"""Tests for basic ROS1 primitive types."""

from ros_parser.ros1_msg import parse_message_string

from .reference_parser import parse_message_string as reference_parse


def test_single_field():
    """Test parsing a single field."""
    msg = parse_message_string("string name")
    assert len(msg.fields) == 1
    assert msg.fields[0].name == "name"
    assert msg.fields[0].type.type_name == "string"
    assert msg.fields[0].type.is_primitive
    assert not msg.fields[0].type.is_array


def test_all_primitive_types():
    """Test parsing all ROS1 primitive types."""
    primitive_types = [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
        "string",
        # ROS1 deprecated types
        "char",
        "byte",
    ]

    for ptype in primitive_types:
        msg = parse_message_string(f"{ptype} value")
        assert len(msg.fields) == 1
        assert msg.fields[0].name == "value"
        assert msg.fields[0].type.type_name == ptype
        assert msg.fields[0].type.is_primitive


def test_multiple_fields():
    """Test parsing multiple fields."""
    definition = """
    float64 x
    float64 y
    float64 z
    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 3
    assert msg.fields[0].name == "x"
    assert msg.fields[1].name == "y"
    assert msg.fields[2].name == "z"
    assert all(f.type.type_name == "float64" for f in msg.fields)


def test_time_and_duration():
    """Test parsing time and duration types (ROS1 built-ins)."""
    msg = parse_message_string("time timestamp\nduration elapsed")
    assert len(msg.fields) == 2
    assert msg.fields[0].name == "timestamp"
    assert msg.fields[0].type.type_name == "time"
    assert msg.fields[0].type.is_primitive
    assert msg.fields[1].name == "elapsed"
    assert msg.fields[1].type.type_name == "duration"
    assert msg.fields[1].type.is_primitive


def test_type_aliases():
    """Test that type names are kept as-is, matching reference parser behavior."""
    # char and byte are NOT normalized to uint8 at parse time
    msg = parse_message_string("char c\nbyte b")
    assert len(msg.fields) == 2
    assert msg.fields[0].type.type_name == "char"
    assert msg.fields[1].type.type_name == "byte"


def test_comments_ignored():
    """Test that comments are properly ignored."""
    definition = """
    # This is a comment
    string name  # inline comment
    # Another comment
    int32 value
    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 2
    assert msg.fields[0].name == "name"
    assert msg.fields[1].name == "value"


def test_empty_lines_ignored():
    """Test that empty lines are properly ignored."""
    definition = """
    string name

    int32 value

    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 2


def test_comparison_with_reference_parser():
    """Test that our parser matches the reference parser for basic types."""
    definition = """
    bool enabled
    int32 count
    float64 ratio
    string name
    time timestamp
    duration elapsed
    """

    # Parse with both parsers
    our_msg = parse_message_string(definition)
    ref_msg = reference_parse(definition)

    # Compare field counts
    assert len(our_msg.fields) == len(ref_msg.fields)

    # Compare each field
    for our_field, ref_field in zip(our_msg.fields, ref_msg.fields, strict=True):
        assert our_field.name == ref_field.name
        assert our_field.type.type_name == ref_field.field_type
