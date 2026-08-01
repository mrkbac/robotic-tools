"""Tests for basic ROS2 primitive types."""

from ros_parser import Type
from ros_parser.ros2_msg import parse_message_string


def test_single_field():
    """Test parsing a single field."""
    msg = parse_message_string("string name")
    assert len(msg.fields) == 1
    assert msg.fields[0].name == "name"
    assert msg.fields[0].type.type_name == "string"
    assert msg.fields[0].type.is_primitive
    assert not msg.fields[0].type.is_array


def test_all_primitive_types():
    """Test parsing all primitive types."""
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
        "wstring",
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
    """Test parsing time and duration types."""
    msg = parse_message_string("time timestamp\nduration elapsed")
    assert len(msg.fields) == 2
    assert msg.fields[0].name == "timestamp"
    assert msg.fields[0].type.type_name == "time"
    assert msg.fields[1].name == "elapsed"
    assert msg.fields[1].type.type_name == "duration"


def test_type_aliases():
    """Test that type names are kept as-is, matching reference parser behavior."""
    # char and byte are NOT normalized to uint8 at parse time (reference parser behavior)
    msg = parse_message_string("char c\nbyte b")
    assert len(msg.fields) == 2
    assert msg.fields[0].type.type_name == "char"
    assert msg.fields[1].type.type_name == "byte"


def test_builtin_interfaces_types():
    """Test that builtin_interfaces types are kept as message types, not primitives."""
    msg = parse_message_string("builtin_interfaces/Time stamp\nbuiltin_interfaces/Duration dur")
    assert len(msg.fields) == 2
    # builtin_interfaces/Time and builtin_interfaces/Duration are message types, not primitives
    assert msg.fields[0].type.type_name == "Time"
    assert msg.fields[0].type.package_name == "builtin_interfaces"
    assert msg.fields[1].type.type_name == "Duration"
    assert msg.fields[1].type.package_name == "builtin_interfaces"


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


def test_recorded_schema_field_names_preserve_spelling():
    msg = parse_message_string("string Name\nint32 Foo\nuint8 field__status")

    assert [field.name for field in msg.fields] == ["Name", "Foo", "field__status"]


def test_recorded_schema_constant_names_preserve_spelling():
    msg = parse_message_string("int32 Status=1\nint32 status__code=2")

    assert [constant.name for constant in msg.constants] == ["Status", "status__code"]


def test_type_string_representation():
    """Test Type.__str__ method."""
    t = Type(type_name="string")
    assert str(t) == "string"

    t = Type(type_name="int32")
    assert str(t) == "int32"

    t = Type(type_name="Point", package_name="geometry_msgs")
    assert str(t) == "geometry_msgs/Point"


def test_complex_type():
    """Test parsing complex types."""
    msg = parse_message_string("geometry_msgs/Point position")
    assert len(msg.fields) == 1
    assert msg.fields[0].name == "position"
    assert msg.fields[0].type.type_name == "Point"
    assert msg.fields[0].type.package_name == "geometry_msgs"
    assert not msg.fields[0].type.is_primitive
