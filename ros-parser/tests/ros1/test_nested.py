"""Tests for ROS1 nested/complex message types."""

from ros_parser.ros1_msg import parse_message_string

from .reference_parser import parse_message_string as reference_parse


def test_simple_complex_type():
    """Test parsing a simple complex type."""
    msg = parse_message_string("geometry_msgs/Point position")
    assert len(msg.fields) == 1
    assert msg.fields[0].name == "position"
    assert msg.fields[0].type.type_name == "Point"
    assert msg.fields[0].type.package_name == "geometry_msgs"
    assert not msg.fields[0].type.is_primitive


def test_multiple_complex_types():
    """Test message with multiple complex types."""
    definition = """
    geometry_msgs/Point position
    geometry_msgs/Quaternion orientation
    std_msgs/Header header
    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 3
    assert msg.fields[0].type.package_name == "geometry_msgs"
    assert msg.fields[1].type.package_name == "geometry_msgs"
    assert msg.fields[2].type.package_name == "std_msgs"


def test_local_type_with_context():
    """Test parsing local types with package context."""
    # When parsing with package context, local types get qualified
    msg = parse_message_string("Point position", context_package_name="geometry_msgs")
    assert len(msg.fields) == 1
    assert msg.fields[0].type.type_name == "Point"
    assert msg.fields[0].type.package_name == "geometry_msgs"


def test_local_type_without_context():
    """Test parsing local types without package context."""
    msg = parse_message_string("CustomType data")
    assert len(msg.fields) == 1
    assert msg.fields[0].type.type_name == "CustomType"
    assert msg.fields[0].type.package_name is None


def test_header_auto_resolution():
    """Test that 'Header' automatically resolves to std_msgs/Header."""
    msg = parse_message_string("Header header")
    assert len(msg.fields) == 1
    assert msg.fields[0].type.type_name == "Header"
    assert msg.fields[0].type.package_name == "std_msgs"


def test_header_in_any_package_context():
    """Test that Header resolves to std_msgs even with different context."""
    msg = parse_message_string("Header header", context_package_name="my_package")
    assert msg.fields[0].type.package_name == "std_msgs"


def test_explicit_std_msgs_header():
    """Test explicitly qualified Header."""
    msg = parse_message_string("std_msgs/Header header")
    assert msg.fields[0].type.type_name == "Header"
    assert msg.fields[0].type.package_name == "std_msgs"


def test_mixed_primitive_and_complex():
    """Test message with mix of primitive and complex types."""
    definition = """
    std_msgs/Header header
    int32 sequence
    geometry_msgs/Pose pose
    string name
    """
    msg = parse_message_string(definition)
    assert len(msg.fields) == 4
    assert not msg.fields[0].type.is_primitive  # Header
    assert msg.fields[1].type.is_primitive  # int32
    assert not msg.fields[2].type.is_primitive  # Pose
    assert msg.fields[3].type.is_primitive  # string


def test_comparison_with_reference_parser():
    """Test that complex type parsing matches reference parser."""
    definition = """
    geometry_msgs/Point position
    geometry_msgs/Quaternion orientation
    """

    our_msg = parse_message_string(definition)
    ref_msg = reference_parse(definition)

    assert len(our_msg.fields) == len(ref_msg.fields)

    for our_field, ref_field in zip(our_msg.fields, ref_msg.fields, strict=True):
        assert our_field.name == ref_field.name
        assert ref_field.field_type.startswith(our_field.type.package_name or "")
        assert our_field.type.type_name in ref_field.field_type


def test_comparison_with_reference_header():
    """Test Header resolution matches reference parser."""
    definition = "Header header"

    our_msg = parse_message_string(definition)
    ref_msg = reference_parse(definition)

    assert len(our_msg.fields) == 1
    assert len(ref_msg.fields) == 1
    assert our_msg.fields[0].name == ref_msg.fields[0].name
    # Reference parser resolves Header to std_msgs/Header
    assert ref_msg.fields[0].field_type == "std_msgs/Header"
    assert our_msg.fields[0].type.package_name == "std_msgs"
    assert our_msg.fields[0].type.type_name == "Header"


def test_comparison_with_reference_local_types():
    """Test local type resolution matches reference parser."""
    definition = """
    Point position
    Quaternion orientation
    """

    our_msg = parse_message_string(definition, context_package_name="geometry_msgs")
    ref_msg = reference_parse(definition, package_context="geometry_msgs")

    assert len(our_msg.fields) == len(ref_msg.fields)

    for our_field, ref_field in zip(our_msg.fields, ref_msg.fields, strict=True):
        assert our_field.name == ref_field.name
        # Reference parser fully qualifies local types
        assert ref_field.field_type == f"geometry_msgs/{our_field.type.type_name}"
        assert our_field.type.package_name == "geometry_msgs"


def test_array_of_complex_types():
    """Test arrays of complex message types."""
    msg = parse_message_string("geometry_msgs/Point[] waypoints")
    assert len(msg.fields) == 1
    assert msg.fields[0].type.type_name == "Point"
    assert msg.fields[0].type.package_name == "geometry_msgs"
    assert msg.fields[0].type.is_array
    assert msg.fields[0].type.is_dynamic_array


def test_fixed_array_of_complex_types():
    """Test fixed arrays of complex message types."""
    msg = parse_message_string("std_msgs/String[10] messages")
    assert len(msg.fields) == 1
    assert msg.fields[0].type.type_name == "String"
    assert msg.fields[0].type.package_name == "std_msgs"
    assert msg.fields[0].type.is_fixed_array
    assert msg.fields[0].type.array_size == 10
