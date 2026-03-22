"""Tests for ROS1 parser edge cases."""

import pytest
from ros_parser.ros1_msg import parse_service_string, parse_string

# ---------------------------------------------------------------------------
# 1. String constants (single-quoted and double-quoted)
# ---------------------------------------------------------------------------


class TestStringConstants:
    def test_double_quoted_string_constant(self):
        msg = parse_string('string GREETING="hello world"')
        assert len(msg.constants) == 1
        assert msg.constants[0].name == "GREETING"
        assert msg.constants[0].value == "hello world"

    def test_single_quoted_string_constant(self):
        msg = parse_string("string GREETING='hello world'")
        assert len(msg.constants) == 1
        assert msg.constants[0].name == "GREETING"
        assert msg.constants[0].value == "hello world"

    def test_quoted_string_with_escape(self):
        msg = parse_string(r'string MSG="line1\nline2"')
        assert msg.constants[0].value == "line1\nline2"

    def test_empty_double_quoted_string(self):
        msg = parse_string('string EMPTY=""')
        assert msg.constants[0].value == ""

    def test_empty_single_quoted_string(self):
        msg = parse_string("string EMPTY=''")
        assert msg.constants[0].value == ""

    def test_quoted_string_with_hash(self):
        """Hash inside quotes should not be treated as a comment."""
        msg = parse_string('string TAG="#robot"')
        assert msg.constants[0].value == "#robot"


# ---------------------------------------------------------------------------
# 2. Boolean constants
# ---------------------------------------------------------------------------


class TestBooleanConstants:
    def test_true_lowercase(self):
        msg = parse_string("bool FLAG=true")
        assert msg.constants[0].value is True

    def test_true_capitalized(self):
        msg = parse_string("bool FLAG=True")
        assert msg.constants[0].value is True

    def test_false_lowercase(self):
        msg = parse_string("bool FLAG=false")
        assert msg.constants[0].value is False

    def test_false_capitalized(self):
        msg = parse_string("bool FLAG=False")
        assert msg.constants[0].value is False

    def test_bool_one_parsed_as_int(self):
        """1 for bool is parsed as numeric integer, not boolean True."""
        msg = parse_string("bool FLAG=1")
        assert msg.constants[0].value == 1
        assert isinstance(msg.constants[0].value, int)
        assert not isinstance(msg.constants[0].value, bool)

    def test_bool_zero_parsed_as_int(self):
        """0 for bool is parsed as numeric integer, not boolean False."""
        msg = parse_string("bool FLAG=0")
        assert msg.constants[0].value == 0
        assert isinstance(msg.constants[0].value, int)
        assert not isinstance(msg.constants[0].value, bool)


# ---------------------------------------------------------------------------
# 3. Numeric constants (hex, float with exponent, negative)
# ---------------------------------------------------------------------------


class TestNumericConstants:
    def test_hex_constant(self):
        msg = parse_string("uint8 MASK=0xFF")
        assert msg.constants[0].value == 255

    def test_hex_constant_lowercase(self):
        msg = parse_string("uint8 MASK=0xff")
        assert msg.constants[0].value == 255

    def test_hex_constant_mixed_case(self):
        msg = parse_string("uint8 MASK=0xAb")
        assert msg.constants[0].value == 0xAB

    def test_float_scientific_notation(self):
        msg = parse_string("float64 BIG=1.5e10")
        assert msg.constants[0].value == pytest.approx(1.5e10)

    def test_float_scientific_notation_uppercase_e(self):
        msg = parse_string("float64 BIG=1.5E10")
        assert msg.constants[0].value == pytest.approx(1.5e10)

    def test_float_negative_exponent(self):
        msg = parse_string("float64 SMALL=1.5e-3")
        assert msg.constants[0].value == pytest.approx(1.5e-3)

    def test_negative_integer(self):
        msg = parse_string("int32 NEG=-42")
        assert msg.constants[0].value == -42

    def test_negative_float(self):
        msg = parse_string("float64 NEG=-3.14")
        assert msg.constants[0].value == pytest.approx(-3.14)

    def test_zero_integer(self):
        msg = parse_string("int32 ZERO=0")
        assert msg.constants[0].value == 0

    def test_binary_constant(self):
        msg = parse_string("uint8 BITS=0b11001100")
        assert msg.constants[0].value == 0b11001100

    def test_octal_constant(self):
        msg = parse_string("uint8 PERM=0o755")
        assert msg.constants[0].value == 0o755


# ---------------------------------------------------------------------------
# 4. Header type special case
# ---------------------------------------------------------------------------


class TestHeaderSpecialCase:
    def test_header_without_package_resolves_to_std_msgs(self):
        """Header without package prefix should resolve to std_msgs/Header."""
        msg = parse_string("Header header")
        assert msg.fields[0].type.type_name == "Header"
        assert msg.fields[0].type.package_name == "std_msgs"

    def test_header_ignores_context_package(self):
        """Header should resolve to std_msgs even if context_package_name differs."""
        msg = parse_string("Header header", context_package_name="sensor_msgs")
        assert msg.fields[0].type.package_name == "std_msgs"

    def test_explicit_std_msgs_header(self):
        """Explicitly qualified std_msgs/Header should also work."""
        msg = parse_string("std_msgs/Header header")
        assert msg.fields[0].type.type_name == "Header"
        assert msg.fields[0].type.package_name == "std_msgs"

    def test_header_is_not_primitive(self):
        msg = parse_string("Header header")
        assert not msg.fields[0].type.is_primitive

    def test_header_array(self):
        """Header should still resolve when used as array element type."""
        msg = parse_string("Header[] headers")
        assert msg.fields[0].type.type_name == "Header"
        assert msg.fields[0].type.package_name == "std_msgs"
        assert msg.fields[0].type.is_array


# ---------------------------------------------------------------------------
# 5. Service definitions using parse_service_string
# ---------------------------------------------------------------------------


class TestServiceDefinitions:
    def test_basic_service(self):
        srv = parse_service_string(
            "AddTwoInts",
            "int64 a\nint64 b\n---\nint64 sum",
        )
        assert srv.name == "AddTwoInts"
        assert len(srv.request.fields) == 2
        assert len(srv.response.fields) == 1
        assert srv.request.fields[0].name == "a"
        assert srv.request.fields[1].name == "b"
        assert srv.response.fields[0].name == "sum"

    def test_service_with_package_name(self):
        srv = parse_service_string(
            "GetMap",
            "---\nnav_msgs/OccupancyGrid map",
            package_name="nav_msgs",
        )
        assert srv.name == "nav_msgs/GetMap"
        assert srv.request.name == "nav_msgs/GetMap_Request"
        assert srv.response.name == "nav_msgs/GetMap_Response"

    def test_service_empty_request(self):
        srv = parse_service_string("Trigger", "---\nbool success\nstring message")
        assert len(srv.request.fields) == 0
        assert len(srv.response.fields) == 2

    def test_service_empty_response(self):
        srv = parse_service_string("Empty", "string data\n---")
        assert len(srv.request.fields) == 1
        assert len(srv.response.fields) == 0

    def test_service_empty_both(self):
        srv = parse_service_string("Empty", "---")
        assert len(srv.request.fields) == 0
        assert len(srv.response.fields) == 0

    def test_service_missing_separator_raises(self):
        with pytest.raises(ValueError, match="exactly one '---' separator"):
            parse_service_string("Bad", "int32 x\nint32 y")

    def test_service_multiple_separators_raises(self):
        with pytest.raises(ValueError, match="exactly one '---' separator"):
            parse_service_string("Bad", "int32 x\n---\nint32 y\n---\nint32 z")


# ---------------------------------------------------------------------------
# 6. Complex types with packages
# ---------------------------------------------------------------------------


class TestComplexTypesWithPackages:
    def test_fully_qualified_complex_type(self):
        msg = parse_string("geometry_msgs/Point pos")
        f = msg.fields[0]
        assert f.name == "pos"
        assert f.type.type_name == "Point"
        assert f.type.package_name == "geometry_msgs"
        assert not f.type.is_primitive

    def test_multiple_different_packages(self):
        definition = """
        geometry_msgs/Point position
        sensor_msgs/Image image
        std_msgs/String label
        """
        msg = parse_string(definition)
        assert len(msg.fields) == 3
        assert msg.fields[0].type.package_name == "geometry_msgs"
        assert msg.fields[1].type.package_name == "sensor_msgs"
        assert msg.fields[2].type.package_name == "std_msgs"

    def test_complex_type_array(self):
        msg = parse_string("geometry_msgs/Point[] waypoints")
        f = msg.fields[0]
        assert f.type.type_name == "Point"
        assert f.type.package_name == "geometry_msgs"
        assert f.type.is_array
        assert f.type.is_dynamic_array

    def test_complex_type_fixed_array(self):
        msg = parse_string("geometry_msgs/Point[3] triangle")
        f = msg.fields[0]
        assert f.type.package_name == "geometry_msgs"
        assert f.type.is_fixed_array
        assert f.type.array_size == 3

    def test_local_type_with_context(self):
        msg = parse_string("Pose data", context_package_name="geometry_msgs")
        assert msg.fields[0].type.type_name == "Pose"
        assert msg.fields[0].type.package_name == "geometry_msgs"

    def test_local_type_without_context(self):
        msg = parse_string("CustomMsg data")
        assert msg.fields[0].type.type_name == "CustomMsg"
        assert msg.fields[0].type.package_name is None

    def test_mixed_primitives_and_complex(self):
        definition = """
        std_msgs/Header header
        float64 x
        geometry_msgs/Twist twist
        string label
        """
        msg = parse_string(definition)
        assert len(msg.fields) == 4
        assert not msg.fields[0].type.is_primitive
        assert msg.fields[1].type.is_primitive
        assert not msg.fields[2].type.is_primitive
        assert msg.fields[3].type.is_primitive
