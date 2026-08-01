"""Tests for advanced ROS2 parsing: defaults, bounded arrays/strings, services, actions."""

import math

import pytest
from ros_parser import MessageDefinitionError
from ros_parser.ros2_msg import parse_action_string, parse_message_string, parse_service_string


class TestDefaultValues:
    """Test parsing fields with default values."""

    def test_bool_default(self):
        msg = parse_message_string("bool flag true")
        assert len(msg.fields) == 1
        f = msg.fields[0]
        assert f.name == "flag"
        assert f.type.type_name == "bool"
        assert f.default_value is True

    def test_bool_default_false(self):
        msg = parse_message_string("bool flag false")
        assert msg.fields[0].default_value is False

    def test_int32_default(self):
        msg = parse_message_string("int32 count 42")
        f = msg.fields[0]
        assert f.name == "count"
        assert f.type.type_name == "int32"
        assert f.default_value == 42

    def test_negative_int_default(self):
        msg = parse_message_string("int32 offset -10")
        assert msg.fields[0].default_value == -10

    def test_float64_default(self):
        msg = parse_message_string("float64 rate 1.5")
        f = msg.fields[0]
        assert f.name == "rate"
        assert f.type.type_name == "float64"
        assert f.default_value == 1.5

    def test_string_default_quoted(self):
        msg = parse_message_string('string name "hello"')
        f = msg.fields[0]
        assert f.name == "name"
        assert f.type.type_name == "string"
        assert f.default_value == "hello"

    def test_string_default_single_quoted(self):
        msg = parse_message_string("string name 'world'")
        f = msg.fields[0]
        assert f.default_value == "world"

    def test_array_default(self):
        msg = parse_message_string("int32[3] values [1, 2, 3]")
        f = msg.fields[0]
        assert f.name == "values"
        assert f.type.type_name == "int32"
        assert f.type.is_array is True
        assert f.type.array_size == 3
        assert f.type.is_upper_bound is False
        assert f.default_value == [1, 2, 3]

    def test_field_without_default(self):
        msg = parse_message_string("int32 count")
        assert msg.fields[0].default_value is None

    def test_multiple_fields_with_defaults(self):
        definition = """
        bool active true
        int32 count 10
        float64 scale 2.0
        """
        msg = parse_message_string(definition)
        assert len(msg.fields) == 3
        assert msg.fields[0].default_value is True
        assert msg.fields[1].default_value == 10
        assert msg.fields[2].default_value == 2.0


class TestBoundedArrays:
    """Test parsing bounded array types."""

    def test_bounded_array(self):
        msg = parse_message_string("int32[<=5] data")
        f = msg.fields[0]
        assert f.name == "data"
        assert f.type.type_name == "int32"
        assert f.type.is_array is True
        assert f.type.is_upper_bound is True
        assert f.type.array_size == 5

    def test_unbounded_array(self):
        msg = parse_message_string("int32[] data")
        f = msg.fields[0]
        assert f.type.is_array is True
        assert f.type.array_size is None
        assert f.type.is_upper_bound is False

    def test_fixed_array(self):
        msg = parse_message_string("int32[3] data")
        f = msg.fields[0]
        assert f.type.is_array is True
        assert f.type.array_size == 3
        assert f.type.is_upper_bound is False
        assert f.type.is_fixed_array is True

    def test_bounded_array_is_dynamic(self):
        msg = parse_message_string("float64[<=100] measurements")
        f = msg.fields[0]
        assert f.type.is_dynamic_array is True
        assert f.type.is_fixed_array is False


class TestBoundedStrings:
    """Test parsing bounded string types."""

    def test_bounded_string(self):
        msg = parse_message_string("string<=10 name")
        f = msg.fields[0]
        assert f.name == "name"
        assert f.type.type_name == "string"
        assert f.type.string_upper_bound == 10

    def test_bounded_wstring(self):
        msg = parse_message_string("wstring<=20 label")
        f = msg.fields[0]
        assert f.type.type_name == "wstring"
        assert f.type.string_upper_bound == 20

    def test_unbounded_string(self):
        msg = parse_message_string("string text")
        assert msg.fields[0].type.string_upper_bound is None

    def test_bounded_string_str_repr(self):
        msg = parse_message_string("string<=10 name")
        assert str(msg.fields[0].type) == "string<=10"


class TestServiceDefinitions:
    """Test parsing service definitions with --- separator."""

    def test_simple_service(self):
        srv_def = """\
int32 a
int32 b
---
int32 sum"""
        srv = parse_service_string("AddTwoInts", srv_def, "example_interfaces")
        assert srv.name == "example_interfaces/AddTwoInts"

        req = srv.request
        assert req.name == "example_interfaces/AddTwoInts_Request"
        assert len(req.fields) == 2
        assert req.fields[0].name == "a"
        assert req.fields[1].name == "b"

        resp = srv.response
        assert resp.name == "example_interfaces/AddTwoInts_Response"
        assert len(resp.fields) == 1
        assert resp.fields[0].name == "sum"

    def test_service_empty_sections(self):
        srv_def = "---\nint32 result"
        srv = parse_service_string("EmptyRequest", srv_def)
        assert len(srv.request.fields) == 0
        assert len(srv.response.fields) == 1

    def test_service_no_separator_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            parse_service_string("Bad", "int32 x")

    def test_service_two_separators_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            parse_service_string("Bad", "int32 x\n---\nint32 y\n---\nint32 z")


class TestActionDefinitions:
    """Test parsing action definitions with two --- separators."""

    def test_simple_action(self):
        action_def = """\
int32 order
---
int32[] sequence
---
int32[] partial_sequence"""
        action = parse_action_string("Fibonacci", action_def, "example_interfaces")
        assert action.name == "example_interfaces/Fibonacci"

        assert action.goal.name == "example_interfaces/Fibonacci_Goal"
        assert len(action.goal.fields) == 1
        assert action.goal.fields[0].name == "order"

        assert action.result.name == "example_interfaces/Fibonacci_Result"
        assert len(action.result.fields) == 1
        assert action.result.fields[0].name == "sequence"
        assert action.result.fields[0].type.is_array is True

        assert action.feedback.name == "example_interfaces/Fibonacci_Feedback"
        assert len(action.feedback.fields) == 1
        assert action.feedback.fields[0].name == "partial_sequence"
        assert action.feedback.fields[0].type.is_array is True

    def test_action_without_package(self):
        action_def = "int32 goal_val\n---\nint32 result_val\n---\nint32 feedback_val"
        action = parse_action_string("MyAction", action_def)
        assert action.name == "MyAction"
        assert action.goal.name == "MyAction_Goal"
        assert action.result.name == "MyAction_Result"
        assert action.feedback.name == "MyAction_Feedback"

    def test_action_wrong_separators_raises(self):
        with pytest.raises(ValueError, match="exactly two"):
            parse_action_string("Bad", "int32 x\n---\nint32 y")

    def test_action_no_separators_raises(self):
        with pytest.raises(ValueError, match="exactly two"):
            parse_action_string("Bad", "int32 x")

    def test_action_multi_field_sections(self):
        action_def = """\
float64 target_x
float64 target_y
---
bool success
string message
---
float64 percent_complete"""
        action = parse_action_string("Navigate", action_def, "nav2_msgs")
        assert len(action.goal.fields) == 2
        assert len(action.result.fields) == 2
        assert len(action.feedback.fields) == 1
        assert action.result.fields[0].type.type_name == "bool"
        assert action.result.fields[1].type.type_name == "string"


class TestConstants:
    """Test parsing constant definitions."""

    def test_int_constant(self):
        msg = parse_message_string("int32 MAX_SIZE=100")
        assert len(msg.constants) == 1
        assert msg.constants[0].name == "MAX_SIZE"
        assert msg.constants[0].value == 100

    def test_string_constant(self):
        msg = parse_message_string("string LABEL='hello'")
        assert msg.constants[0].name == "LABEL"
        assert msg.constants[0].value == "hello"

    def test_mixed_fields_and_constants(self):
        definition = """\
int32 STATUS_OK=0
int32 STATUS_ERR=1
int32 status
string message"""
        msg = parse_message_string(definition)
        assert len(msg.constants) == 2
        assert len(msg.fields) == 2
        assert len(msg.fields_all) == 4


class TestSemanticValidation:
    @pytest.mark.parametrize(
        "definition",
        [
            "string<=0 label",
            "int32[0] values",
            "int32[<=0] values",
        ],
    )
    def test_bounds_must_be_positive(self, definition):
        with pytest.raises(ValueError, match="greater than zero"):
            parse_message_string(definition)

    @pytest.mark.parametrize(
        "definition",
        [
            "uint8 value 256",
            "int8 value -129",
            "int32 value 1.5",
            "bool value 2",
            'float64 value "not a number"',
            'string<=2 value "abc"',
            "time timestamp 0",
        ],
    )
    def test_field_defaults_match_declared_type(self, definition):
        with pytest.raises(MessageDefinitionError, match="default"):
            parse_message_string(definition)

    @pytest.mark.parametrize(
        "definition",
        [
            "uint8 MASK=256",
            "int32 COUNT=1.5",
            "bool ENABLED=2",
            "string<=2 LABEL='abc'",
        ],
    )
    def test_constants_match_declared_type(self, definition):
        with pytest.raises(MessageDefinitionError, match="Constant"):
            parse_message_string(definition)

    def test_ros2_byte_uses_unsigned_octet_range(self):
        field = parse_message_string("byte value 255").fields[0]
        assert field.default_value == 255

        with pytest.raises(MessageDefinitionError, match="default"):
            parse_message_string("byte value -1")

        constants = parse_message_string("byte MIN=0\nbyte MAX=255").constants
        assert [constant.value for constant in constants] == [0, 255]
        for value in (-1, 256):
            with pytest.raises(MessageDefinitionError, match="default"):
                parse_message_string(f"byte VALUE={value}")

    @pytest.mark.parametrize(
        ("definition", "expected"),
        [("bool enabled 0", False), ("bool enabled 1", True)],
    )
    def test_bool_numeric_domain_is_normalized(self, definition, expected):
        field = parse_message_string(definition).fields[0]
        assert field.default_value is expected
        assert type(field.default_value) is bool

    @pytest.mark.parametrize(
        "definition",
        [
            "int32[3] values [1, 2]",
            "int32[<=2] values [1, 2, 3]",
            "int32[2] values 1",
        ],
    )
    def test_array_defaults_match_declared_cardinality(self, definition):
        with pytest.raises(MessageDefinitionError, match="array"):
            parse_message_string(definition)

    @pytest.mark.parametrize(
        "definition",
        [
            "geometry_msgs/Pose pose 1",
            "geometry_msgs/Pose[] poses [1]",
        ],
    )
    def test_prohibited_defaults_are_rejected(self, definition):
        with pytest.raises(MessageDefinitionError, match="default"):
            parse_message_string(definition)

    @pytest.mark.parametrize(
        ("definition", "expected"),
        [
            ("string label hello", "hello"),
            ("string value 1", "1"),
            ("string value 1.5", "1.5"),
            ("string value true", "true"),
            ("string value 0x1", "0x1"),
            ('string[] values ["hello", "a,b"]', ["hello", "a,b"]),
            ("string[] values [1, true, 1.5]", ["1", "true", "1.5"]),
            ("wstring[2] values [hello, world]", ["hello", "world"]),
            ("string[<=2] values [hello]", ["hello"]),
        ],
    )
    def test_string_defaults_match_rosidl_accepted_forms(self, definition, expected):
        assert parse_message_string(definition).fields[0].default_value == expected

    @pytest.mark.parametrize(("spelling", "expected"), [("TRUE", True), ("FALSE", False)])
    def test_bool_defaults_accept_case_insensitive_words(self, spelling, expected):
        value = parse_message_string(f"bool value {spelling}").fields[0].default_value
        assert value is expected

    @pytest.mark.parametrize("spelling", ["0x1", "+1", "-0"])
    def test_bool_defaults_reject_noncanonical_numeric_spellings(self, spelling):
        with pytest.raises(MessageDefinitionError, match="default"):
            parse_message_string(f"bool value {spelling}")

    def test_nonfinite_float_literals_and_string_constant_public_type(self):
        msg = parse_message_string('float32 value nan\nfloat64 NEG=-inf\nstring LABEL="x"')

        assert math.isnan(msg.fields[0].default_value)
        assert msg.constants[0].value == -math.inf
        assert type(msg.constants[1].value) is str
