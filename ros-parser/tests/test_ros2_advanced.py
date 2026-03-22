"""Tests for advanced ROS2 parsing: defaults, bounded arrays/strings, services, actions."""

import pytest
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
