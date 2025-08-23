"""Tests for Message Path Grammar Parser.

Converted from TypeScript tests in LichtblickSuite.
"""

import pytest

from digitalis.grammar.parser import (
    ArrayAccess,
    FieldAccess,
    Filter,
    SliceNotation,
    Variable,
    parse_message_path,
)


class TestBasicParsing:
    """Test basic message path parsing functionality."""

    def test_simple_topic(self) -> None:
        """Test parsing simple topic names."""
        result = parse_message_path("/some_topic")
        assert result.topic_name == "/some_topic"
        assert result.message_path == []
        assert result.modifier is None

    def test_topic_without_leading_slash(self) -> None:
        """Test parsing topic names without leading slash."""
        result = parse_message_path("some_topic")
        assert result.topic_name == "some_topic"
        assert result.message_path == []

    def test_topic_with_field_access(self) -> None:
        """Test parsing topic with field access."""
        result = parse_message_path("/topic.field")
        assert result.topic_name == "/topic"
        assert len(result.message_path) == 1
        assert isinstance(result.message_path[0], FieldAccess)
        assert result.message_path[0].field == "field"

    def test_nested_field_access(self) -> None:
        """Test parsing nested field access."""
        result = parse_message_path("/topic.position.x")
        assert result.topic_name == "/topic"
        assert len(result.message_path) == 2
        assert isinstance(result.message_path[0], FieldAccess)
        assert result.message_path[0].field == "position"
        assert isinstance(result.message_path[1], FieldAccess)
        assert result.message_path[1].field == "x"

    def test_array_index_access(self) -> None:
        """Test parsing array index access."""
        result = parse_message_path("/topic.array[0]")
        assert result.topic_name == "/topic"
        assert len(result.message_path) == 2
        assert isinstance(result.message_path[0], FieldAccess)
        assert result.message_path[0].field == "array"
        assert isinstance(result.message_path[1], ArrayAccess)
        assert result.message_path[1].index == 0

    def test_negative_array_index(self) -> None:
        """Test parsing negative array index."""
        result = parse_message_path("/topic.array[-1]")
        assert result.topic_name == "/topic"
        assert len(result.message_path) == 2
        assert isinstance(result.message_path[1], ArrayAccess)
        assert result.message_path[1].index == -1

    def test_complex_path(self) -> None:
        """Test parsing complex message path."""
        result = parse_message_path("/some0/nice_topic.with[99].stuff[0]")
        assert result.topic_name == "/some0/nice_topic"
        assert len(result.message_path) == 4

        # Check each component
        assert isinstance(result.message_path[0], FieldAccess)
        assert result.message_path[0].field == "with"

        assert isinstance(result.message_path[1], ArrayAccess)
        assert result.message_path[1].index == 99

        assert isinstance(result.message_path[2], FieldAccess)
        assert result.message_path[2].field == "stuff"

        assert isinstance(result.message_path[3], ArrayAccess)
        assert result.message_path[3].index == 0


class TestSliceParsing:
    """Test slice notation parsing."""

    def test_simple_slice(self) -> None:
        """Test parsing simple slice notation."""
        result = parse_message_path("/topic.array[1:5]")
        assert result.topic_name == "/topic"
        assert len(result.message_path) == 2
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, SliceNotation)
        slice_obj = result.message_path[1].index
        assert slice_obj.start == 1
        assert slice_obj.end == 5

    def test_open_ended_slice(self) -> None:
        """Test parsing open-ended slice notation."""
        result = parse_message_path("/topic.array[1:]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, SliceNotation)
        slice_obj = result.message_path[1].index
        assert slice_obj.start == 1
        assert slice_obj.end is None

    def test_start_open_slice(self) -> None:
        """Test parsing slice with open start."""
        result = parse_message_path("/topic.array[:5]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, SliceNotation)
        slice_obj = result.message_path[1].index
        assert slice_obj.start is None
        assert slice_obj.end == 5

    def test_full_slice(self) -> None:
        """Test parsing full slice notation."""
        result = parse_message_path("/topic.array[:]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, SliceNotation)
        slice_obj = result.message_path[1].index
        assert slice_obj.start is None
        assert slice_obj.end is None

    def test_variable_slice(self) -> None:
        """Test parsing slice with variables."""
        result = parse_message_path("/topic.array[$start:$end]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, SliceNotation)
        slice_obj = result.message_path[1].index
        assert isinstance(slice_obj.start, Variable)
        assert slice_obj.start.name == "start"
        assert isinstance(slice_obj.end, Variable)
        assert slice_obj.end.name == "end"


class TestFilterParsing:
    """Test filter parsing functionality."""

    def test_simple_filter(self) -> None:
        """Test parsing simple filter."""
        result = parse_message_path("/topic{id==5}")
        assert result.topic_name == "/topic"
        assert len(result.message_path) == 1
        assert isinstance(result.message_path[0], Filter)
        filter_obj = result.message_path[0]
        assert filter_obj.field == "id"
        assert filter_obj.operator == "=="
        assert filter_obj.value == 5

    def test_string_filter(self) -> None:
        """Test parsing filter with string value."""
        result = parse_message_path('/topic{type=="arrow"}')
        assert isinstance(result.message_path[0], Filter)
        filter_obj = result.message_path[0]
        assert filter_obj.field == "type"
        assert filter_obj.operator == "=="
        assert filter_obj.value == "arrow"

    def test_comparison_operators(self) -> None:
        """Test different comparison operators."""
        test_cases = [
            ("/topic{value>=10}", ">=", 10),
            ("/topic{value<=10}", "<=", 10),
            ("/topic{value>10}", ">", 10),
            ("/topic{value<10}", "<", 10),
            ("/topic{value!=10}", "!=", 10),
        ]

        for path, expected_op, expected_val in test_cases:
            result = parse_message_path(path)
            assert isinstance(result.message_path[0], Filter)
            filter_obj = result.message_path[0]
            assert filter_obj.operator == expected_op
            assert filter_obj.value == expected_val

    def test_boolean_filter(self) -> None:
        """Test parsing filter with boolean values."""
        result = parse_message_path("/topic{active==true}")
        assert isinstance(result.message_path[0], Filter)
        filter_obj = result.message_path[0]
        assert filter_obj.value is True

        result = parse_message_path("/topic{active==false}")
        assert isinstance(result.message_path[0], Filter)
        filter_obj = result.message_path[0]
        assert filter_obj.value is False

    def test_nested_field_filter(self) -> None:
        """Test parsing filter on nested field."""
        result = parse_message_path("/topic.items[:]{pose.position.x>1.0}")
        assert result.topic_name == "/topic"
        assert len(result.message_path) == 3

        # Check slice access
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, SliceNotation)

        # Check filter
        assert isinstance(result.message_path[2], Filter)
        filter_obj = result.message_path[2]
        assert filter_obj.field == "pose.position.x"
        assert filter_obj.operator == ">"
        assert filter_obj.value == 1.0


class TestVariableParsing:
    """Test variable parsing functionality."""

    def test_simple_variable(self) -> None:
        """Test parsing simple variable."""
        result = parse_message_path("/topic.array[$index]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, Variable)
        var = result.message_path[1].index
        assert var.name == "index"
        assert var.repr == "$index"

    def test_empty_variable(self) -> None:
        """Test parsing empty variable."""
        result = parse_message_path("/topic.array[$]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, Variable)
        var = result.message_path[1].index
        assert var.name == ""
        assert var.repr == "$"

    def test_variable_in_filter(self) -> None:
        """Test parsing variable in filter."""
        result = parse_message_path("/topic{id==$var}")
        assert isinstance(result.message_path[0], Filter)
        filter_obj = result.message_path[0]
        assert isinstance(filter_obj.value, Variable)
        assert filter_obj.value.name == "var"


class TestQuotedNames:
    """Test parsing quoted topic and field names."""

    def test_quoted_topic_name(self) -> None:
        """Test parsing quoted topic name."""
        result = parse_message_path('"/foo/bar".baz')
        assert result.topic_name == "/foo/bar"
        assert isinstance(result.message_path[0], FieldAccess)
        assert result.message_path[0].field == "baz"

    def test_single_quoted_topic(self) -> None:
        """Test parsing single-quoted topic name."""
        result = parse_message_path("'/topic with spaces'.field")
        assert result.topic_name == "/topic with spaces"
        assert isinstance(result.message_path[0], FieldAccess)
        assert result.message_path[0].field == "field"


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_syntax(self) -> None:
        """Test that invalid syntax raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse message path"):
            parse_message_path("/topic.field[")

    def test_empty_input(self) -> None:
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse message path"):
            parse_message_path("")

    def test_invalid_filter(self) -> None:
        """Test that invalid filter syntax raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse message path"):
            parse_message_path("/topic{field=}")


class TestReprGeneration:
    """Test that repr fields are generated correctly."""

    def test_field_access_repr(self) -> None:
        """Test field access repr generation."""
        result = parse_message_path("/topic.field")
        assert isinstance(result.message_path[0], FieldAccess)
        assert result.message_path[0].repr == ".field"

    def test_array_access_repr(self) -> None:
        """Test array access repr generation."""
        result = parse_message_path("/topic.array[5]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert result.message_path[1].repr == "[5]"

    def test_slice_repr(self) -> None:
        """Test slice notation repr generation."""
        result = parse_message_path("/topic.array[1:5]")
        assert isinstance(result.message_path[1], ArrayAccess)
        assert isinstance(result.message_path[1].index, SliceNotation)
        assert result.message_path[1].index.repr == "1:5"

    def test_filter_repr(self) -> None:
        """Test filter repr generation."""
        result = parse_message_path("/topic{id==5}")
        assert isinstance(result.message_path[0], Filter)
        assert result.message_path[0].repr == "{id==5}"
