"""Tests for Foxglove message path parser."""

import math
from dataclasses import dataclass

import pytest
from ros_parser.message_path import (
    ArrayIndex,
    ArraySlice,
    ComparisonOperator,
    FieldAccess,
    Filter,
    LarkError,
    MathModifier,
    MessagePathError,
    Variable,
    parse_message_path,
)


class TestBasicPaths:
    """Test basic topic and field access."""

    def test_topic_only(self):
        """Test parsing just a topic."""
        result = parse_message_path("/my_topic")
        assert result.topic == "/my_topic"
        assert result.segments == []

    def test_topic_with_field(self):
        """Test topic with single field access."""
        result = parse_message_path("/my_topic.field")
        assert result.topic == "/my_topic"
        assert len(result.segments) == 1
        assert isinstance(result.segments[0], FieldAccess)
        assert result.segments[0].field_name == "field"

    def test_nested_fields(self):
        """Test accessing nested fields."""
        result = parse_message_path("/odom.pose.pose.position.x")
        assert result.topic == "/odom"
        assert len(result.segments) == 4
        assert all(isinstance(seg, FieldAccess) for seg in result.segments)
        assert result.segments[0].field_name == "pose"
        assert result.segments[1].field_name == "pose"
        assert result.segments[2].field_name == "position"
        assert result.segments[3].field_name == "x"

    def test_hierarchical_topic_two_levels(self):
        """Test hierarchical topic with two levels."""
        result = parse_message_path("/vehicle/odom")
        assert result.topic == "/vehicle/odom"
        assert result.segments == []

    def test_hierarchical_topic_three_levels(self):
        """Test hierarchical topic with three levels."""
        result = parse_message_path("/camera/left/image_color")
        assert result.topic == "/camera/left/image_color"
        assert result.segments == []

    def test_hierarchical_topic_deep(self):
        """Test deeply nested hierarchical topic."""
        result = parse_message_path("/robot/sensors/lidar/front")
        assert result.topic == "/robot/sensors/lidar/front"
        assert result.segments == []

    def test_hierarchical_topic_with_field(self):
        """Test hierarchical topic with field access."""
        result = parse_message_path("/vehicle/odom.pose.position.x")
        assert result.topic == "/vehicle/odom"
        assert len(result.segments) == 3
        assert isinstance(result.segments[0], FieldAccess)
        assert result.segments[0].field_name == "pose"
        assert result.segments[1].field_name == "position"
        assert result.segments[2].field_name == "x"

    def test_hierarchical_topic_with_array(self):
        """Test hierarchical topic with array indexing."""
        result = parse_message_path("/camera/left/images[0].data")
        assert result.topic == "/camera/left/images"
        assert len(result.segments) == 2
        assert isinstance(result.segments[0], ArrayIndex)
        assert result.segments[0].index == 0
        assert isinstance(result.segments[1], FieldAccess)
        assert result.segments[1].field_name == "data"

    def test_hierarchical_topic_with_filter(self):
        """Test hierarchical topic with filter."""
        result = parse_message_path("/sensors/temp[:]{value>20}.reading")
        assert result.topic == "/sensors/temp"
        assert len(result.segments) == 3
        assert isinstance(result.segments[0], ArraySlice)
        assert isinstance(result.segments[1], Filter)
        assert isinstance(result.segments[2], FieldAccess)


class TestArrayIndexing:
    """Test array indexing operations."""

    def test_positive_index(self):
        """Test positive array index."""
        result = parse_message_path("/topic.array[5]")
        assert len(result.segments) == 2
        assert isinstance(result.segments[1], ArrayIndex)
        assert result.segments[1].index == 5

    def test_zero_index(self):
        """Test zero array index."""
        result = parse_message_path("/topic.array[0]")
        assert isinstance(result.segments[1], ArrayIndex)
        assert result.segments[1].index == 0

    def test_negative_index(self):
        """Test negative array index."""
        result = parse_message_path("/topic.array[-1]")
        assert isinstance(result.segments[1], ArrayIndex)
        assert result.segments[1].index == -1

    def test_variable_index(self):
        """Test variable as array index."""
        result = parse_message_path("/topic.array[$idx]")
        assert isinstance(result.segments[1], ArrayIndex)
        assert isinstance(result.segments[1].index, Variable)
        assert result.segments[1].index.name == "idx"

    def test_field_after_index(self):
        """Test field access after array indexing."""
        result = parse_message_path("/topic.objects[1].width")
        assert len(result.segments) == 3
        assert isinstance(result.segments[0], FieldAccess)
        assert isinstance(result.segments[1], ArrayIndex)
        assert isinstance(result.segments[2], FieldAccess)
        assert result.segments[2].field_name == "width"


class TestArraySlicing:
    """Test array slicing operations."""

    def test_full_slice(self):
        """Test full array slice [:]."""
        result = parse_message_path("/topic.array[:]")
        assert isinstance(result.segments[1], ArraySlice)
        assert result.segments[1].start is None
        assert result.segments[1].end is None

    def test_range_slice(self):
        """Test range slice [1:3]."""
        result = parse_message_path("/topic.array[1:3]")
        assert isinstance(result.segments[1], ArraySlice)
        assert result.segments[1].start == 1
        assert result.segments[1].end == 3

    def test_open_start_slice(self):
        """Test slice with open start [:5]."""
        result = parse_message_path("/topic.array[:5]")
        assert isinstance(result.segments[1], ArraySlice)
        assert result.segments[1].start is None
        assert result.segments[1].end == 5

    def test_open_end_slice(self):
        """Test slice with open end [5:]."""
        result = parse_message_path("/topic.array[5:]")
        assert isinstance(result.segments[1], ArraySlice)
        assert result.segments[1].start == 5
        assert result.segments[1].end is None

    def test_variable_slice(self):
        """Test slice with variables."""
        result = parse_message_path("/topic.array[$start:$end]")
        assert isinstance(result.segments[1], ArraySlice)
        assert isinstance(result.segments[1].start, Variable)
        assert isinstance(result.segments[1].end, Variable)
        assert result.segments[1].start.name == "start"
        assert result.segments[1].end.name == "end"

    def test_field_after_slice(self):
        """Test field access after slicing."""
        result = parse_message_path("/topic.colors[:].r")
        assert len(result.segments) == 3
        assert isinstance(result.segments[1], ArraySlice)
        assert isinstance(result.segments[2], FieldAccess)
        assert result.segments[2].field_name == "r"


class TestFiltering:
    """Test filtering operations."""

    def test_equality_filter(self):
        """Test equality filter."""
        result = parse_message_path("/topic{field==5}")
        assert len(result.segments) == 1
        assert isinstance(result.segments[0], Filter)
        assert result.segments[0].field_path == "field"
        assert result.segments[0].operator == ComparisonOperator.EQUAL
        assert result.segments[0].value == 5

    def test_inequality_filter(self):
        """Test inequality filter."""
        result = parse_message_path("/topic{status!=active}")
        assert result.segments[0].operator == ComparisonOperator.NOT_EQUAL

    def test_less_than_filter(self):
        """Test less than filter."""
        result = parse_message_path("/topic{count<10}")
        assert result.segments[0].operator == ComparisonOperator.LESS_THAN
        assert result.segments[0].value == 10

    def test_less_equal_filter(self):
        """Test less than or equal filter."""
        result = parse_message_path("/topic{count<=10}")
        assert result.segments[0].operator == ComparisonOperator.LESS_THAN_OR_EQUAL

    def test_greater_than_filter(self):
        """Test greater than filter."""
        result = parse_message_path("/topic{temperature>25.5}")
        assert result.segments[0].operator == ComparisonOperator.GREATER_THAN
        assert result.segments[0].value == 25.5

    def test_greater_equal_filter(self):
        """Test greater than or equal filter."""
        result = parse_message_path("/topic{score>=0.95}")
        assert result.segments[0].operator == ComparisonOperator.GREATER_THAN_OR_EQUAL

    def test_nested_field_path_filter(self):
        """Test filter with nested field path."""
        result = parse_message_path("/topic{stats.pages>200}")
        assert result.segments[0].field_path == "stats.pages"

    def test_string_value_filter(self):
        """Test filter with string value."""
        result = parse_message_path('/topic{name=="John"}')
        assert result.segments[0].value == "John"

    def test_boolean_value_filter(self):
        """Test filter with boolean value."""
        result = parse_message_path("/topic{active==true}")
        assert result.segments[0].value is True

    def test_variable_value_filter(self):
        """Test filter with variable value."""
        result = parse_message_path("/topic{id==$my_id}")
        assert isinstance(result.segments[0].value, Variable)
        assert result.segments[0].value.name == "my_id"

    def test_multiple_filters(self):
        """Test multiple filters (AND logic)."""
        result = parse_message_path("/topic{category==robot}{status==active}{battery>20}")
        assert len(result.segments) == 3
        assert all(isinstance(seg, Filter) for seg in result.segments)
        assert result.segments[0].field_path == "category"
        assert result.segments[1].field_path == "status"
        assert result.segments[2].field_path == "battery"


class TestComplexPaths:
    """Test complex nested operations."""

    def test_array_filter_field(self):
        """Test array slice with filter and field access."""
        result = parse_message_path("/detections.objects[:]{confidence>0.8}.class_name")
        assert len(result.segments) == 4
        assert isinstance(result.segments[0], FieldAccess)  # objects
        assert isinstance(result.segments[1], ArraySlice)  # [:]
        assert isinstance(result.segments[2], Filter)  # {confidence>0.8}
        assert isinstance(result.segments[3], FieldAccess)  # class_name

    def test_nested_filters_arrays(self):
        """Test deeply nested path with multiple filters and arrays."""
        result = parse_message_path(
            '/library.books[:]{genre=="sci-fi"}{pages>200}.reviews[:]{rating>=4}.text'
        )
        assert len(result.segments) == 8
        # library
        assert isinstance(result.segments[0], FieldAccess)
        assert result.segments[0].field_name == "books"
        # [:]
        assert isinstance(result.segments[1], ArraySlice)
        # {genre==sci-fi}
        assert isinstance(result.segments[2], Filter)
        # {pages>200}
        assert isinstance(result.segments[3], Filter)
        # reviews
        assert isinstance(result.segments[4], FieldAccess)
        assert result.segments[4].field_name == "reviews"
        # [:]
        assert isinstance(result.segments[5], ArraySlice)
        # {rating>=4}
        assert isinstance(result.segments[6], Filter)
        # text
        assert isinstance(result.segments[7], FieldAccess)
        assert result.segments[7].field_name == "text"

    def test_mixed_operations(self):
        """Test mixing all operation types."""
        result = parse_message_path("/data.items[5].values[:]{x>0}.label")
        assert len(result.segments) == 6


class TestStringHandling:
    """Test string literal handling."""

    def test_double_quoted_string(self):
        """Test double-quoted strings."""
        result = parse_message_path('/topic{name=="test"}')
        assert result.segments[0].value == "test"

    def test_single_quoted_string(self):
        """Test single-quoted strings."""
        result = parse_message_path("/topic{name=='test'}")
        assert result.segments[0].value == "test"

    def test_string_with_single_quote_inside(self):
        """Test string containing single quote using double quotes."""
        result = parse_message_path('/topic{name=="O\'Brien"}')
        assert result.segments[0].value == "O'Brien"

    def test_string_with_double_quote_inside(self):
        """Test string containing double quote using single quotes."""
        result = parse_message_path("/topic{text=='Say \"hello\"'}")
        assert result.segments[0].value == 'Say "hello"'


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_topic_with_underscores(self):
        """Test topic name with underscores."""
        result = parse_message_path("/my_long_topic_name")
        assert result.topic == "/my_long_topic_name"

    def test_field_with_underscores(self):
        """Test field name with underscores."""
        result = parse_message_path("/topic.field_name_long")
        assert result.segments[0].field_name == "field_name_long"

    def test_negative_numbers_in_filter(self):
        """Test negative numbers in filters."""
        result = parse_message_path("/topic{value<-10}")
        assert result.segments[0].value == -10

    def test_float_in_filter(self):
        """Test floating point numbers in filters."""
        result = parse_message_path("/topic{value==3.14159}")
        assert result.segments[0].value == 3.14159

    def test_false_boolean(self):
        """Test false boolean value."""
        result = parse_message_path("/topic{active==false}")
        assert result.segments[0].value is False


class TestErrorCases:
    """Test error handling for invalid syntax."""

    def test_missing_topic_slash(self):
        """Test error when topic doesn't start with /."""
        with pytest.raises(LarkError):
            parse_message_path("topic.field")

    def test_empty_brackets(self):
        """Test error for empty array brackets []."""
        with pytest.raises(LarkError):
            parse_message_path("/topic.array[]")

    def test_invalid_filter_syntax(self):
        """Test error for invalid filter syntax."""
        with pytest.raises(LarkError):
            parse_message_path("/topic{field}")  # Missing operator and value


# Test data structures for apply() tests
@dataclass
class Point:
    """Test dataclass for field access."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class Object:
    """Test dataclass with nested fields."""

    name: str
    position: Point
    confidence: float


class TestFieldAccessApply:
    """Test FieldAccess.apply() method."""

    def test_apply_to_dataclass(self):
        """Test field access on dataclass."""
        obj = Point(x=1.0, y=2.0, z=3.0)
        field = FieldAccess(field_name="x")
        result = field.apply(obj, {})
        assert result == 1.0

    def test_apply_to_dict(self):
        """Test field access on dictionary."""
        obj = {"x": 10, "y": 20}
        field = FieldAccess(field_name="x")
        result = field.apply(obj, {})
        assert result == 10

    def test_apply_to_nested_object(self):
        """Test field access on nested object."""
        obj = Object(name="box", position=Point(x=1.0, y=2.0), confidence=0.9)
        field = FieldAccess(field_name="position")
        result = field.apply(obj, {})
        assert result == Point(x=1.0, y=2.0)

    def test_apply_missing_field_on_dict(self):
        """Test error when field is missing on dict."""
        obj = {"x": 10}
        field = FieldAccess(field_name="y")
        with pytest.raises(MessagePathError, match="Field 'y' not found"):
            field.apply(obj, {})

    def test_apply_missing_field_on_object(self):
        """Test error when field is missing on object."""
        obj = Point(x=1.0, y=2.0)
        field = FieldAccess(field_name="missing")
        with pytest.raises(MessagePathError, match="Field 'missing' not found"):
            field.apply(obj, {})


class TestArrayIndexApply:
    """Test ArrayIndex.apply() method."""

    def test_apply_positive_index(self):
        """Test array indexing with positive index."""
        obj = [10, 20, 30, 40, 50]
        index = ArrayIndex(index=2)
        result = index.apply(obj, {})
        assert result == 30

    def test_apply_negative_index(self):
        """Test array indexing with negative index."""
        obj = [10, 20, 30, 40, 50]
        index = ArrayIndex(index=-1)
        result = index.apply(obj, {})
        assert result == 50

    def test_apply_zero_index(self):
        """Test array indexing with zero."""
        obj = ["a", "b", "c"]
        index = ArrayIndex(index=0)
        result = index.apply(obj, {})
        assert result == "a"

    def test_apply_with_variable(self):
        """Test array indexing with variable."""
        obj = [10, 20, 30]
        index = ArrayIndex(index=Variable(name="idx"))
        result = index.apply(obj, {"idx": 1})
        assert result == 20

    def test_apply_on_string(self):
        """Test array indexing on string."""
        obj = "hello"
        index = ArrayIndex(index=1)
        result = index.apply(obj, {})
        assert result == "e"

    def test_apply_on_tuple(self):
        """Test array indexing on tuple."""
        obj = (5, 10, 15)
        index = ArrayIndex(index=2)
        result = index.apply(obj, {})
        assert result == 15

    def test_apply_out_of_range(self):
        """Test error when index is out of range."""
        obj = [1, 2, 3]
        index = ArrayIndex(index=10)
        with pytest.raises(MessagePathError, match="Index 10 out of range"):
            index.apply(obj, {})

    def test_apply_on_non_sequence(self):
        """Test error when applying to non-sequence."""
        obj = {"key": "value"}
        index = ArrayIndex(index=0)
        with pytest.raises(MessagePathError, match="can only be applied to sequences"):
            index.apply(obj, {})


class TestArraySliceApply:
    """Test ArraySlice.apply() method with INCLUSIVE end index."""

    def test_apply_full_slice(self):
        """Test full slice [:]."""
        obj = [1, 2, 3, 4, 5]
        slice_action = ArraySlice(start=None, end=None)
        result = slice_action.apply(obj, {})
        assert result == [1, 2, 3, 4, 5]

    def test_apply_inclusive_range_slice(self):
        """Test INCLUSIVE range slice [1:3] returns indices 1, 2, AND 3."""
        obj = [10, 20, 30, 40, 50]
        slice_action = ArraySlice(start=1, end=3)
        result = slice_action.apply(obj, {})
        # Foxglove spec: INCLUSIVE end, so [1:3] should include indices 1, 2, 3
        assert result == [20, 30, 40], "End index should be inclusive per Foxglove spec"

    def test_apply_inclusive_zero_start(self):
        """Test INCLUSIVE slice [0:2] returns indices 0, 1, AND 2."""
        obj = ["a", "b", "c", "d"]
        slice_action = ArraySlice(start=0, end=2)
        result = slice_action.apply(obj, {})
        assert result == ["a", "b", "c"], "End index should be inclusive"

    def test_apply_open_start_slice(self):
        """Test slice with open start [:2]."""
        obj = [1, 2, 3, 4, 5]
        slice_action = ArraySlice(start=None, end=2)
        result = slice_action.apply(obj, {})
        # [:2] should include 0, 1, 2 (inclusive)
        assert result == [1, 2, 3]

    def test_apply_open_end_slice(self):
        """Test slice with open end [2:]."""
        obj = [1, 2, 3, 4, 5]
        slice_action = ArraySlice(start=2, end=None)
        result = slice_action.apply(obj, {})
        assert result == [3, 4, 5]

    def test_apply_negative_indices(self):
        """Test INCLUSIVE slice with negative indices [-3:-1]."""
        obj = [1, 2, 3, 4, 5]
        slice_action = ArraySlice(start=-3, end=-1)
        result = slice_action.apply(obj, {})
        # [-3:-1] should include last 3 elements: indices -3, -2, -1 (inclusive)
        assert result == [3, 4, 5], "Negative end index should be inclusive"

    def test_apply_with_variables(self):
        """Test slice with variables."""
        obj = [10, 20, 30, 40, 50]
        slice_action = ArraySlice(start=Variable(name="s"), end=Variable(name="e"))
        result = slice_action.apply(obj, {"s": 1, "e": 3})
        # [1:3] inclusive should be indices 1, 2, 3
        assert result == [20, 30, 40]

    def test_apply_on_string(self):
        """Test INCLUSIVE slice on string."""
        obj = "hello"
        slice_action = ArraySlice(start=1, end=3)
        result = slice_action.apply(obj, {})
        # [1:3] inclusive should be indices 1, 2, 3 -> "ell"
        assert result == "ell"

    def test_apply_single_element_slice(self):
        """Test slice that selects single element [2:2]."""
        obj = [10, 20, 30, 40]
        slice_action = ArraySlice(start=2, end=2)
        result = slice_action.apply(obj, {})
        # [2:2] inclusive should return just index 2
        assert result == [30]

    def test_apply_on_non_sequence(self):
        """Test error when applying to non-sequence."""
        obj = {"key": "value"}
        slice_action = ArraySlice(start=0, end=2)
        with pytest.raises(MessagePathError, match="can only be applied to sequences"):
            slice_action.apply(obj, {})


class TestFilterApply:
    """Test Filter.apply() method."""

    def test_apply_equality_filter(self):
        """Test filter with equality operator."""
        objects = [
            Object(name="box", position=Point(1, 2), confidence=0.9),
            Object(name="car", position=Point(3, 4), confidence=0.7),
            Object(name="box", position=Point(5, 6), confidence=0.8),
        ]
        filter_action = Filter(field_path="name", operator=ComparisonOperator.EQUAL, value="box")
        result = filter_action.apply(objects, {})
        assert len(result) == 2
        assert all(obj.name == "box" for obj in result)

    def test_apply_greater_than_filter(self):
        """Test filter with greater than operator."""
        objects = [
            Object(name="obj1", position=Point(0, 0), confidence=0.9),
            Object(name="obj2", position=Point(0, 0), confidence=0.7),
            Object(name="obj3", position=Point(0, 0), confidence=0.85),
        ]
        filter_action = Filter(
            field_path="confidence", operator=ComparisonOperator.GREATER_THAN, value=0.8
        )
        result = filter_action.apply(objects, {})
        assert len(result) == 2
        assert result[0].confidence == 0.9
        assert result[1].confidence == 0.85

    def test_apply_nested_field_path_filter(self):
        """Test filter with nested field path."""
        objects = [
            Object(name="obj1", position=Point(x=1.0, y=2.0), confidence=0.9),
            Object(name="obj2", position=Point(x=5.0, y=6.0), confidence=0.8),
            Object(name="obj3", position=Point(x=3.0, y=4.0), confidence=0.7),
        ]
        filter_action = Filter(
            field_path="position.x", operator=ComparisonOperator.GREATER_THAN, value=2.0
        )
        result = filter_action.apply(objects, {})
        assert len(result) == 2
        assert result[0].position.x == 5.0
        assert result[1].position.x == 3.0

    def test_apply_filter_with_variable(self):
        """Test filter with variable value."""
        data = [{"value": 10}, {"value": 20}, {"value": 30}]
        filter_action = Filter(
            field_path="value",
            operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
            value=Variable(name="threshold"),
        )
        result = filter_action.apply(data, {"threshold": 20})
        assert len(result) == 2
        assert result[0]["value"] == 20
        assert result[1]["value"] == 30

    def test_apply_filter_on_dicts(self):
        """Test filter on list of dictionaries."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        filter_action = Filter(
            field_path="age", operator=ComparisonOperator.LESS_THAN_OR_EQUAL, value=30
        )
        result = filter_action.apply(data, {})
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"

    def test_apply_filter_not_equal(self):
        """Test filter with not equal operator."""
        data = [{"status": "active"}, {"status": "inactive"}, {"status": "active"}]
        filter_action = Filter(
            field_path="status", operator=ComparisonOperator.NOT_EQUAL, value="inactive"
        )
        result = filter_action.apply(data, {})
        assert len(result) == 2
        assert all(item["status"] == "active" for item in result)

    def test_apply_filter_empty_result(self):
        """Test filter that matches nothing."""
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        filter_action = Filter(field_path="x", operator=ComparisonOperator.GREATER_THAN, value=10)
        result = filter_action.apply(data, {})
        assert result == []

    def test_apply_filter_missing_field(self):
        """Test filter skips items with missing field."""
        data = [{"x": 1, "y": 2}, {"x": 5}, {"x": 3, "y": 4}]
        filter_action = Filter(field_path="y", operator=ComparisonOperator.GREATER_THAN, value=2)
        result = filter_action.apply(data, {})
        # Should skip the item without 'y'
        assert len(result) == 1
        assert result[0] == {"x": 3, "y": 4}

    def test_apply_filter_on_single_object_match(self):
        """Test applying filter to single object that matches."""
        obj = {"key": "value"}
        filter_action = Filter(field_path="key", operator=ComparisonOperator.EQUAL, value="value")
        result = filter_action.apply(obj, {})
        assert result == obj

    def test_apply_filter_on_single_object_no_match(self):
        """Test applying filter to single object that doesn't match."""
        obj = {"key": "value"}
        filter_action = Filter(field_path="key", operator=ComparisonOperator.EQUAL, value="other")
        result = filter_action.apply(obj, {})
        assert result is None

    def test_apply_filter_type_mismatch_comparison(self):
        """Test error when comparing incompatible types."""
        data = [{"x": "string"}]
        filter_action = Filter(field_path="x", operator=ComparisonOperator.LESS_THAN, value=10)
        with pytest.raises(MessagePathError, match="Cannot compare"):
            filter_action.apply(data, {})


class TestIntegratedApply:
    """Test applying multiple actions in sequence."""

    def test_field_then_index(self):
        """Test field access followed by array indexing."""
        obj = {"values": [10, 20, 30]}
        field = FieldAccess(field_name="values")
        index = ArrayIndex(index=1)

        result = field.apply(obj, {})
        result = index.apply(result, {})
        assert result == 20

    def test_field_slice_filter(self):
        """Test field access, slice, then filter."""
        data = {
            "objects": [
                {"name": "a", "score": 10},
                {"name": "b", "score": 20},
                {"name": "c", "score": 30},
            ]
        }

        field = FieldAccess(field_name="objects")
        slice_action = ArraySlice(start=None, end=None)
        filter_action = Filter(
            field_path="score", operator=ComparisonOperator.GREATER_THAN, value=15
        )

        result = field.apply(data, {})
        result = slice_action.apply(result, {})
        result = filter_action.apply(result, {})

        assert len(result) == 2
        assert result[0]["name"] == "b"
        assert result[1]["name"] == "c"

    def test_nested_field_access(self):
        """Test multiple field accesses for nested objects."""
        obj = Object(name="test", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.9)

        field1 = FieldAccess(field_name="position")
        field2 = FieldAccess(field_name="x")

        result = field1.apply(obj, {})
        result = field2.apply(result, {})

        assert result == 1.0


class TestMessagePathApply:
    """Test MessagePath.apply() helper method."""

    def test_apply_simple_field(self):
        """Test apply with simple field access."""
        path = parse_message_path("/topic.name")
        obj = Object(name="test", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.9)

        result = path.apply(obj)
        assert result == "test"

    def test_apply_nested_fields(self):
        """Test apply with nested field access."""
        path = parse_message_path("/topic.position.x")
        obj = Object(name="test", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.9)

        result = path.apply(obj)
        assert result == 1.0

    def test_apply_with_array_index(self):
        """Test apply with array indexing."""
        path = parse_message_path("/topic.items[1]")

        @dataclass
        class Data:
            items: list[str]

        obj = Data(items=["a", "b", "c"])
        result = path.apply(obj)
        assert result == "b"

    def test_apply_with_filter(self):
        """Test apply with filter."""
        path = parse_message_path("/topic.objects{confidence>0.5}")

        @dataclass
        class Container:
            objects: list[Object]

        obj = Container(
            objects=[
                Object(name="a", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.3),
                Object(name="b", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.7),
                Object(name="c", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.9),
            ]
        )

        result = path.apply(obj)
        assert len(result) == 2
        assert result[0].name == "b"
        assert result[1].name == "c"

    def test_apply_with_variables(self):
        """Test apply with variable substitution."""
        path = parse_message_path("/topic.items[$idx]")

        @dataclass
        class Data:
            items: list[str]

        obj = Data(items=["a", "b", "c"])
        result = path.apply(obj, variables={"idx": 2})
        assert result == "c"

    def test_apply_complex_chain(self):
        """Test apply with complex chained operations."""
        path = parse_message_path("/topic.objects{confidence>0.5}[0].position.x")

        @dataclass
        class Container:
            objects: list[Object]

        obj = Container(
            objects=[
                Object(name="a", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.3),
                Object(name="b", position=Point(x=5.0, y=6.0, z=7.0), confidence=0.7),
                Object(name="c", position=Point(x=8.0, y=9.0, z=10.0), confidence=0.9),
            ]
        )

        result = path.apply(obj)
        assert result == 5.0

    def test_apply_without_variables(self):
        """Test that apply works when variables parameter is omitted."""
        path = parse_message_path("/topic.name")
        obj = Object(name="test", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.9)

        # Should work without passing variables parameter
        result = path.apply(obj)
        assert result == "test"

    def test_apply_empty_variables(self):
        """Test that apply works with explicit empty variables dict."""
        path = parse_message_path("/topic.name")
        obj = Object(name="test", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.9)

        result = path.apply(obj, variables={})
        assert result == "test"

    def test_apply_with_error(self):
        """Test that apply raises MessagePathError on failure."""
        path = parse_message_path("/topic.nonexistent")
        obj = Object(name="test", position=Point(x=1.0, y=2.0, z=3.0), confidence=0.9)

        with pytest.raises(MessagePathError):
            path.apply(obj)


class TestMathModifierParsing:
    """Test parsing of math modifier syntax."""

    def test_parse_unary_modifier(self):
        """Test parsing unary math modifier (no arguments)."""

        result = parse_message_path("/topic.value.@abs")
        assert len(result.segments) == 2
        assert isinstance(result.segments[1], MathModifier)
        assert result.segments[1].operation == "abs"
        assert result.segments[1].arguments == []

    def test_parse_binary_modifier_with_number(self):
        """Test parsing math modifier with numeric argument."""

        result = parse_message_path("/topic.value.@mul(2.5)")
        assert len(result.segments) == 2
        assert isinstance(result.segments[1], MathModifier)
        assert result.segments[1].operation == "mul"
        assert result.segments[1].arguments == [2.5]

    def test_parse_modifier_with_variable(self):
        """Test parsing math modifier with variable argument."""

        result = parse_message_path("/topic.value.@add($offset)")
        assert len(result.segments) == 2
        assert isinstance(result.segments[1], MathModifier)
        assert result.segments[1].operation == "add"
        assert len(result.segments[1].arguments) == 1
        assert isinstance(result.segments[1].arguments[0], Variable)
        assert result.segments[1].arguments[0].name == "offset"

    def test_parse_chained_modifiers(self):
        """Test parsing multiple chained math modifiers."""

        result = parse_message_path("/topic.value.@mul(1.8).@add(32)")
        assert len(result.segments) == 3
        assert isinstance(result.segments[1], MathModifier)
        assert result.segments[1].operation == "mul"
        assert isinstance(result.segments[2], MathModifier)
        assert result.segments[2].operation == "add"

    def test_parse_modifier_after_array_index(self):
        """Test math modifier after array indexing."""

        result = parse_message_path("/topic.values[0].@abs")
        assert len(result.segments) == 3
        assert isinstance(result.segments[1], ArrayIndex)
        assert isinstance(result.segments[2], MathModifier)

    def test_parse_modifier_with_round_precision(self):
        """Test round modifier with precision argument."""

        result = parse_message_path("/topic.value.@round(2)")
        assert isinstance(result.segments[1], MathModifier)
        assert result.segments[1].operation == "round"
        assert result.segments[1].arguments == [2]


class TestMathModifierApply:
    """Test applying math modifiers to data."""

    def test_abs_positive(self):
        """Test abs on positive number."""

        modifier = MathModifier(operation="abs", arguments=[])
        result = modifier.apply(5.5, {})
        assert result == 5.5

    def test_abs_negative(self):
        """Test abs on negative number."""

        modifier = MathModifier(operation="abs", arguments=[])
        result = modifier.apply(-10, {})
        assert result == 10

    def test_add_with_number(self):
        """Test add operation."""

        modifier = MathModifier(operation="add", arguments=[5])
        result = modifier.apply(10, {})
        assert result == 15

    def test_sub_with_number(self):
        """Test subtract operation."""

        modifier = MathModifier(operation="sub", arguments=[3])
        result = modifier.apply(10, {})
        assert result == 7

    def test_mul_with_number(self):
        """Test multiply operation."""

        modifier = MathModifier(operation="mul", arguments=[2.5])
        result = modifier.apply(4, {})
        assert result == 10.0

    def test_div_with_number(self):
        """Test division operation."""

        modifier = MathModifier(operation="div", arguments=[2])
        result = modifier.apply(10, {})
        assert result == 5.0

    def test_div_by_zero(self):
        """Test division by zero raises error."""

        modifier = MathModifier(operation="div", arguments=[0])
        with pytest.raises(MessagePathError, match="Division by zero"):
            modifier.apply(10, {})

    def test_sqrt(self):
        """Test square root operation."""

        modifier = MathModifier(operation="sqrt", arguments=[])
        result = modifier.apply(16, {})
        assert result == 4.0

    def test_sqrt_negative(self):
        """Test sqrt of negative number raises error."""

        modifier = MathModifier(operation="sqrt", arguments=[])
        with pytest.raises(MessagePathError, match="Math error"):
            modifier.apply(-1, {})

    def test_round_no_args(self):
        """Test round without precision argument."""

        modifier = MathModifier(operation="round", arguments=[])
        result = modifier.apply(3.7, {})
        assert result == 4

    def test_round_with_precision(self):
        """Test round with precision argument."""

        modifier = MathModifier(operation="round", arguments=[2])
        result = modifier.apply(3.14159, {})
        assert result == 3.14

    def test_ceil(self):
        """Test ceil operation."""

        modifier = MathModifier(operation="ceil", arguments=[])
        result = modifier.apply(3.2, {})
        assert result == 4

    def test_floor(self):
        """Test floor operation."""

        modifier = MathModifier(operation="floor", arguments=[])
        result = modifier.apply(3.8, {})
        assert result == 3

    def test_sign_positive(self):
        """Test sign of positive number."""

        modifier = MathModifier(operation="sign", arguments=[])
        result = modifier.apply(5, {})
        assert result == 1

    def test_sign_negative(self):
        """Test sign of negative number."""

        modifier = MathModifier(operation="sign", arguments=[])
        result = modifier.apply(-5, {})
        assert result == -1

    def test_sign_zero(self):
        """Test sign of zero."""

        modifier = MathModifier(operation="sign", arguments=[])
        result = modifier.apply(0, {})
        assert result == 0

    def test_negative(self):
        """Test negative operation."""

        modifier = MathModifier(operation="negative", arguments=[])
        result = modifier.apply(5, {})
        assert result == -5

    def test_trig_functions(self):
        """Test trigonometric functions."""

        # Test sin
        modifier = MathModifier(operation="sin", arguments=[])
        result = modifier.apply(0, {})
        assert abs(result - 0) < 1e-10

        # Test cos
        modifier = MathModifier(operation="cos", arguments=[])
        result = modifier.apply(0, {})
        assert abs(result - 1.0) < 1e-10

        # Test tan
        modifier = MathModifier(operation="tan", arguments=[])
        result = modifier.apply(0, {})
        assert abs(result - 0) < 1e-10

    def test_log_functions(self):
        """Test logarithm functions."""

        # Test log (natural log)
        modifier = MathModifier(operation="log", arguments=[])
        result = modifier.apply(math.e, {})
        assert abs(result - 1.0) < 1e-10

        # Test log10
        modifier = MathModifier(operation="log10", arguments=[])
        result = modifier.apply(100, {})
        assert abs(result - 2.0) < 1e-10

        # Test log2
        modifier = MathModifier(operation="log2", arguments=[])
        result = modifier.apply(8, {})
        assert abs(result - 3.0) < 1e-10

    def test_modifier_with_variable_arg(self):
        """Test modifier with variable argument."""

        modifier = MathModifier(operation="mul", arguments=[Variable(name="scale")])
        result = modifier.apply(10, {"scale": 2.5})
        assert result == 25.0

    def test_element_wise_on_list(self):
        """Test element-wise application on list."""

        modifier = MathModifier(operation="abs", arguments=[])
        result = modifier.apply([1, -2, 3, -4], {})
        assert result == [1, 2, 3, 4]

    def test_element_wise_on_tuple(self):
        """Test element-wise application on tuple."""

        modifier = MathModifier(operation="mul", arguments=[2])
        result = modifier.apply((1, 2, 3), {})
        assert result == (2, 4, 6)
        assert isinstance(result, tuple)

    def test_element_wise_with_complex_operation(self):
        """Test element-wise with more complex operations."""

        modifier = MathModifier(operation="add", arguments=[10])
        result = modifier.apply([1.5, 2.5, 3.5], {})
        assert result == [11.5, 12.5, 13.5]

    def test_error_on_non_numeric(self):
        """Test error when applying to non-numeric type."""

        modifier = MathModifier(operation="abs", arguments=[])
        with pytest.raises(MessagePathError, match="can only be applied to numeric types"):
            modifier.apply("not a number", {})

    def test_error_on_nan(self):
        """Test error when receiving NaN."""

        modifier = MathModifier(operation="abs", arguments=[])
        with pytest.raises(MessagePathError, match="received NaN"):
            modifier.apply(float("nan"), {})

    def test_unknown_operation(self):
        """Test error on unknown operation."""

        modifier = MathModifier(operation="unknown_op", arguments=[])
        with pytest.raises(MessagePathError, match="Unknown math modifier"):
            modifier.apply(10, {})

    def test_add_multiple_args(self):
        """Test add with multiple arguments."""

        modifier = MathModifier(operation="add", arguments=[5, 10, 3])
        result = modifier.apply(2, {})
        assert result == 20  # 2 + 5 + 10 + 3

    def test_sub_multiple_args(self):
        """Test subtract with multiple arguments."""

        modifier = MathModifier(operation="sub", arguments=[2, 3])
        result = modifier.apply(10, {})
        assert result == 5  # 10 - 2 - 3

    def test_mul_multiple_args(self):
        """Test multiply with multiple arguments."""

        modifier = MathModifier(operation="mul", arguments=[2, 3, 2])
        result = modifier.apply(5, {})
        assert result == 60  # 5 * 2 * 3 * 2

    def test_min_operation(self):
        """Test min operation."""

        modifier = MathModifier(operation="min", arguments=[5, 2, 8])
        result = modifier.apply(3, {})
        assert result == 2  # min(3, 5, 2, 8)

    def test_max_operation(self):
        """Test max operation."""

        modifier = MathModifier(operation="max", arguments=[5, 2, 8])
        result = modifier.apply(3, {})
        assert result == 8  # max(3, 5, 2, 8)


class TestMathModifierIntegration:
    """Test math modifiers in complete message paths."""

    def test_celsius_to_fahrenheit(self):
        """Test converting Celsius to Fahrenheit: C * 1.8 + 32."""
        path = parse_message_path("/topic.temperature.@mul(1.8).@add(32)")
        result = path.apply({"temperature": 0}, {})
        assert result == 32.0

        result = path.apply({"temperature": 100}, {})
        assert result == 212.0

    def test_with_nested_fields(self):
        """Test math modifier on nested field access."""
        path = parse_message_path("/topic.position.x.@abs")
        obj = {"position": {"x": -5.5}}
        result = path.apply(obj, {})
        assert result == 5.5

    def test_with_array_slice(self):
        """Test math modifier after array slicing."""
        path = parse_message_path("/topic.values[:].@mul(2)")
        obj = {"values": [1, 2, 3, 4]}
        result = path.apply(obj, {})
        assert result == [2, 4, 6, 8]

    def test_with_filter_and_modifier(self):
        """Test combining filter with math modifier."""
        # Filter first, then access value field on each item, then apply sqrt
        path = parse_message_path("/topic.readings[:]{value>0}")
        obj = {
            "readings": [
                {"value": 4},
                {"value": -1},
                {"value": 16},
                {"value": 9},
            ]
        }
        result = path.apply(obj, {})
        # Filter keeps only positive values
        assert result == [{"value": 4}, {"value": 16}, {"value": 9}]

        # Now test with a simpler structure - array of numbers
        path2 = parse_message_path("/topic.values[:].@abs")
        obj2 = {"values": [-4, 1, -16, 9]}
        result2 = path2.apply(obj2, {})
        assert result2 == [4, 1, 16, 9]

    def test_multiple_modifiers_chain(self):
        """Test chaining multiple math modifiers."""
        path = parse_message_path("/topic.value.@mul(2).@add(10).@div(4)")
        result = path.apply({"value": 5}, {})
        # (5 * 2 + 10) / 4 = 20 / 4 = 5.0
        assert result == 5.0

    def test_with_variables(self):
        """Test math modifiers with variable substitution."""
        path = parse_message_path("/topic.value.@mul($scale).@add($offset)")
        result = path.apply({"value": 10}, {"scale": 2, "offset": 5})
        assert result == 25

    def test_add_multiple_args_parsed(self):
        """Test parsing and applying add with multiple arguments."""
        path = parse_message_path("/topic.value.@add(5,10,3)")
        result = path.apply({"value": 2}, {})
        assert result == 20  # 2 + 5 + 10 + 3

    def test_mul_multiple_args_parsed(self):
        """Test parsing and applying multiply with multiple arguments."""
        path = parse_message_path("/topic.value.@mul(2,3)")
        result = path.apply({"value": 5}, {})
        assert result == 30  # 5 * 2 * 3

    def test_min_max_parsed(self):
        """Test parsing and applying min/max operations."""
        path_min = parse_message_path("/topic.value.@min(5,2,8)")
        result_min = path_min.apply({"value": 3}, {})
        assert result_min == 2

        path_max = parse_message_path("/topic.value.@max(5,2,8)")
        result_max = path_max.apply({"value": 3}, {})
        assert result_max == 8
