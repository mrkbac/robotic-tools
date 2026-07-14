"""Tests for message path validator."""

import pytest
from ros_parser import Constant, Field, MessageDefinition, Type
from ros_parser.message_path import ValidationError, parse_message_path


class TestBasicValidation:
    """Test basic validation scenarios."""

    def test_simple_field_access_valid(self):
        """Test that valid field access passes validation."""
        msgdef = MessageDefinition(
            name="test_msgs/SimpleMessage",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
            ],
        )
        path = parse_message_path("/topic.x")
        all_defs = {"test_msgs/SimpleMessage": msgdef, "SimpleMessage": msgdef}

        # Should not raise
        path.validate(msgdef, all_defs)

    def test_invalid_field_access(self):
        """Test that accessing non-existent field raises ValidationError."""
        msgdef = MessageDefinition(
            name="test_msgs/SimpleMessage",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
            ],
        )
        path = parse_message_path("/topic.z")
        all_defs = {"test_msgs/SimpleMessage": msgdef}

        with pytest.raises(ValidationError) as exc_info:
            path.validate(msgdef, all_defs)

        assert "Field 'z' not found" in str(exc_info.value)
        assert "Available fields: x, y" in str(exc_info.value)

    def test_field_access_on_primitive(self):
        """Test that accessing field on primitive type raises ValidationError."""
        msgdef = MessageDefinition(
            name="test_msgs/SimpleMessage",
            fields_all=[
                Field(Type(type_name="float64"), "value"),
            ],
        )
        path = parse_message_path("/topic.value.something")
        all_defs = {"test_msgs/SimpleMessage": msgdef}

        with pytest.raises(ValidationError) as exc_info:
            path.validate(msgdef, all_defs)

        assert "Cannot access field" in str(exc_info.value)
        assert "primitive type" in str(exc_info.value)


class TestNestedMessages:
    """Test validation with nested message types."""

    def test_nested_message_access_valid(self):
        """Test accessing fields in nested messages."""
        point_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
                Field(Type(type_name="float64"), "z"),
            ],
        )
        pose_def = MessageDefinition(
            name="geometry_msgs/Pose",
            fields_all=[
                Field(Type(type_name="Point", package_name="geometry_msgs"), "position"),
            ],
        )

        all_defs = {
            "geometry_msgs/msg/Point": point_def,
            "geometry_msgs/Point": point_def,
            "Point": point_def,
            "geometry_msgs/msg/Pose": pose_def,
            "geometry_msgs/Pose": pose_def,
            "Pose": pose_def,
        }

        path = parse_message_path("/topic.position.x")
        path.validate(pose_def, all_defs)

    def test_nested_message_invalid_field(self):
        """Test accessing invalid field in nested message."""
        point_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
            ],
        )
        pose_def = MessageDefinition(
            name="geometry_msgs/Pose",
            fields_all=[
                Field(Type(type_name="Point", package_name="geometry_msgs"), "position"),
            ],
        )

        all_defs = {
            "geometry_msgs/msg/Point": point_def,
            "geometry_msgs/Point": point_def,
            "Point": point_def,
            "geometry_msgs/Pose": pose_def,
        }

        path = parse_message_path("/topic.position.w")

        with pytest.raises(ValidationError) as exc_info:
            path.validate(pose_def, all_defs)

        assert "Field 'w' not found" in str(exc_info.value)


class TestArrayValidation:
    """Test validation with arrays."""

    def test_array_index_valid(self):
        """Test that indexing array is valid."""
        msgdef = MessageDefinition(
            name="test_msgs/ArrayMessage",
            fields_all=[
                Field(
                    Type(type_name="float64", is_array=True, array_size=None),
                    "values",
                ),
            ],
        )
        path = parse_message_path("/topic.values[0]")
        all_defs = {"test_msgs/ArrayMessage": msgdef}

        path.validate(msgdef, all_defs)

    def test_array_index_on_non_array(self):
        """Test that indexing non-array raises ValidationError."""
        msgdef = MessageDefinition(
            name="test_msgs/SimpleMessage",
            fields_all=[
                Field(Type(type_name="float64"), "value"),
            ],
        )
        path = parse_message_path("/topic.value[0]")
        all_defs = {"test_msgs/SimpleMessage": msgdef}

        with pytest.raises(ValidationError) as exc_info:
            path.validate(msgdef, all_defs)

        assert "Cannot apply array index" in str(exc_info.value)
        assert "non-array type" in str(exc_info.value)

    def test_array_slice_valid(self):
        """Test that slicing array is valid."""
        msgdef = MessageDefinition(
            name="test_msgs/ArrayMessage",
            fields_all=[
                Field(
                    Type(type_name="float64", is_array=True, array_size=None),
                    "values",
                ),
            ],
        )
        path = parse_message_path("/topic.values[1:3]")
        all_defs = {"test_msgs/ArrayMessage": msgdef}

        path.validate(msgdef, all_defs)

    def test_array_of_complex_types(self):
        """Test accessing fields in array elements."""
        point_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
            ],
        )
        msgdef = MessageDefinition(
            name="test_msgs/PointArray",
            fields_all=[
                Field(
                    Type(
                        type_name="Point",
                        package_name="geometry_msgs",
                        is_array=True,
                        array_size=None,
                    ),
                    "points",
                ),
            ],
        )

        all_defs = {
            "geometry_msgs/msg/Point": point_def,
            "geometry_msgs/Point": point_def,
            "Point": point_def,
            "test_msgs/PointArray": msgdef,
        }

        path = parse_message_path("/topic.points[0].x")
        path.validate(msgdef, all_defs)

    def test_field_access_on_array_without_index(self):
        """Test that accessing field on array without indexing raises error."""
        point_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
            ],
        )
        msgdef = MessageDefinition(
            name="test_msgs/PointArray",
            fields_all=[
                Field(
                    Type(
                        type_name="Point",
                        package_name="geometry_msgs",
                        is_array=True,
                    ),
                    "points",
                ),
            ],
        )

        all_defs = {
            "geometry_msgs/Point": point_def,
            "test_msgs/PointArray": msgdef,
        }

        path = parse_message_path("/topic.points.x")

        with pytest.raises(ValidationError) as exc_info:
            path.validate(msgdef, all_defs)

        assert "Cannot access field 'x' on array type" in str(exc_info.value)


class TestFilterValidation:
    """Test validation of filter operations."""

    def test_simple_filter_valid(self):
        """Test that valid filter passes validation on array."""
        item_def = MessageDefinition(
            name="test_msgs/Item",
            fields_all=[
                Field(Type(type_name="float64"), "value"),
            ],
        )
        msgdef = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[
                Field(
                    Type(type_name="Item", package_name="test_msgs", is_array=True),
                    "items",
                ),
            ],
        )

        all_defs = {
            "test_msgs/Message": msgdef,
            "test_msgs/Item": item_def,
            "Item": item_def,
        }

        path = parse_message_path("/topic.items{value>5.0}")
        path.validate(msgdef, all_defs)

    def test_current_value_filter_valid_on_primitive(self) -> None:
        msgdef = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[Field(Type(type_name="float64"), "value")],
        )

        parse_message_path("/topic.value{<=30}").validate(msgdef, {"test_msgs/Message": msgdef})

    def test_current_value_filter_valid_on_primitive_array(self) -> None:
        msgdef = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[Field(Type(type_name="float64", is_array=True, array_size=None), "values")],
        )

        parse_message_path("/topic.values[:]{<=30}").validate(msgdef, {"test_msgs/Message": msgdef})

    def test_current_value_filter_rejects_complex_object(self) -> None:
        point_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[Field(Type(type_name="float64"), "x")],
        )
        msgdef = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[Field(Type(type_name="Point", package_name="geometry_msgs"), "point")],
        )
        definitions = {
            "test_msgs/Message": msgdef,
            "geometry_msgs/Point": point_def,
        }

        with pytest.raises(ValidationError, match=r"current-value filter.*primitive"):
            parse_message_path("/topic.point{<=30}").validate(msgdef, definitions)

    def test_current_value_in_filter_valid_on_primitive(self) -> None:
        msgdef = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[Field(Type(type_name="int32"), "value")],
        )

        parse_message_path("/topic.value{in [1, 3]}").validate(
            msgdef, {"test_msgs/Message": msgdef}
        )

    def test_enum_name_filter_resolves_schema_constant(self) -> None:
        uint8 = Type(type_name="uint8")
        msgdef = MessageDefinition(
            name="test_msgs/State",
            fields_all=[
                Constant(uint8, "MOVING", 2),
                Field(uint8, "status"),
            ],
        )
        path = parse_message_path("/topic{status==MOVING}")

        path.validate(msgdef, {"test_msgs/State": msgdef})

        assert path.apply({"status": 2}) == {"status": 2}
        assert path.apply({"status": 1}) is None

    def test_enum_name_filter_is_case_sensitive(self) -> None:
        uint8 = Type(type_name="uint8")
        msgdef = MessageDefinition(
            name="test_msgs/State",
            fields_all=[
                Constant(uint8, "MOVING", 2),
                Field(uint8, "status"),
            ],
        )

        with pytest.raises(ValidationError, match="moving"):
            parse_message_path("/topic{status==moving}").validate(
                msgdef, {"test_msgs/State": msgdef}
            )

    def test_nested_enum_name_uses_field_owner_constants(self) -> None:
        uint8 = Type(type_name="uint8")
        state_def = MessageDefinition(
            name="test_msgs/State",
            fields_all=[
                Constant(uint8, "MOVING", 2),
                Field(uint8, "status"),
            ],
        )
        root_def = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[Field(Type(type_name="State", package_name="test_msgs"), "state")],
        )
        definitions = {
            "test_msgs/Message": root_def,
            "test_msgs/State": state_def,
        }
        path = parse_message_path("/topic{state.status==MOVING}")

        path.validate(root_def, definitions)

        assert path.apply({"state": {"status": 2}}) == {"state": {"status": 2}}

    def test_enum_names_work_in_membership_filter(self) -> None:
        uint8 = Type(type_name="uint8")
        msgdef = MessageDefinition(
            name="test_msgs/State",
            fields_all=[
                Constant(uint8, "STOPPED", 1),
                Constant(uint8, "MOVING", 2),
                Field(uint8, "status"),
            ],
        )
        path = parse_message_path("/topic{status in [STOPPED, MOVING]}")

        path.validate(msgdef, {"test_msgs/State": msgdef})

        assert path.apply({"status": 1}) == {"status": 1}
        assert path.apply({"status": 2}) == {"status": 2}
        assert path.apply({"status": 3}) is None

    def test_filter_with_field_path_valid(self):
        """Test filter with field path."""
        point_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
            ],
        )
        msgdef = MessageDefinition(
            name="test_msgs/PointArray",
            fields_all=[
                Field(
                    Type(
                        type_name="Point",
                        package_name="geometry_msgs",
                        is_array=True,
                    ),
                    "points",
                ),
            ],
        )

        all_defs = {
            "geometry_msgs/msg/Point": point_def,
            "geometry_msgs/Point": point_def,
            "Point": point_def,
            "test_msgs/PointArray": msgdef,
        }

        path = parse_message_path("/topic.points{x>0}")
        path.validate(msgdef, all_defs)

    def test_filter_with_invalid_field(self):
        """Test filter with invalid field path raises error."""
        point_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
            ],
        )
        msgdef = MessageDefinition(
            name="test_msgs/PointArray",
            fields_all=[
                Field(
                    Type(
                        type_name="Point",
                        package_name="geometry_msgs",
                        is_array=True,
                    ),
                    "points",
                ),
            ],
        )

        all_defs = {
            "geometry_msgs/Point": point_def,
            "test_msgs/PointArray": msgdef,
        }

        path = parse_message_path("/topic.points{z>0}")

        with pytest.raises(ValidationError) as exc_info:
            path.validate(msgdef, all_defs)

        assert "Field 'z' not found" in str(exc_info.value)

    def test_filter_nested_field_path(self):
        """Test filter with nested field path."""
        position_def = MessageDefinition(
            name="geometry_msgs/Point",
            fields_all=[
                Field(Type(type_name="float64"), "x"),
                Field(Type(type_name="float64"), "y"),
            ],
        )
        pose_def = MessageDefinition(
            name="geometry_msgs/Pose",
            fields_all=[
                Field(Type(type_name="Point", package_name="geometry_msgs"), "position"),
            ],
        )
        msgdef = MessageDefinition(
            name="test_msgs/PoseArray",
            fields_all=[
                Field(
                    Type(
                        type_name="Pose",
                        package_name="geometry_msgs",
                        is_array=True,
                    ),
                    "poses",
                ),
            ],
        )

        all_defs = {
            "geometry_msgs/msg/Point": position_def,
            "geometry_msgs/Point": position_def,
            "Point": position_def,
            "geometry_msgs/msg/Pose": pose_def,
            "geometry_msgs/Pose": pose_def,
            "Pose": pose_def,
            "test_msgs/PoseArray": msgdef,
        }

        path = parse_message_path("/topic.poses{position.x>0}")
        path.validate(msgdef, all_defs)


class TestErrorMessages:
    """Test that error messages are helpful and informative."""

    def test_error_includes_path_context(self):
        """Test that error messages include the path being validated."""
        msgdef = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[
                Field(Type(type_name="float64"), "value"),
            ],
        )
        path = parse_message_path("/topic.value.nested")
        all_defs = {"test_msgs/Message": msgdef}

        with pytest.raises(ValidationError) as exc_info:
            path.validate(msgdef, all_defs)

        # Error should mention where in the path the error occurred
        error_msg = str(exc_info.value)
        assert "/topic.value.nested" in error_msg or "nested" in error_msg

    def test_error_suggests_available_fields(self):
        """Test that error messages list available fields."""
        msgdef = MessageDefinition(
            name="test_msgs/Message",
            fields_all=[
                Field(Type(type_name="float64"), "foo"),
                Field(Type(type_name="float64"), "bar"),
                Field(Type(type_name="float64"), "baz"),
            ],
        )
        path = parse_message_path("/topic.invalid")
        all_defs = {"test_msgs/Message": msgdef}

        with pytest.raises(ValidationError) as exc_info:
            path.validate(msgdef, all_defs)

        error_msg = str(exc_info.value)
        assert "Available fields" in error_msg
        assert "foo" in error_msg
        assert "bar" in error_msg
        assert "baz" in error_msg
