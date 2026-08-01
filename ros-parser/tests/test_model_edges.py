import pytest
from ros_parser.models import (
    ActionDefinition,
    Constant,
    Field,
    MessageDefinition,
    ServiceDefinition,
    Type,
)


def test_type_properties_and_string_representations() -> None:
    primitive = Type("int32")
    bounded = Type("int32", is_array=True, array_size=4, is_upper_bound=True)
    fixed = Type("Point", package_name="geometry_msgs", is_array=True, array_size=2)

    assert primitive.is_primitive
    assert not primitive.is_dynamic_array
    assert not primitive.is_fixed_array
    assert bounded.is_dynamic_array
    assert not bounded.is_fixed_array
    assert str(bounded) == "int32[<=4]"
    assert str(fixed) == "geometry_msgs/Point[2]"


def test_field_and_constant_validation_and_strings() -> None:
    assert str(Field(Type("string"), "label", "robot")) == "string label 'robot'"
    assert str(Field(Type("int32"), "count", 3)) == "int32 count 3"
    assert str(Constant(Type("string"), "LABEL", "robot")) == "string LABEL='robot'"
    assert str(Constant(Type("int32"), "COUNT", 3)) == "int32 COUNT=3"

    with pytest.raises(TypeError, match="primitive, non-array"):
        Constant(Type("Point", package_name="geometry_msgs"), "POINT", 1)
    with pytest.raises(TypeError, match="primitive, non-array"):
        Constant(Type("int32", is_array=True), "VALUES", 1)


def test_definition_validation_and_string_representations() -> None:
    field = Field(Type("int32"), "value")
    constant = Constant(Type("int32"), "DEFAULT", 1)
    message = MessageDefinition("example/Message", [constant, field])

    assert message.fields == [field]
    assert message.constants == [constant]
    assert str(message) == "# example/Message\nint32 DEFAULT=1\nint32 value"
    assert str(MessageDefinition(None, [field])) == "int32 value"

    with pytest.raises(ValueError, match="Duplicate field/constant names: value"):
        MessageDefinition(
            "example/Duplicate",
            [field, Constant(Type("int32"), "value", 1)],
        )

    request = MessageDefinition("example/Add_Request", [Field(Type("int32"), "a")])
    response = MessageDefinition("example/Add_Response", [Field(Type("int32"), "sum")])
    service = ServiceDefinition("example/Add", request, response)
    assert str(service) == (
        "# example/Add\n# example/Add_Request\nint32 a\n---\n# example/Add_Response\nint32 sum"
    )

    goal = MessageDefinition("example/Move_Goal", [Field(Type("int32"), "target")])
    result = MessageDefinition("example/Move_Result", [Field(Type("bool"), "success")])
    feedback = MessageDefinition("example/Move_Feedback", [Field(Type("float32"), "progress")])
    action = ActionDefinition("example/Move", goal, result, feedback)
    assert str(action) == (
        "# example/Move\n"
        "# example/Move_Goal\nint32 target\n"
        "---\n"
        "# example/Move_Result\nbool success\n"
        "---\n"
        "# example/Move_Feedback\nfloat32 progress"
    )
