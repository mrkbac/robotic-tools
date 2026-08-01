from types import SimpleNamespace

import mcap_ros2_support_fast._planner as planner
import pytest
from mcap_ros2_support_fast._plans import ComplexAction, McapROS2DecodeError
from ros_parser import MessageDefinition
from ros_parser.models import Field, Type


def test_constant_cannot_be_deleted() -> None:
    message_type, _ = planner.generate_plans("example_msgs/State", "uint8 READY=1\nuint8 state")

    with pytest.raises(AttributeError, match=r"READY.*read-only"):
        del message_type.READY


def test_nonconstant_class_attribute_can_be_deleted() -> None:
    message_type, _ = planner.generate_plans("example_msgs/State", "uint8 READY=1\nuint8 state")
    message_type.description = "temporary"

    del message_type.description

    assert "description" not in vars(message_type)


def test_generate_plans_rejects_missing_nested_definition() -> None:
    with pytest.raises(ValueError, match="Message definition not found"):
        planner.generate_plans("example_msgs/Wrapper", "example_msgs/Missing value")


def test_generate_plans_rejects_unknown_primitive() -> None:
    message_definition = MessageDefinition(
        "example_msgs/Value",
        [Field(Type(type_name="unknown"), "value")],
    )

    with pytest.raises(ValueError, match="Unknown primitive type"):
        planner._generate_plan(message_definition, {})


def test_generate_plans_rejects_missing_primary_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(planner, "for_each_msgdef", lambda *_args: None)

    with pytest.raises(McapROS2DecodeError, match="schema parsing failed"):
        planner.generate_plans("example_msgs/Value", "uint32 value")


def test_find_groupable_primitives_handles_empty_plan() -> None:
    assert planner._find_groupable_primitives([]) == []


def test_create_primitive_groups_rejects_nonprimitive_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_plan = (SimpleNamespace, [])
    steps = [ComplexAction("value", empty_plan)]
    monkeypatch.setattr(planner, "_find_groupable_primitives", lambda _steps: [(0, 0)])

    with pytest.raises(ValueError, match="Unexpected action type"):
        planner._create_primitive_groups(steps)
