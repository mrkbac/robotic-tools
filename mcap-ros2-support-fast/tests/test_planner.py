from types import SimpleNamespace

from mcap_ros2_support_fast._planner import generate_plans, optimize_plan
from mcap_ros2_support_fast._plans import (
    ActionType,
    ComplexAction,
    PlanList,
    PrimitiveAction,
    PrimitiveGroupAction,
    TypeId,
)

TF_MSG = """
geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id # the frame id of the child frame
Transform transform
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: geometry_msgs/Transform
Vector3 translation
Quaternion rotation
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""


def compare_plans(actual_plan: PlanList, expected_plan: PlanList) -> None:
    """Compare two plans for equality."""
    assert actual_plan[0].__name__ == expected_plan[0].__name__
    assert len(actual_plan[1]) == len(expected_plan[1])
    for actual_action, expected_action in zip(actual_plan[1], expected_plan[1], strict=True):
        # Compare dataclass actions directly
        assert actual_action.type == expected_action.type
        if actual_action.type in (
            ActionType.PRIMITIVE,
            ActionType.PRIMITIVE_ARRAY,
            ActionType.PRIMITIVE_GROUP,
        ):
            assert actual_action == expected_action
        elif (
            actual_action.type == ActionType.COMPLEX and expected_action.type == ActionType.COMPLEX
        ):
            assert actual_action.target == expected_action.target
            compare_plans(actual_action.plan, expected_action.plan)
        elif (
            actual_action.type == ActionType.COMPLEX_ARRAY
            and expected_action.type == ActionType.COMPLEX_ARRAY
        ):
            assert actual_action.target == expected_action.target
            assert actual_action.size == expected_action.size
            compare_plans(actual_action.plan, expected_action.plan)
        else:
            raise ValueError(
                f"Unexpected action type: {actual_action.type} - {expected_action.type}"
            )


def test_generate_plan() -> None:
    """Test that plan generation produces the expected structure."""
    plan = generate_plans("geometry_msgs/TransformStamped", TF_MSG)

    expected_plan = (
        type("geometry_msgs_TransformStamped", (SimpleNamespace,), {}),
        [
            ComplexAction(
                target="header",
                plan=(
                    type("std_msgs_Header", (SimpleNamespace,), {}),
                    [
                        ComplexAction(
                            target="stamp",
                            plan=(
                                type("builtin_interfaces_Time", (SimpleNamespace,), {}),
                                [
                                    PrimitiveAction(target="sec", data=TypeId.INT32),
                                    PrimitiveAction(target="nanosec", data=TypeId.UINT32),
                                ],
                            ),
                        ),
                        PrimitiveAction(target="frame_id", data=TypeId.STRING),
                    ],
                ),
            ),
            PrimitiveAction(target="child_frame_id", data=TypeId.STRING),
            ComplexAction(
                target="transform",
                plan=(
                    # transform_cls,
                    type("geometry_msgs_Transform", (SimpleNamespace,), {}),
                    [
                        ComplexAction(
                            target="translation",
                            plan=(
                                type("geometry_msgs_Vector3", (SimpleNamespace,), {}),
                                [
                                    PrimitiveAction(target="x", data=TypeId.FLOAT64),
                                    PrimitiveAction(target="y", data=TypeId.FLOAT64),
                                    PrimitiveAction(target="z", data=TypeId.FLOAT64),
                                ],
                            ),
                        ),
                        ComplexAction(
                            target="rotation",
                            plan=(
                                type("geometry_msgs_Quaternion", (SimpleNamespace,), {}),
                                [
                                    PrimitiveAction(target="x", data=TypeId.FLOAT64),
                                    PrimitiveAction(target="y", data=TypeId.FLOAT64),
                                    PrimitiveAction(target="z", data=TypeId.FLOAT64),
                                    PrimitiveAction(target="w", data=TypeId.FLOAT64),
                                ],
                            ),
                        ),
                    ],
                ),
            ),
        ],
    )

    compare_plans(plan, expected_plan)


def test_generate_plan_optimized() -> None:
    plan = generate_plans("geometry_msgs/TransformStamped", TF_MSG)
    plan = optimize_plan(plan)

    expected_plan = (
        type("geometry_msgs_TransformStamped", (SimpleNamespace,), {}),
        [
            ComplexAction(
                target="header",
                plan=(
                    type("std_msgs_Header", (SimpleNamespace,), {}),
                    [
                        ComplexAction(
                            target="stamp",
                            plan=(
                                type("builtin_interfaces_Time", (SimpleNamespace,), {}),
                                [
                                    PrimitiveGroupAction(
                                        targets=[("sec", TypeId.INT32), ("nanosec", TypeId.UINT32)]
                                    )
                                ],
                            ),
                        ),
                        PrimitiveAction(target="frame_id", data=TypeId.STRING),
                    ],
                ),
            ),
            PrimitiveAction(target="child_frame_id", data=TypeId.STRING),
            ComplexAction(
                target="transform",
                plan=(
                    type("geometry_msgs_Transform", (SimpleNamespace,), {}),
                    [
                        ComplexAction(
                            target="translation",
                            plan=(
                                type("geometry_msgs_Vector3", (SimpleNamespace,), {}),
                                [
                                    PrimitiveGroupAction(
                                        targets=[
                                            ("x", TypeId.FLOAT64),
                                            ("y", TypeId.FLOAT64),
                                            ("z", TypeId.FLOAT64),
                                        ]
                                    )
                                ],
                            ),
                        ),
                        ComplexAction(
                            target="rotation",
                            plan=(
                                type("geometry_msgs_Quaternion", (SimpleNamespace,), {}),
                                [
                                    PrimitiveGroupAction(
                                        targets=[
                                            ("x", TypeId.FLOAT64),
                                            ("y", TypeId.FLOAT64),
                                            ("z", TypeId.FLOAT64),
                                            ("w", TypeId.FLOAT64),
                                        ]
                                    )
                                ],
                            ),
                        ),
                    ],
                ),
            ),
        ],
    )

    compare_plans(plan, expected_plan)
