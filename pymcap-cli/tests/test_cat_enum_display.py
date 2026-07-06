"""Tests for `pymcap-cli cat` enum display + tree rendering."""

from __future__ import annotations

import dataclasses
import io
import json
import math
import sys
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from mcap_ros2_support_fast import ROS2EncoderFactory
from pymcap_cli.cmd import cat_cmd as cat_cmd_module
from pymcap_cli.cmd.cat_cmd import cat
from pymcap_cli.display.cat_helpers import plan_for_query, query_result_is_empty
from pymcap_cli.display.message_render import (
    BytesMode,
    EnumField,
    EnumPlan,
    RenderContext,
    TimeKind,
    _scalar_text,
    build_enum_plan,
    changed_leaf_paths,
    render_message_flat,
    render_message_tree,
)
from rich.console import Console
from rich.text import Text
from ros_parser.message_path import parse_message_path
from ros_parser.models import Constant, Field, MessageDefinition
from ros_parser.models import Type as RosType
from small_mcap.writer import McapWriter

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _msg(name: str, *items: Field | Constant) -> MessageDefinition:
    return MessageDefinition(name=name, fields_all=list(items))


def _field(
    type_name: str, name: str, *, package: str | None = None, is_array: bool = False
) -> Field:
    return Field(
        type=RosType(type_name=type_name, package_name=package, is_array=is_array),
        name=name,
    )


def _const(type_name: str, name: str, value: bool | float | str) -> Constant:
    return Constant(type=RosType(type_name=type_name), name=name, value=value)


def test_build_enum_plan_separate_enum_pattern() -> None:
    defs = {
        "pkg/Wrapper": _msg(
            "pkg/Wrapper",
            _field("Status", "status", package="pkg"),
            _field("string", "note"),
        ),
        "pkg/Status": _msg(
            "pkg/Status",
            _const("uint8", "OK", 0),
            _const("uint8", "WARN", 1),
            _const("uint8", "ERROR", 2),
            _field("uint8", "data"),
        ),
    }
    plan = build_enum_plan("pkg/Wrapper", defs)
    assert plan is not None
    assert "status" in plan.enum_fields
    entry = plan.enum_fields["status"]
    assert entry.inner_field == "data"
    assert entry.by_value == {0: "OK", 1: "WARN", 2: "ERROR"}
    assert plan.skip_fields == frozenset()
    assert set(plan.enum_fields) == {"status"}


def test_build_enum_plan_inline_annotation_single_underscore() -> None:
    defs = {
        "pkg/Wrapper": _msg(
            "pkg/Wrapper",
            _field("uint8", "level"),
            _field("LevelEnum", "level_foxglove_enum", package="pkg"),
        ),
        "pkg/LevelEnum": _msg(
            "pkg/LevelEnum",
            _const("uint8", "OK", 0),
            _const("uint8", "WARN", 1),
        ),
    }
    plan = build_enum_plan("pkg/Wrapper", defs)
    assert plan is not None
    assert plan.enum_fields["level"].by_value == {0: "OK", 1: "WARN"}
    assert "level_foxglove_enum" in plan.skip_fields


def test_build_enum_plan_inline_annotation_double_underscore() -> None:
    defs = {
        "pkg/Wrapper": _msg(
            "pkg/Wrapper",
            _field("uint8", "level"),
            _field("LevelEnum", "level__foxglove_enum", package="pkg"),
        ),
        "pkg/LevelEnum": _msg(
            "pkg/LevelEnum",
            _const("uint8", "OK", 0),
        ),
    }
    plan = build_enum_plan("pkg/Wrapper", defs)
    assert plan is not None
    assert plan.enum_fields["level"].by_value == {0: "OK"}
    assert "level__foxglove_enum" in plan.skip_fields


def test_build_enum_plan_custom_inner_field_name() -> None:
    defs = {
        "pkg/Wrapper": _msg("pkg/Wrapper", _field("Color", "color", package="pkg")),
        "pkg/Color": _msg(
            "pkg/Color",
            _const("uint8", "RED", 1),
            _const("uint8", "BLUE", 2),
            _field("uint8", "value"),
        ),
    }
    plan = build_enum_plan("pkg/Wrapper", defs)
    assert plan is not None
    entry = plan.enum_fields["color"]
    assert entry.inner_field == "value"
    assert entry.by_value == {1: "RED", 2: "BLUE"}


def test_build_enum_plan_no_constants_returns_none() -> None:
    defs = {"pkg/Plain": _msg("pkg/Plain", _field("string", "data"))}
    assert build_enum_plan("pkg/Plain", defs) is None


def test_build_enum_plan_type_mismatch_returns_none() -> None:
    defs = {
        "pkg/Wrapper": _msg("pkg/Wrapper", _field("Status", "status", package="pkg")),
        "pkg/Status": _msg(
            "pkg/Status",
            _const("uint8", "OK", 0),
            _field("float32", "data"),
        ),
    }
    assert build_enum_plan("pkg/Wrapper", defs) is None


def test_build_enum_plan_recurses_into_nested() -> None:
    defs = {
        "pkg/Outer": _msg("pkg/Outer", _field("Inner", "inner", package="pkg")),
        "pkg/Inner": _msg("pkg/Inner", _field("Status", "status", package="pkg")),
        "pkg/Status": _msg(
            "pkg/Status",
            _const("uint8", "OK", 0),
            _field("uint8", "data"),
        ),
    }
    plan = build_enum_plan("pkg/Outer", defs)
    assert plan is not None
    inner_plan = plan.nested_plans["inner"]
    assert "status" in inner_plan.enum_fields


def test_build_enum_plan_same_message_constants() -> None:
    # DiagnosticStatus pattern: constants and matching primitive field in one message.
    defs = {
        "diagnostic_msgs/DiagnosticStatus": _msg(
            "diagnostic_msgs/DiagnosticStatus",
            _const("byte", "OK", 0),
            _const("byte", "WARN", 1),
            _const("byte", "ERROR", 2),
            _const("byte", "STALE", 3),
            _field("byte", "level"),
            _field("string", "name"),
            _field("string", "message"),
        ),
    }
    plan = build_enum_plan("diagnostic_msgs/DiagnosticStatus", defs)
    assert plan is not None
    assert plan.enum_fields["level"].by_value == {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}
    # String fields next to byte constants must NOT get decorated.
    assert "name" not in plan.enum_fields
    assert "message" not in plan.enum_fields


def test_build_enum_plan_skips_string_constants_in_same_message() -> None:
    defs = {
        "pkg/Cfg": _msg(
            "pkg/Cfg",
            _const("string", "DEFAULT", "default"),
            _field("string", "value"),
        ),
    }
    # Strings excluded from the same-message rule to avoid spurious decoration.
    assert build_enum_plan("pkg/Cfg", defs) is None


def test_build_enum_plan_handles_msg_path_in_schema_name() -> None:
    defs = {
        "pkg/Wrapper": _msg("pkg/Wrapper", _field("Status", "status", package="pkg")),
        "pkg/Status": _msg(
            "pkg/Status",
            _const("uint8", "OK", 0),
            _field("uint8", "data"),
        ),
    }
    # Schema name in MCAP is the full ROS2 path; resolver should fall back to short form.
    plan = build_enum_plan("pkg/msg/Wrapper", defs)
    assert plan is not None
    assert "status" in plan.enum_fields


def _make_msg_class(class_name: str, slots: list[str]) -> type:
    return dataclasses.make_dataclass(class_name, [(s, Any) for s in slots], slots=True, eq=True)


def _capture(tree: Any) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, width=200, color_system=None).print(tree)
    return buf.getvalue()


def test_render_inline_annotation_skips_annotation_field() -> None:
    annotation_cls = _make_msg_class("Annotation", [])
    wrapper_cls = _make_msg_class("Wrapper", ["level", "level_foxglove_enum", "note"])
    plan = EnumPlan(
        skip_fields=frozenset({"level_foxglove_enum"}),
        enum_fields={"level": EnumField(by_value={0: "OK", 1: "WARN"})},
        nested_plans={},
    )
    obj = wrapper_cls(level=1, level_foxglove_enum=annotation_cls(), note="hi")
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "level: 1 [WARN]" in out
    assert "level_foxglove_enum" not in out
    assert "note: " in out
    assert '"hi"' in out


def test_render_separate_enum_collapses() -> None:
    status_cls = _make_msg_class("Status", ["data"])
    wrapper_cls = _make_msg_class("Wrapper", ["status"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={
            "status": EnumField(
                by_value={0: "OK", 1: "WARN", 2: "ERROR"},
                inner_field="data",
            )
        },
        nested_plans={},
    )
    obj = wrapper_cls(status=status_cls(data=2))
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "status: 2 [ERROR]" in out
    assert "data:" not in out


def test_render_unknown_enum_value_keeps_raw() -> None:
    wrapper_cls = _make_msg_class("Wrapper", ["level"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={"level": EnumField(by_value={0: "OK", 1: "WARN"})},
        nested_plans={},
    )
    obj = wrapper_cls(level=42)
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "level: 42" in out
    assert "[OK]" not in out
    assert "[WARN]" not in out


def test_render_enum_array_values() -> None:
    wrapper_cls = _make_msg_class("Wrapper", ["levels"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={"levels": EnumField(by_value={0: "OK", 1: "WARN"})},
        nested_plans={},
    )
    obj = wrapper_cls(levels=[0, 1, 42])
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "levels: [0 [OK], 1 [WARN], 42]" in out


def test_render_without_plan_works() -> None:
    wrapper_cls = _make_msg_class("Wrapper", ["x", "y"])
    obj = wrapper_cls(x=1, y="hi")
    out = _capture(
        render_message_tree(
            obj, None, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "x: 1" in out
    assert 'y: "hi"' in out


def test_render_dict_input() -> None:
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={"level": EnumField(by_value={1: "WARN"})},
        nested_plans={},
    )
    out = _capture(
        render_message_tree(
            {"level": 1, "msg": "hello"},
            plan,
            title=Text("/topic"),
            bytes_mode=BytesMode.SMART,
            truncate_bytes=0,
        )
    )
    assert "level: 1 [WARN]" in out
    assert 'msg: "hello"' in out


_SEPARATE_ENUM_SCHEMA = (
    b"pymcap_test/Status status\n"
    b"string note\n"
    b"================================================================================\n"
    b"MSG: pymcap_test/msg/Status\n"
    b"uint8 OK=0\n"
    b"uint8 WARN=1\n"
    b"uint8 ERROR=2\n"
    b"uint8 data\n"
)


def _build_separate_enum_mcap(path: Path) -> Path:
    output = BytesIO()
    writer = McapWriter(output=output, encoder_factory=ROS2EncoderFactory())
    writer.start()
    writer.add_schema(1, "pymcap_test/msg/Wrapper", "ros2msg", _SEPARATE_ENUM_SCHEMA)
    writer.add_channel(1, "/wrapper", "cdr", 1)
    for i, level in enumerate([0, 1, 2]):
        writer.add_message_encode(
            channel_id=1,
            log_time=i * 1000,
            data={"status": {"data": level}, "note": f"msg{i}"},
            publish_time=i * 1000,
            sequence=i,
        )
    writer.finish()
    path.write_bytes(output.getvalue())
    return path


def test_cat_tty_shows_enum_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcap_path = _build_separate_enum_mcap(tmp_path / "enum.mcap")
    buf = io.StringIO()
    fake_console = Console(file=buf, force_terminal=True, width=200, color_system=None)
    monkeypatch.setattr(cat_cmd_module, "console_out", fake_console)
    with patch.object(sys.stdout, "isatty", return_value=True):
        rc = cat(str(mcap_path))
    assert rc == 0
    output = buf.getvalue()
    assert "[OK]" in output
    assert "[WARN]" in output
    assert "[ERROR]" in output
    # Inline-collapse: the inner `data` slot shouldn't render as its own subtree.
    assert "data:" not in output
    # Header carries the schema name.
    assert "pymcap_test/msg/Wrapper" in output


def test_cat_jsonl_does_not_decorate_enums(tmp_path: Path) -> None:
    mcap_path = _build_separate_enum_mcap(tmp_path / "enum.mcap")
    out_file = tmp_path / "out.jsonl"
    rc = cat(str(mcap_path), output=out_file)
    assert rc == 0
    lines = [json.loads(line) for line in out_file.read_text().splitlines() if line]
    assert [entry["message"]["status"]["data"] for entry in lines] == [0, 1, 2]
    assert [entry["message"]["note"] for entry in lines] == ["msg0", "msg1", "msg2"]
    for entry in lines:
        for key in entry["message"]:
            assert "_foxglove_enum" not in key
            assert "__name" not in key


def test_build_enum_plan_detects_time_field() -> None:
    defs = {
        "std_msgs/Header": _msg(
            "std_msgs/Header",
            _field("Time", "stamp", package="builtin_interfaces"),
            _field("string", "frame_id"),
        ),
        "builtin_interfaces/Time": _msg(
            "builtin_interfaces/Time",
            _field("int32", "sec"),
            _field("uint32", "nanosec"),
        ),
    }
    plan = build_enum_plan("std_msgs/Header", defs)
    assert plan is not None
    assert plan.time_fields == {"stamp": TimeKind.TIME}


def test_build_enum_plan_detects_duration_field() -> None:
    defs = {
        "pkg/Timeout": _msg(
            "pkg/Timeout",
            _field("Duration", "timeout", package="builtin_interfaces"),
        ),
    }
    plan = build_enum_plan("pkg/Timeout", defs)
    assert plan is not None
    assert plan.time_fields == {"timeout": TimeKind.DURATION}


def test_build_enum_plan_ignores_time_array() -> None:
    defs = {
        "pkg/Stamps": _msg(
            "pkg/Stamps",
            _field("Time", "stamps", package="builtin_interfaces", is_array=True),
        ),
    }
    assert build_enum_plan("pkg/Stamps", defs) is None


def test_render_time_field_annotates_utc_and_keeps_children() -> None:
    time_cls = _make_msg_class("Time", ["sec", "nanosec"])
    header_cls = _make_msg_class("Header", ["stamp", "frame_id"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={},
        nested_plans={},
        time_fields={"stamp": TimeKind.TIME},
    )
    obj = header_cls(stamp=time_cls(sec=1783357226, nanosec=534681181), frame_id="base")
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "2026-07-06T17:00:26.534681181Z UTC" in out
    # Raw breakdown is preserved beneath the annotated header.
    assert "sec: 1783357226" in out
    assert "nanosec: 534681181" in out


def test_render_duration_field_annotates_seconds() -> None:
    dur_cls = _make_msg_class("Duration", ["sec", "nanosec"])
    wrapper_cls = _make_msg_class("Wrapper", ["timeout"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={},
        nested_plans={},
        time_fields={"timeout": TimeKind.DURATION},
    )
    obj = wrapper_cls(timeout=dur_cls(sec=1, nanosec=534681181))
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "1.534681181s" in out
    assert "nanosec: 534681181" in out


def test_render_negative_duration_field() -> None:
    dur_cls = _make_msg_class("Duration", ["sec", "nanosec"])
    wrapper_cls = _make_msg_class("Wrapper", ["timeout"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={},
        nested_plans={},
        time_fields={"timeout": TimeKind.DURATION},
    )
    obj = wrapper_cls(timeout=dur_cls(sec=-2, nanosec=500000000))
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/topic"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "-1.500000000s" in out


def _header_command_defs() -> dict[str, MessageDefinition]:
    return {
        "pkg/Command": _msg(
            "pkg/Command",
            _field("Header", "header", package="std_msgs"),
            _field("float64", "velocity"),
        ),
        "std_msgs/Header": _msg(
            "std_msgs/Header",
            _field("Time", "stamp", package="builtin_interfaces"),
            _field("string", "frame_id"),
        ),
        "builtin_interfaces/Time": _msg(
            "builtin_interfaces/Time",
            _field("int32", "sec"),
            _field("uint32", "nanosec"),
        ),
    }


def test_plan_for_query_none_returns_root_plan() -> None:
    root = build_enum_plan("pkg/Command", _header_command_defs())
    assert plan_for_query(root, None) is root


def test_plan_for_query_topic_only_returns_root_plan() -> None:
    root = build_enum_plan("pkg/Command", _header_command_defs())
    assert plan_for_query(root, parse_message_path("/cmd")) is root


def test_plan_for_query_descends_into_nested_field() -> None:
    root = build_enum_plan("pkg/Command", _header_command_defs())
    plan = plan_for_query(root, parse_message_path("/cmd.header"))
    assert plan is not None
    # The header sub-plan still carries the stamp time annotation.
    assert plan.time_fields == {"stamp": TimeKind.TIME}


def test_plan_for_query_scalar_leaf_returns_none() -> None:
    root = build_enum_plan("pkg/Command", _header_command_defs())
    assert plan_for_query(root, parse_message_path("/cmd.velocity")) is None


def test_plan_for_query_time_leaf_returns_none() -> None:
    # `.header.stamp` extracts the Time message itself; the full-message plan
    # has nothing to attach there, so decoration is skipped (sec/nanosec still render).
    root = build_enum_plan("pkg/Command", _header_command_defs())
    assert plan_for_query(root, parse_message_path("/cmd.header.stamp")) is None


def _quaternion_defs() -> dict[str, MessageDefinition]:
    return {
        "pkg/Pose": _msg("pkg/Pose", _field("Quaternion", "rotation", package="geometry_msgs")),
        "geometry_msgs/Quaternion": _msg(
            "geometry_msgs/Quaternion",
            _field("float64", "x"),
            _field("float64", "y"),
            _field("float64", "z"),
            _field("float64", "w"),
        ),
    }


def test_build_enum_plan_detects_quaternion_field() -> None:
    plan = build_enum_plan("pkg/Pose", _quaternion_defs())
    assert plan is not None
    assert plan.quaternion_fields == frozenset({"rotation"})


def test_build_enum_plan_ignores_quaternion_array() -> None:
    defs = {
        "pkg/Path": _msg(
            "pkg/Path",
            _field("Quaternion", "rotations", package="geometry_msgs", is_array=True),
        ),
    }
    assert build_enum_plan("pkg/Path", defs) is None


def test_render_quaternion_identity_annotates_zero_rpy() -> None:
    quat_cls = _make_msg_class("Quaternion", ["x", "y", "z", "w"])
    pose_cls = _make_msg_class("Pose", ["rotation"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={},
        nested_plans={},
        quaternion_fields=frozenset({"rotation"}),
    )
    obj = pose_cls(rotation=quat_cls(x=0.0, y=0.0, z=0.0, w=1.0))
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/pose"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "rpy [0.0°, 0.0°, 0.0°]" in out
    # x/y/z/w breakdown is preserved.
    assert "w: 1.0" in out


def test_render_quaternion_yaw_90_degrees() -> None:
    quat_cls = _make_msg_class("Quaternion", ["x", "y", "z", "w"])
    pose_cls = _make_msg_class("Pose", ["rotation"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={},
        nested_plans={},
        quaternion_fields=frozenset({"rotation"}),
    )
    # A +90° rotation about z: (x,y,z,w) = (0, 0, sin(45°), cos(45°)). Oracle from spec.
    half = math.radians(45.0)
    obj = pose_cls(rotation=quat_cls(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half)))
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/pose"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "rpy [0.0°, 0.0°, 90.0°]" in out


def test_render_quaternion_zero_norm_skips_annotation() -> None:
    quat_cls = _make_msg_class("Quaternion", ["x", "y", "z", "w"])
    pose_cls = _make_msg_class("Pose", ["rotation"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={},
        nested_plans={},
        quaternion_fields=frozenset({"rotation"}),
    )
    obj = pose_cls(rotation=quat_cls(x=0.0, y=0.0, z=0.0, w=0.0))
    out = _capture(
        render_message_tree(
            obj, plan, title=Text("/pose"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "rpy" not in out
    # Degenerate quaternion still shows its raw components.
    assert "w: 0.0" in out


def test_render_top_level_message_array() -> None:
    ts_cls = _make_msg_class("TransformStamped", ["child_frame_id"])
    obj = [ts_cls(child_frame_id="a"), ts_cls(child_frame_id="b")]
    out = _capture(
        render_message_tree(
            obj, None, title=Text("/tf.transforms"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    assert "[0]" in out
    assert "[1]" in out
    assert '"a"' in out
    assert '"b"' in out


def test_render_top_level_scalar_array() -> None:
    out = _capture(
        render_message_tree(
            ["a", "b", "c"],
            None,
            title=Text("names"),
            bytes_mode=BytesMode.SMART,
            truncate_bytes=0,
        )
    )
    assert '"a", "b", "c"' in out


def test_query_result_is_empty_cases() -> None:
    assert query_result_is_empty(None) is True
    assert query_result_is_empty([]) is True
    assert query_result_is_empty(()) is True
    # Falsy scalars are real values, not "empty".
    assert query_result_is_empty(0) is False
    assert query_result_is_empty(0.0) is False
    assert query_result_is_empty(False) is False
    assert query_result_is_empty("") is False
    assert query_result_is_empty([1]) is False


def test_scalar_text_highlights_non_finite_floats() -> None:
    ctx = RenderContext(BytesMode.SMART, 0)
    assert _scalar_text(float("nan"), ctx).style == "bold red"
    assert _scalar_text(float("inf"), ctx).style == "bold red"
    assert _scalar_text(float("-inf"), ctx).style == "bold red"
    # Finite / non-float values are not highlighted.
    assert _scalar_text(1.5, ctx).style != "bold red"
    assert _scalar_text(0.0, ctx).style != "bold red"
    assert _scalar_text(42, ctx).style != "bold red"


def test_render_highlights_nan_in_message() -> None:
    cls = _make_msg_class("M", ["good", "bad"])
    obj = cls(good=1.5, bad=float("nan"))
    out = _capture(
        render_message_tree(
            obj, None, title=Text("/m"), bytes_mode=BytesMode.SMART, truncate_bytes=0
        )
    )
    # Value still shown (styling is stripped by the color_system=None capture console).
    assert "bad" in out
    assert "nan" in out
    assert "good" in out
    assert "1.5" in out


def _flat_capture(lines: list[Any]) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=200, color_system=None)
    for line in lines:
        console.print(line)
    return buf.getvalue()


def test_render_flat_dotted_paths_and_array_indices() -> None:
    ts_cls = _make_msg_class("TS", ["child_frame_id"])
    root_cls = _make_msg_class("TF", ["transforms"])
    obj = root_cls(transforms=[ts_cls(child_frame_id="odom"), ts_cls(child_frame_id="base")])
    out = _flat_capture(
        render_message_flat(obj, None, bytes_mode=BytesMode.SMART, truncate_bytes=0)
    )
    assert "transforms[0].child_frame_id" in out
    assert '"odom"' in out
    assert "transforms[1].child_frame_id" in out
    assert '"base"' in out


def test_render_flat_collapses_enum_field() -> None:
    cls = _make_msg_class("Wrapper", ["level"])
    plan = EnumPlan(
        skip_fields=frozenset(),
        enum_fields={"level": EnumField(by_value={0: "OK", 1: "WARN"})},
        nested_plans={},
    )
    out = _flat_capture(
        render_message_flat(cls(level=1), plan, bytes_mode=BytesMode.SMART, truncate_bytes=0)
    )
    assert "level" in out
    assert "[WARN]" in out


def test_render_flat_top_level_scalar_has_no_path() -> None:
    # A query that extracts a scalar leaf: flat prints just the value.
    out = _flat_capture(
        render_message_flat(3.5, None, bytes_mode=BytesMode.SMART, truncate_bytes=0)
    )
    assert out.strip() == "3.5"


def test_render_flat_top_level_scalar_list() -> None:
    out = _flat_capture(
        render_message_flat(["a", "b"], None, bytes_mode=BytesMode.SMART, truncate_bytes=0)
    )
    assert '"a", "b"' in out


def test_changed_leaf_paths_detects_scalar_and_array_changes() -> None:
    header_cls = _make_msg_class("H", ["stamp_sec", "frame_id"])
    msg_cls = _make_msg_class("M", ["header", "velocity", "flags"])
    prev = msg_cls(header=header_cls(stamp_sec=1, frame_id="a"), velocity=0.0, flags=[1, 2, 3])
    cur = msg_cls(header=header_cls(stamp_sec=2, frame_id="a"), velocity=0.0, flags=[1, 9, 3])
    assert changed_leaf_paths(prev, cur) == frozenset({"header.stamp_sec", "flags"})


def test_changed_leaf_paths_identical_is_empty() -> None:
    cls = _make_msg_class("M", ["a", "b"])
    assert changed_leaf_paths(cls(a=1, b=2.0), cls(a=1, b=2.0)) == frozenset()


def test_changed_leaf_paths_nested_message_array() -> None:
    ts_cls = _make_msg_class("TS", ["child_frame_id"])
    root_cls = _make_msg_class("TF", ["transforms"])
    prev = root_cls(transforms=[ts_cls(child_frame_id="a"), ts_cls(child_frame_id="b")])
    cur = root_cls(transforms=[ts_cls(child_frame_id="a"), ts_cls(child_frame_id="X")])
    assert changed_leaf_paths(prev, cur) == frozenset({"transforms[1].child_frame_id"})


def test_render_highlights_changed_leaf() -> None:
    cls = _make_msg_class("M", ["value"])
    changed = changed_leaf_paths(cls(value=1), cls(value=2))
    ctx = RenderContext(BytesMode.SMART, 0, changed)
    # The changed leaf renders bold yellow; an unchanged path stays plain.
    assert _scalar_text(2, ctx, "value").style == "bold yellow"
    assert _scalar_text(2, ctx, "other").style != "bold yellow"


def test_render_changed_does_not_override_nan_highlight() -> None:
    ctx = RenderContext(BytesMode.SMART, 0, frozenset({"x"}))
    # NaN highlighting wins over change highlighting.
    assert _scalar_text(float("nan"), ctx, "x").style == "bold red"
