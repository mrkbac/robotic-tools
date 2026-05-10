"""Tests for `pymcap-cli cat` enum display + tree rendering."""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from mcap_ros2_support_fast import ROS2EncoderFactory
from pymcap_cli.cmd import cat_cmd as cat_cmd_module
from pymcap_cli.cmd.cat_cmd import cat
from pymcap_cli.display.message_render import (
    BytesMode,
    EnumField,
    EnumPlan,
    build_enum_plan,
    render_message_tree,
)
from rich.console import Console
from rich.text import Text
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
