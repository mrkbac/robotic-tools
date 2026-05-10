"""Shared decoded-message rendering helpers.

Used by both `pymcap-cli cat` (file streaming) and `pymcap-cli bridge cat`
(live bridge streaming) so the two commands stay byte-for-byte consistent.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    import re

    from ros_parser.models import Constant, MessageDefinition
    from ros_parser.models import Type as RosType

TTY_BYTES_TRUNCATE = 32
# Threshold (bytes) below which `smart` mode inlines the full payload as an int
# list. Anything larger becomes a `<N bytes>` placeholder so large binary
# payloads (Image, PointCloud2) don't drown out the rest of the message.
SMART_BYTES_INLINE_LIMIT = 64


class BytesMode(str, Enum):
    """How to serialize bytes fields in JSON output."""

    SMART = "smart"
    INTS = "ints"
    BASE64 = "base64"
    SKIP = "skip"


def message_matches_grep(obj: Any, pattern: re.Pattern[str]) -> bool:
    """Return True if ``pattern`` matches any scalar value reachable from ``obj``.

    Walks decoded message dataclasses, lists/tuples, dicts, scalars. Bytes-like
    fields are skipped — regex'ing image/point-cloud payloads is rarely what
    a user wants and is expensive on large messages.
    """
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return False

    if _is_message_obj(obj):
        return any(message_matches_grep(value, pattern) for _, value in _message_items(obj))

    if isinstance(obj, (list, tuple)):
        return any(message_matches_grep(item, pattern) for item in obj)

    if isinstance(obj, dict):
        return any(message_matches_grep(value, pattern) for value in obj.values())

    if obj is None:
        return False

    return bool(pattern.search(str(obj)))


def message_to_dict(
    obj: Any,
    *,
    bytes_mode: BytesMode = BytesMode.SMART,
    truncate_bytes: int = 0,
) -> Any:
    """Recursively convert a message object to a JSON-serializable dict.

    Handles __slots__-based objects, sequences, and bytes-like objects.
    The ``truncate_bytes`` parameter is used for TTY display to keep output manageable.
    """
    recurse = lambda v: message_to_dict(v, bytes_mode=bytes_mode, truncate_bytes=truncate_bytes)  # noqa: E731

    if _is_message_obj(obj):
        return {name: recurse(value) for name, value in _message_items(obj)}

    if isinstance(obj, (list, tuple)):
        return [recurse(item) for item in obj]

    if isinstance(obj, (bytes, bytearray, memoryview)):
        total = len(obj)
        if bytes_mode == BytesMode.SKIP:
            return f"<{total} bytes>"
        if bytes_mode == BytesMode.BASE64:
            return base64.b64encode(bytes(obj)).decode("ascii")
        if bytes_mode == BytesMode.SMART:
            # Small payloads inline as ints for easy inspection; large ones
            # collapse to a placeholder so output stays grep/jq-friendly.
            if total <= SMART_BYTES_INLINE_LIMIT:
                return list(obj)
            return f"<{total} bytes>"
        if truncate_bytes and total > truncate_bytes:
            return [*list(obj[:truncate_bytes]), f"... ({total} bytes total)"]
        return list(obj)

    return obj


_FOXGLOVE_ENUM_SUFFIXES = ("__foxglove_enum", "_foxglove_enum")
# Integer-ish primitive types eligible for the "same-message constants" rule.
# Floats/strings/time/duration are excluded because they're rarely used as enums
# and the heuristic would over-trigger (e.g. on string defaults).
_ENUM_PRIMITIVE_TYPES = frozenset(
    {
        "bool",
        "byte",
        "char",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    }
)
# Cap rendered list lengths so a 100k-element array doesn't drown the panel.
_TREE_PRIMITIVE_ARRAY_LIMIT = 12
_TREE_COMPLEX_ARRAY_LIMIT = 16


@dataclass(frozen=True)
class EnumField:
    """Enum labels for one rendered field, optionally read from a wrapper member."""

    by_value: dict[bool | int | float | str, str]
    inner_field: str | None = None


@dataclass(frozen=True)
class EnumPlan:
    """Per-MessageDefinition rendering hints used by the TTY tree renderer."""

    skip_fields: frozenset[str]
    enum_fields: dict[str, EnumField]
    nested_plans: dict[str, EnumPlan]


@dataclass(frozen=True)
class RenderContext:
    bytes_mode: BytesMode
    truncate_bytes: int

    def format(self, value: Any) -> str:
        return _format_scalar(value, bytes_mode=self.bytes_mode, truncate_bytes=self.truncate_bytes)


def _message_items(obj: Any) -> list[tuple[str, Any]]:
    return [
        (f.name, getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if not f.name.startswith("_")
    ]


def _is_tree_obj(obj: Any) -> bool:
    return isinstance(obj, dict) or _is_message_obj(obj)


def _annotated_target_name(field_name: str) -> str | None:
    """Return the value-field name if `field_name` is a `_foxglove_enum` annotation, else None."""
    for suffix in _FOXGLOVE_ENUM_SUFFIXES:
        if field_name.endswith(suffix):
            base = field_name[: -len(suffix)]
            if base:
                return base
    return None


def _resolve_msgdef(
    field_type: RosType,
    all_definitions: dict[str, MessageDefinition],
) -> MessageDefinition | None:
    """Resolve a complex Type to its MessageDefinition, mirroring ros_parser's key probing."""
    if field_type.is_primitive:
        return None
    type_name = field_type.type_name
    package = field_type.package_name
    if package is not None:
        return (
            all_definitions.get(f"{package}/msg/{type_name}")
            or all_definitions.get(f"{package}/{type_name}")
            or all_definitions.get(type_name)
        )
    return all_definitions.get(type_name)


def resolve_msgdef_by_name(
    schema_name: str,
    all_definitions: dict[str, MessageDefinition],
) -> MessageDefinition | None:
    """Look up a MessageDefinition by schema name, trying common key variants."""
    msgdef = all_definitions.get(schema_name)
    if msgdef is not None:
        return msgdef
    if "/msg/" in schema_name:
        msgdef = all_definitions.get(schema_name.replace("/msg/", "/"))
        if msgdef is not None:
            return msgdef
    bare = schema_name.rsplit("/", maxsplit=1)[-1]
    return all_definitions.get(bare)


def _constants_to_field_enum(
    constants: list[Constant],
    target_type_name: str,
    *,
    inner_field: str | None = None,
) -> EnumField | None:
    """Build an EnumField from constants whose primitive type matches `target_type_name`."""
    by_value: dict[bool | int | float | str, str] = {}
    for c in constants:
        if c.type.type_name != target_type_name:
            continue
        # First name wins on duplicate values — match the source ordering.
        by_value.setdefault(c.value, c.name)
    if not by_value:
        return None
    return EnumField(by_value=by_value, inner_field=inner_field)


def _separate_enum_field(msgdef: MessageDefinition) -> EnumField | None:
    """Return a collapsible enum field for messages shaped as constants + one value field."""
    if not msgdef.constants or len(msgdef.fields) != 1:
        return None
    inner = msgdef.fields[0]
    if not inner.type.is_primitive or inner.type.is_array:
        return None
    return _constants_to_field_enum(
        msgdef.constants,
        inner.type.type_name,
        inner_field=inner.name,
    )


def build_enum_plan(
    schema_name: str,
    all_definitions: dict[str, MessageDefinition],
    _visited: dict[str, EnumPlan | None] | None = None,
) -> EnumPlan | None:
    """Walk a schema graph and return an EnumPlan describing where to surface enum names.

    Returns None when no field anywhere in the message tree carries enum decoration.
    """
    if _visited is None:
        _visited = {}
    if schema_name in _visited:
        return _visited[schema_name]

    msgdef = resolve_msgdef_by_name(schema_name, all_definitions)
    if msgdef is None:
        _visited[schema_name] = None
        return None

    # Tentative None guards against cycles while we recurse.
    _visited[schema_name] = None

    skip_fields: set[str] = set()
    enum_fields: dict[str, EnumField] = {}
    nested_plans: dict[str, EnumPlan] = {}

    fields_by_name = {f.name: f for f in msgdef.fields}

    for f in msgdef.fields:
        target = _annotated_target_name(f.name)
        if target is None or target not in fields_by_name:
            continue
        target_field = fields_by_name[target]
        if not target_field.type.is_primitive or target_field.type.is_array:
            continue
        ann_msgdef = _resolve_msgdef(f.type, all_definitions)
        if ann_msgdef is None or not ann_msgdef.constants:
            continue
        field_enum = _constants_to_field_enum(ann_msgdef.constants, target_field.type.type_name)
        if field_enum is None:
            continue
        enum_fields[target] = field_enum
        skip_fields.add(f.name)

    for f in msgdef.fields:
        if (
            f.name in skip_fields
            or _annotated_target_name(f.name) is not None
            or f.name in enum_fields
        ):
            continue

        if f.type.is_primitive:
            # Same-message constants: e.g. DiagnosticStatus declares OK/WARN/ERROR
            # constants alongside a `level: byte` field. Restricted to integer-ish
            # types to avoid wrongly decorating string fields next to string constants.
            if msgdef.constants and f.type.type_name in _ENUM_PRIMITIVE_TYPES:
                same_msg_enum = _constants_to_field_enum(msgdef.constants, f.type.type_name)
                if same_msg_enum is not None:
                    enum_fields[f.name] = same_msg_enum
            continue

        nested_msgdef = _resolve_msgdef(f.type, all_definitions)
        if nested_msgdef is None:
            continue

        # "Separate enum message" pattern: constants + a single primitive non-array field,
        # consumed only when accessed via a non-array field (the array case is rare and the
        # collapse semantics are unclear there).
        if not f.type.is_array:
            separate_enum = _separate_enum_field(nested_msgdef)
            if separate_enum is not None:
                enum_fields[f.name] = separate_enum
                continue

        nested_key = (
            f"{f.type.package_name}/{f.type.type_name}" if f.type.package_name else f.type.type_name
        )
        sub_plan = build_enum_plan(nested_key, all_definitions, _visited)
        if sub_plan is not None:
            nested_plans[f.name] = sub_plan

    if not (skip_fields or enum_fields or nested_plans):
        _visited[schema_name] = None
        return None

    plan = EnumPlan(
        skip_fields=frozenset(skip_fields),
        enum_fields=dict(enum_fields),
        nested_plans=dict(nested_plans),
    )
    _visited[schema_name] = plan
    return plan


def _is_message_obj(obj: Any) -> bool:
    """True if `obj` is a slot-based ROS2 decoded message instance."""
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _format_bytes_for_tree(
    value: bytes | bytearray | memoryview,
    *,
    bytes_mode: BytesMode,
    truncate_bytes: int,
) -> str:
    total = len(value)
    if bytes_mode == BytesMode.SKIP:
        return f"<{total} bytes>"
    if bytes_mode == BytesMode.BASE64:
        return json.dumps(base64.b64encode(bytes(value)).decode("ascii"))
    if bytes_mode == BytesMode.SMART:
        if total <= SMART_BYTES_INLINE_LIMIT:
            return repr(list(value))
        return f"<{total} bytes>"
    if truncate_bytes and total > truncate_bytes:
        return f"{list(value[:truncate_bytes])} ... ({total} bytes total)"
    return repr(list(value))


def _format_scalar(value: Any, *, bytes_mode: BytesMode, truncate_bytes: int) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _format_bytes_for_tree(value, bytes_mode=bytes_mode, truncate_bytes=truncate_bytes)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def _enum_label(value: Any, enum_field: EnumField) -> Text:
    name = enum_field.by_value.get(value)
    text = Text()
    text.append(str(value))
    if name is not None:
        text.append(" [", style="dim")
        text.append(name, style="bold magenta")
        text.append("]", style="dim")
    return text


def _key_label(name: str) -> Text:
    text = Text()
    text.append(name, style="bold cyan")
    text.append(": ", style="dim")
    return text


def _wrapper_value(value: Any, inner_field: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(inner_field)
    return getattr(value, inner_field, None)


def _add_sequence(
    parent: Tree,
    name: str,
    value: list[Any] | tuple[Any, ...],
    enum_field: EnumField | None,
    ctx: RenderContext,
) -> None:
    label = _key_label(name)
    label.append("[", style="dim")
    shown = list(value)[:_TREE_PRIMITIVE_ARRAY_LIMIT]
    for i, item in enumerate(shown):
        if i:
            label.append(", ", style="dim")
        if enum_field is None:
            label.append(ctx.format(item))
        else:
            label.append_text(_enum_label(item, enum_field))
    if len(value) > _TREE_PRIMITIVE_ARRAY_LIMIT:
        label.append(f", … ({len(value)} total)", style="dim")
    label.append("]", style="dim")
    parent.add(label)


def _render_child(
    parent: Tree,
    label: Text,
    obj: Any,
    plan: EnumPlan | None,
    ctx: RenderContext,
) -> None:
    _render_into(parent.add(label), obj, plan, ctx)


def _render_into(
    parent: Tree,
    obj: Any,
    plan: EnumPlan | None,
    ctx: RenderContext,
) -> None:
    """Populate `parent` with one node per (name, value) pair on `obj`."""
    if isinstance(obj, dict):
        items = [(str(k), v) for k, v in obj.items()]
    elif _is_message_obj(obj):
        items = _message_items(obj)
    else:
        parent.add(ctx.format(obj))
        return

    skip = plan.skip_fields if plan else frozenset()
    enum_fields = plan.enum_fields if plan else {}
    nested_plans = plan.nested_plans if plan else {}

    for name, value in items:
        if name in skip:
            continue

        enum_field = enum_fields.get(name)
        if enum_field is not None:
            enum_value = (
                _wrapper_value(value, enum_field.inner_field)
                if enum_field.inner_field is not None
                else value
            )
            if isinstance(enum_value, (list, tuple)):
                _add_sequence(parent, name, enum_value, enum_field, ctx)
            else:
                label = _key_label(name)
                label.append_text(_enum_label(enum_value, enum_field))
                parent.add(label)
            continue

        if _is_tree_obj(value):
            _render_child(
                parent,
                Text(name, style="bold cyan"),
                value,
                nested_plans.get(name),
                ctx,
            )
            continue

        if isinstance(value, (list, tuple)):
            if value and _is_tree_obj(value[0]):
                header = Text()
                header.append(name, style="bold cyan")
                header.append(f"  ({len(value)})", style="dim")
                child = parent.add(header)
                inner_plan = nested_plans.get(name)
                shown_items = list(value)[:_TREE_COMPLEX_ARRAY_LIMIT]
                for i, item in enumerate(shown_items):
                    _render_child(child, Text(f"[{i}]", style="dim"), item, inner_plan, ctx)
                if len(value) > _TREE_COMPLEX_ARRAY_LIMIT:
                    child.add(
                        Text(
                            f"… ({len(value) - _TREE_COMPLEX_ARRAY_LIMIT} more)",
                            style="dim",
                        )
                    )
                continue

            _add_sequence(parent, name, value, None, ctx)
            continue

        label = _key_label(name)
        label.append(ctx.format(value))
        parent.add(label)


def render_message_tree(
    obj: Any,
    plan: EnumPlan | None,
    *,
    title: Text,
    bytes_mode: BytesMode,
    truncate_bytes: int,
) -> Tree:
    """Render a decoded message as a Rich Tree, decorating known enum fields with names."""
    tree = Tree(title)
    _render_into(tree, obj, plan, RenderContext(bytes_mode, truncate_bytes))
    return tree
