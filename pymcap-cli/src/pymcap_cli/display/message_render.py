"""Shared decoded-message rendering helpers.

Used by both `pymcap-cli cat` (file streaming) and `pymcap-cli bridge cat`
(live bridge streaming) so the two commands stay byte-for-byte consistent.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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


def format_bytes_skip(value: bytes | bytearray | memoryview) -> str:
    return f"<{len(value)} bytes>"


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
            return format_bytes_skip(obj)
        if bytes_mode == BytesMode.BASE64:
            return base64.b64encode(bytes(obj)).decode("ascii")
        if bytes_mode == BytesMode.SMART:
            # Small payloads inline as ints for easy inspection; large ones
            # collapse to a placeholder so output stays grep/jq-friendly.
            if total <= SMART_BYTES_INLINE_LIMIT:
                return list(obj)
            return format_bytes_skip(obj)
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

_TIME_PACKAGE = "builtin_interfaces"
_QUATERNION_PACKAGE = "geometry_msgs"
_QUATERNION_TYPE = "Quaternion"


class TimeKind(str, Enum):
    """Kind of `builtin_interfaces` timestamp field to annotate in the tree."""

    TIME = "Time"
    DURATION = "Duration"


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
    time_fields: dict[str, TimeKind] = dataclasses.field(default_factory=dict)
    quaternion_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RenderContext:
    bytes_mode: BytesMode
    truncate_bytes: int
    # Dotted paths whose value changed since the previous message on this topic
    # (``--changed``). None disables change highlighting.
    changed_paths: frozenset[str] | None = None

    def format(self, value: Any) -> str:
        return _format_scalar(value, bytes_mode=self.bytes_mode, truncate_bytes=self.truncate_bytes)

    def is_changed(self, path: str) -> bool:
        return self.changed_paths is not None and path in self.changed_paths


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
    time_fields: dict[str, TimeKind] = {}
    quaternion_fields: set[str] = set()

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

        # builtin_interfaces/Time and /Duration: keep the sec/nanosec breakdown but
        # tag the field so the renderer can annotate it with a human-readable value.
        # Detected on the declared type so it works even without their MessageDefinition.
        time_kind = _time_kind(f.type)
        if time_kind is not None and not f.type.is_array:
            time_fields[f.name] = time_kind
            continue

        # geometry_msgs/Quaternion: keep the x/y/z/w breakdown but tag the field so
        # the renderer can annotate it with human-readable roll/pitch/yaw.
        if _is_quaternion_type(f.type) and not f.type.is_array:
            quaternion_fields.add(f.name)
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

    if not (skip_fields or enum_fields or nested_plans or time_fields or quaternion_fields):
        _visited[schema_name] = None
        return None

    plan = EnumPlan(
        skip_fields=frozenset(skip_fields),
        enum_fields=dict(enum_fields),
        nested_plans=dict(nested_plans),
        time_fields=dict(time_fields),
        quaternion_fields=frozenset(quaternion_fields),
    )
    _visited[schema_name] = plan
    return plan


def _time_kind(field_type: RosType) -> TimeKind | None:
    """Return the TimeKind for a `builtin_interfaces/{Time,Duration}` field type, else None."""
    if field_type.package_name != _TIME_PACKAGE:
        return None
    try:
        return TimeKind(field_type.type_name)
    except ValueError:
        return None


def _is_quaternion_type(field_type: RosType) -> bool:
    """True for a `geometry_msgs/Quaternion` field type."""
    return (
        field_type.package_name == _QUATERNION_PACKAGE and field_type.type_name == _QUATERNION_TYPE
    )


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
        return format_bytes_skip(value)
    if bytes_mode == BytesMode.BASE64:
        return json.dumps(base64.b64encode(bytes(value)).decode("ascii"))
    if bytes_mode == BytesMode.SMART:
        if total <= SMART_BYTES_INLINE_LIMIT:
            return repr(list(value))
        return format_bytes_skip(value)
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


def _time_parts(value: Any) -> tuple[int, int] | None:
    """Extract integer (sec, nanosec) from a decoded Time/Duration message or dict."""
    if isinstance(value, dict):
        sec, nanosec = value.get("sec"), value.get("nanosec")
    elif _is_message_obj(value):
        fields = dict(_message_items(value))
        sec, nanosec = fields.get("sec"), fields.get("nanosec")
    else:
        return None
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        return None
    if isinstance(sec, int) and isinstance(nanosec, int):
        return sec, nanosec
    return None


def _format_ros_time(sec: int, nanosec: int) -> str:
    """UTC ISO-8601 timestamp with nanosecond precision for a ROS `Time`."""
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{nanosec:09d}Z"


def _format_ros_duration(sec: int, nanosec: int) -> str:
    """Signed seconds string with nanosecond precision for a ROS `Duration`."""
    total_ns = sec * 1_000_000_000 + nanosec
    sign = "-" if total_ns < 0 else ""
    magnitude = abs(total_ns)
    return f"{sign}{magnitude // 1_000_000_000}.{magnitude % 1_000_000_000:09d}s"


def _time_annotation(value: Any, kind: TimeKind) -> Text | None:
    """Human-readable annotation for a Time/Duration field, or None if unrenderable."""
    parts = _time_parts(value)
    if parts is None:
        return None
    sec, nanosec = parts
    try:
        if kind is TimeKind.TIME:
            text = f"{_format_ros_time(sec, nanosec)} UTC"
        else:
            text = _format_ros_duration(sec, nanosec)
    except (OverflowError, OSError, ValueError):
        return None
    return Text(text, style="green")


def _as_float(value: Any) -> float | None:
    """Return `value` as a float if it is a real number (not bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _quaternion_parts(value: Any) -> tuple[float, float, float, float] | None:
    """Extract numeric (x, y, z, w) from a decoded Quaternion message or dict."""
    if isinstance(value, dict):
        source = value
    elif _is_message_obj(value):
        source = dict(_message_items(value))
    else:
        return None
    x = _as_float(source.get("x"))
    y = _as_float(source.get("y"))
    z = _as_float(source.get("z"))
    w = _as_float(source.get("w"))
    if x is None or y is None or z is None or w is None:
        return None
    return x, y, z, w


def _quaternion_to_rpy_deg(
    x: float, y: float, z: float, w: float
) -> tuple[float, float, float] | None:
    """Convert a quaternion to (roll, pitch, yaw) in degrees, or None if degenerate."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _quaternion_annotation(value: Any) -> Text | None:
    """`rpy [r°, p°, y°]` annotation for a Quaternion field, or None if unrenderable."""
    parts = _quaternion_parts(value)
    if parts is None:
        return None
    rpy = _quaternion_to_rpy_deg(*parts)
    if rpy is None:
        return None
    roll, pitch, yaw = rpy
    return Text(f"rpy [{roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°]", style="green")


def _wrapper_value(value: Any, inner_field: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(inner_field)
    return getattr(value, inner_field, None)


def _scalar_text(value: Any, ctx: RenderContext, path: str = "") -> Text:
    """Formatted scalar as Rich `Text`.

    Non-finite floats (NaN/±Inf) render bold red; a value that changed since the
    previous message (``--changed``) renders bold yellow. NaN highlighting wins.
    """
    formatted = ctx.format(value)
    if isinstance(value, float) and not math.isfinite(value):
        return Text(formatted, style="bold red")
    if ctx.is_changed(path):
        return Text(formatted, style="bold yellow")
    return Text(formatted)


def _sequence_body(
    value: list[Any] | tuple[Any, ...],
    enum_field: EnumField | None,
    ctx: RenderContext,
) -> Text:
    """Render a scalar sequence inline as ``[a, b, c]`` (truncated past the limit)."""
    label = Text("[", style="dim")
    shown = list(value)[:_TREE_PRIMITIVE_ARRAY_LIMIT]
    for i, item in enumerate(shown):
        if i:
            label.append(", ", style="dim")
        if enum_field is None:
            label.append_text(_scalar_text(item, ctx))
        else:
            label.append_text(_enum_label(item, enum_field))
    if len(value) > _TREE_PRIMITIVE_ARRAY_LIMIT:
        label.append(f", … ({len(value)} total)", style="dim")
    label.append("]", style="dim")
    return label


def _apply_changed(text: Text, ctx: RenderContext, path: str) -> Text:
    if ctx.is_changed(path):
        text.stylize("bold yellow")
    return text


def _add_sequence(
    parent: Tree,
    name: str,
    value: list[Any] | tuple[Any, ...],
    enum_field: EnumField | None,
    ctx: RenderContext,
    path: str = "",
) -> None:
    label = _key_label(name)
    label.append_text(_apply_changed(_sequence_body(value, enum_field, ctx), ctx, path))
    parent.add(label)


def _render_message_array(
    parent: Tree,
    value: list[Any] | tuple[Any, ...],
    element_plan: EnumPlan | None,
    ctx: RenderContext,
    path: str = "",
) -> None:
    """Render a sequence of message/dict elements as ``[i]`` subtrees under ``parent``."""
    shown = list(value)[:_TREE_COMPLEX_ARRAY_LIMIT]
    for i, item in enumerate(shown):
        _render_child(parent, Text(f"[{i}]", style="dim"), item, element_plan, ctx, f"{path}[{i}]")
    remaining = len(value) - _TREE_COMPLEX_ARRAY_LIMIT
    if remaining > 0:
        parent.add(Text(f"… ({remaining} more)", style="dim"))


def _render_child(
    parent: Tree,
    label: Text,
    obj: Any,
    plan: EnumPlan | None,
    ctx: RenderContext,
    path: str = "",
) -> None:
    _render_into(parent.add(label), obj, plan, ctx, path)


def _render_into(
    parent: Tree,
    obj: Any,
    plan: EnumPlan | None,
    ctx: RenderContext,
    path: str = "",
) -> None:
    """Populate `parent` with one node per (name, value) pair on `obj`."""
    if isinstance(obj, dict):
        items = [(str(k), v) for k, v in obj.items()]
    elif _is_message_obj(obj):
        items = _message_items(obj)
    elif isinstance(obj, (list, tuple)):
        # A query that extracts an array (e.g. `/tf.transforms` or `.transforms[:]`)
        # hands us a bare sequence; render its elements rather than a raw repr.
        if obj and _is_tree_obj(obj[0]):
            _render_message_array(parent, obj, plan, ctx, path)
        else:
            parent.add(_apply_changed(_sequence_body(obj, None, ctx), ctx, path))
        return
    else:
        parent.add(_scalar_text(obj, ctx, path))
        return

    skip = plan.skip_fields if plan else frozenset()
    enum_fields = plan.enum_fields if plan else {}
    nested_plans = plan.nested_plans if plan else {}
    time_fields = plan.time_fields if plan else {}
    quaternion_fields = plan.quaternion_fields if plan else frozenset()

    for name, value in items:
        if name in skip:
            continue

        child_path = f"{path}.{name}" if path else name

        enum_field = enum_fields.get(name)
        if enum_field is not None:
            enum_value = (
                _wrapper_value(value, enum_field.inner_field)
                if enum_field.inner_field is not None
                else value
            )
            if isinstance(enum_value, (list, tuple)):
                _add_sequence(parent, name, enum_value, enum_field, ctx, child_path)
            else:
                label = _key_label(name)
                label.append_text(
                    _apply_changed(_enum_label(enum_value, enum_field), ctx, child_path)
                )
                parent.add(label)
            continue

        if _is_tree_obj(value):
            header = Text(name, style="bold cyan")
            time_kind = time_fields.get(name)
            if time_kind is not None:
                annotation = _time_annotation(value, time_kind)
            elif name in quaternion_fields:
                annotation = _quaternion_annotation(value)
            else:
                annotation = None
            if annotation is not None:
                header.append("  ")
                header.append_text(annotation)
            _render_child(parent, header, value, nested_plans.get(name), ctx, child_path)
            continue

        if isinstance(value, (list, tuple)):
            if value and _is_tree_obj(value[0]):
                header = Text()
                header.append(name, style="bold cyan")
                header.append(f"  ({len(value)})", style="dim")
                child = parent.add(header)
                _render_message_array(child, value, nested_plans.get(name), ctx, child_path)
                continue

            _add_sequence(parent, name, value, None, ctx, child_path)
            continue

        label = _key_label(name)
        label.append_text(_scalar_text(value, ctx, child_path))
        parent.add(label)


def render_message_tree(
    obj: Any,
    plan: EnumPlan | None,
    *,
    title: Text,
    bytes_mode: BytesMode,
    truncate_bytes: int,
    changed_paths: frozenset[str] | None = None,
) -> Tree:
    """Render a decoded message as a Rich Tree, decorating known enum fields with names."""
    tree = Tree(title)
    _render_into(tree, obj, plan, RenderContext(bytes_mode, truncate_bytes, changed_paths))
    return tree


def _flat_leaf(path: str, value_text: Text) -> Text:
    if not path:
        return value_text
    line = Text()
    line.append(path, style="bold cyan")
    line.append(": ", style="dim")
    line.append_text(value_text)
    return line


def _flat_lines(value: Any, plan: EnumPlan | None, path: str, ctx: RenderContext) -> list[Text]:
    if isinstance(value, (list, tuple)):
        if value and _is_tree_obj(value[0]):
            lines: list[Text] = []
            shown = list(value)[:_TREE_COMPLEX_ARRAY_LIMIT]
            for i, item in enumerate(shown):
                lines.extend(_flat_lines(item, plan, f"{path}[{i}]", ctx))
            remaining = len(value) - _TREE_COMPLEX_ARRAY_LIMIT
            if remaining > 0:
                lines.append(_flat_leaf(f"{path}[…]", Text(f"({remaining} more)", style="dim")))
            return lines
        return [_flat_leaf(path, _apply_changed(_sequence_body(value, None, ctx), ctx, path))]

    if not _is_tree_obj(value):
        return [_flat_leaf(path, _scalar_text(value, ctx, path))]

    if isinstance(value, dict):
        items = [(str(k), v) for k, v in value.items()]
    else:
        items = _message_items(value)

    skip = plan.skip_fields if plan else frozenset()
    enum_fields = plan.enum_fields if plan else {}
    nested_plans = plan.nested_plans if plan else {}

    lines = []
    for name, child_value in items:
        if name in skip:
            continue
        child_path = f"{path}.{name}" if path else name
        enum_field = enum_fields.get(name)
        if enum_field is not None:
            enum_value = (
                _wrapper_value(child_value, enum_field.inner_field)
                if enum_field.inner_field is not None
                else child_value
            )
            if isinstance(enum_value, (list, tuple)):
                body = _apply_changed(_sequence_body(enum_value, enum_field, ctx), ctx, child_path)
            else:
                body = _apply_changed(_enum_label(enum_value, enum_field), ctx, child_path)
            lines.append(_flat_leaf(child_path, body))
            continue
        lines.extend(_flat_lines(child_value, nested_plans.get(name), child_path, ctx))
    return lines


def render_message_flat(
    obj: Any,
    plan: EnumPlan | None,
    *,
    bytes_mode: BytesMode,
    truncate_bytes: int,
    changed_paths: frozenset[str] | None = None,
) -> list[Text]:
    """Render a decoded message as flat ``dotted.path: value`` lines (one per leaf).

    Honors the same enum decoration and NaN/Inf highlighting as the tree view.
    Pairs well with ``--query`` to pull a subtree and list its leaves compactly.
    """
    return _flat_lines(obj, plan, "", RenderContext(bytes_mode, truncate_bytes, changed_paths))


_MISSING = object()


def _diff_leaves(prev: Any, cur: Any, path: str, out: set[str]) -> None:
    prev_fields = _field_map(prev)
    cur_fields = _field_map(cur)
    if prev_fields is not None and cur_fields is not None:
        for name in set(prev_fields) | set(cur_fields):
            child_path = f"{path}.{name}" if path else name
            _diff_leaves(
                prev_fields.get(name, _MISSING), cur_fields.get(name, _MISSING), child_path, out
            )
        return

    if isinstance(prev, (list, tuple)) and isinstance(cur, (list, tuple)):
        prev_has_msgs = bool(prev) and _is_tree_obj(prev[0])
        cur_has_msgs = bool(cur) and _is_tree_obj(cur[0])
        if prev_has_msgs or cur_has_msgs:
            for i in range(max(len(prev), len(cur))):
                prev_item = prev[i] if i < len(prev) else _MISSING
                cur_item = cur[i] if i < len(cur) else _MISSING
                _diff_leaves(prev_item, cur_item, f"{path}[{i}]", out)
            return
        # Scalar array: compare as a whole; highlight the field if anything moved.
        if list(prev) != list(cur):
            out.add(path)
        return

    if prev is _MISSING or cur is _MISSING or prev != cur:
        out.add(path)


def _field_map(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    if _is_message_obj(obj):
        return dict(_message_items(obj))
    return None


def changed_leaf_paths(previous: Any, current: Any) -> frozenset[str]:
    """Dotted paths whose scalar value differs between two decoded messages.

    Paths use the same convention as the flat renderer (``a.b``, ``a[0].b``), so the
    result can drive ``--changed`` highlighting in either the tree or flat view.
    """
    out: set[str] = set()
    _diff_leaves(previous, current, "", out)
    return frozenset(out)
