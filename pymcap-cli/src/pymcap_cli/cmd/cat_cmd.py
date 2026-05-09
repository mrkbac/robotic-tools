"""Cat command for pymcap-cli - stream MCAP messages to stdout."""

import base64
import dataclasses
import json
import logging
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from small_mcap import DecodedMessage

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import Schema
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    MessagePath,
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from ros_parser.models import Constant, MessageDefinition
from ros_parser.models import Type as RosType
from small_mcap import Channel, JSONDecoderFactory, read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.utils import MAX_INT64, ProgressTrackingIO, file_progress, parse_timestamp_args

logger = logging.getLogger(__name__)
console_out = Console()

FILTERING_GROUP = Group("Filtering")
OUTPUT_GROUP = Group("Output")

_TTY_BYTES_TRUNCATE = 32
# Threshold (bytes) below which `smart` mode inlines the full payload as an int
# list. Anything larger becomes a `<N bytes>` placeholder so large binary
# payloads (Image, PointCloud2) don't drown out the rest of the message.
_SMART_BYTES_INLINE_LIMIT = 64


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
            if total <= _SMART_BYTES_INLINE_LIMIT:
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
    nested_plans: dict[str, "EnumPlan"]


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


def _resolve_msgdef_by_name(
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

    msgdef = _resolve_msgdef_by_name(schema_name, all_definitions)
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
        if total <= _SMART_BYTES_INLINE_LIMIT:
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


def cat(
    file: str,
    *,
    topics: Annotated[
        list[str] | None,
        Parameter(
            name=["-t", "--topics"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    exclude_topics: Annotated[
        list[str] | None,
        Parameter(
            name=["-x", "--exclude-topics", "-n"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    query: Annotated[
        str | None,
        Parameter(
            name=["-q", "--query"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    grep: Annotated[
        str | None,
        Parameter(
            name=["-g", "--grep"],
            group=FILTERING_GROUP,
            help=(
                "Regex applied to every scalar value in the decoded message. "
                "Messages with no match are skipped. Bytes-like fields are not "
                "searched. Composes with --query: the regex runs on the post-"
                "query result so '--query <path> --grep <re>' scopes the search."
            ),
        ),
    ] = None,
    grep_ignore_case: Annotated[
        bool,
        Parameter(
            name=["-i", "--grep-ignore-case"],
            group=FILTERING_GROUP,
        ),
    ] = False,
    start: Annotated[
        str,
        Parameter(
            name=["-S", "--start"],
            group=FILTERING_GROUP,
        ),
    ] = "",
    start_secs: Annotated[
        int,
        Parameter(
            name=["-s", "--start-secs"],
            group=FILTERING_GROUP,
        ),
    ] = 0,
    end: Annotated[
        str,
        Parameter(
            name=["-E", "--end"],
            group=FILTERING_GROUP,
        ),
    ] = "",
    end_secs: Annotated[
        int,
        Parameter(
            name=["-e", "--end-secs"],
            group=FILTERING_GROUP,
        ),
    ] = 0,
    limit: Annotated[
        int | None,
        Parameter(
            name=["-l", "--limit"],
            group=OUTPUT_GROUP,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(
            name=["-o", "--output"],
            group=OUTPUT_GROUP,
        ),
    ] = None,
    bytes_mode: Annotated[
        BytesMode,
        Parameter(
            name=["--bytes"],
            group=OUTPUT_GROUP,
            help=(
                "How to render `bytes` fields in JSON output. `smart` (default) "
                f"inlines payloads ≤{_SMART_BYTES_INLINE_LIMIT} bytes as int lists "
                "and collapses larger ones to `<N bytes>` so `cat` stays readable "
                "on messages with Image/PointCloud2 payloads. Use `ints` for the "
                "full int list, `base64` for a compact serialisable string, or "
                "`skip` to always drop the payload."
            ),
        ),
    ] = BytesMode.SMART,
) -> int:
    """Stream MCAP messages to stdout.

    Decodes ROS2 messages and outputs as JSON. When output is to a terminal (TTY),
    displays messages in a Rich table. When piped, outputs JSONL (one JSON per line).

    Examples:
      # Display messages in a table (interactive)
      pymcap-cli cat recording.mcap

      # Pipe to file as JSONL
      pymcap-cli cat recording.mcap > messages.jsonl

      # Write to file with progress bar
      pymcap-cli cat recording.mcap -o messages.jsonl

      # Filter specific topics
      pymcap-cli cat recording.mcap --topics /camera/image

      # Filter by time range
      pymcap-cli cat recording.mcap --start-secs 10 --end-secs 20

      # Limit output
      pymcap-cli cat recording.mcap --limit 100

      # Query specific field using message path
      pymcap-cli cat recording.mcap --query '/odom.pose.position.x'

      # Filter array elements
      pymcap-cli cat recording.mcap --query '/detections.objects[:]{confidence>0.8}'

      # Skip binary data (images, pointclouds)
      pymcap-cli cat recording.mcap --bytes skip

      # Base64-encode binary data
      pymcap-cli cat recording.mcap --bytes base64
    """

    start_time_ns = parse_timestamp_args(start, 0, start_secs) or 0
    end_time_ns = parse_timestamp_args(end, 0, end_secs)
    end_time_ns = MAX_INT64 if end_time_ns is None else end_time_ns

    parsed_query = None
    if query:
        try:
            parsed_query = parse_message_path(query)
        except Exception:
            logger.exception("Invalid query syntax")
            return 1

    grep_pattern: re.Pattern[str] | None = None
    if grep:
        try:
            grep_pattern = re.compile(grep, re.IGNORECASE if grep_ignore_case else 0)
        except re.error:
            logger.exception("Invalid --grep regex")
            return 1

    try:
        topic_patterns = [re.compile(pattern) for pattern in topics] if topics else []
        exclude_topic_patterns = (
            [re.compile(pattern) for pattern in exclude_topics] if exclude_topics else []
        )
    except re.error:
        logger.exception("Invalid topic regex")
        return 1

    writing_to_file = output is not None
    is_tty = not writing_to_file and sys.stdout.isatty()

    message_count = 0
    validated_topics: set[str] = set()
    parsed_schemas: dict[int, dict[str, MessageDefinition] | None] = {}
    enum_plans: dict[int, EnumPlan | None] = {}

    def should_include_message(
        channel: Channel,
        _schema: Schema | None,
    ) -> bool:
        topic = channel.topic

        if parsed_query:
            return topic == parsed_query.topic and not any(
                p.search(topic) for p in exclude_topic_patterns
            )
        if topic_patterns and not any(p.search(topic) for p in topic_patterns):
            return False

        return not any(p.search(topic) for p in exclude_topic_patterns)

    def _get_parsed_schema(schema: Schema) -> dict[str, MessageDefinition] | None:
        if schema.id in parsed_schemas:
            return parsed_schemas[schema.id]
        try:
            parsed = parse_schema_to_definitions(schema.name, schema.data)
        except Exception:
            logger.exception(f"Failed to parse schema '{schema.name}'")
            parsed = None
        parsed_schemas[schema.id] = parsed
        return parsed

    def _get_enum_plan(schema: Schema) -> EnumPlan | None:
        if schema.id in enum_plans:
            return enum_plans[schema.id]
        parsed = _get_parsed_schema(schema)
        plan = build_enum_plan(schema.name, parsed) if parsed else None
        enum_plans[schema.id] = plan
        return plan

    def _validate_query(query_path: MessagePath, schema: Schema, topic: str) -> int | None:
        """Validate query against schema. Returns 1 on error, None on success."""
        all_definitions = _get_parsed_schema(schema)
        if all_definitions is None:
            return None
        try:
            root_msgdef = _resolve_msgdef_by_name(schema.name, all_definitions)
            if root_msgdef is None:
                logger.warning(f"Could not find message definition for schema '{schema.name}'")
            else:
                query_path.validate(root_msgdef, all_definitions)
        except ValidationError:
            logger.exception(f"Query validation error for topic '{topic}'")
            logger.exception(f"Query: {query}  Schema: {schema.name}")
            return 1
        return None

    def _to_jsonl(msg: "DecodedMessage", data: Any) -> str:
        entry: dict[str, Any] = {
            "topic": msg.channel.topic,
            "sequence": msg.message.sequence,
            "log_time": msg.message.log_time,
            "publish_time": msg.message.publish_time,
        }
        if msg.schema:
            entry["schema"] = msg.schema.name
        entry["message"] = message_to_dict(data, bytes_mode=bytes_mode)
        return json.dumps(entry, separators=(",", ":"))

    try:
        with open_input(file) as (input_stream, file_size), ExitStack() as stack:
            stream: IO[bytes] = input_stream
            if writing_to_file and file_size:
                progress = file_progress("[bold blue]Reading MCAP...")
                progress.start()
                stack.callback(progress.stop)
                task = progress.add_task("Processing", total=file_size)
                stream = ProgressTrackingIO(input_stream, task, progress, input_stream.tell())

            out_file = stack.enter_context(output.open("w")) if output else None

            for msg in read_message_decoded(
                stream,
                decoder_factories=[JSONDecoderFactory(), DecoderFactory()],
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                should_include=should_include_message,
            ):
                if limit is not None and message_count >= limit:
                    break

                # Validate query against schema on first message of each topic
                if parsed_query and msg.channel.topic not in validated_topics:
                    validated_topics.add(msg.channel.topic)

                    if msg.schema is None:
                        logger.warning(
                            f"Cannot validate query for topic '{msg.channel.topic}' "
                            "(no schema available)"
                        )
                    elif _validate_query(parsed_query, msg.schema, msg.channel.topic):
                        return 1

                # Apply query filter
                if parsed_query:
                    try:
                        data = parsed_query.apply(msg.decoded_message)
                        if data is None:
                            continue
                    except MessagePathError as e:
                        logger.warning(f"Filter error on {msg.channel.topic}: {e}")
                        continue
                else:
                    data = msg.decoded_message

                if grep_pattern is not None and not message_matches_grep(data, grep_pattern):
                    continue

                message_count += 1

                if is_tty:
                    schema = msg.schema
                    header = Text()
                    header.append(msg.channel.topic, style="bold cyan")
                    header.append(" @ ", style="dim")
                    header.append(str(msg.message.log_time), style="green")
                    header.append(" [", style="dim")
                    header.append(schema.name if schema else "unknown", style="yellow")
                    header.append("]", style="dim")

                    plan = None if parsed_query or schema is None else _get_enum_plan(schema)

                    tree = render_message_tree(
                        data,
                        plan,
                        title=header,
                        bytes_mode=bytes_mode,
                        truncate_bytes=_TTY_BYTES_TRUNCATE,
                    )

                    console_out.print(Panel(tree, border_style="blue", expand=False))
                else:
                    line = _to_jsonl(msg, data)
                    if out_file is not None:
                        out_file.write(line + "\n")
                    else:
                        print(line, file=sys.stdout)  # noqa: T201

        if writing_to_file:
            logger.info(f"Wrote {message_count:,} messages to {output}")

        if parsed_query and not validated_topics:
            logger.error(f"Topic '{parsed_query.topic}' not found in MCAP file")
            return 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0

    except Exception:
        logger.exception("Error reading MCAP")
        return 1

    return 0
