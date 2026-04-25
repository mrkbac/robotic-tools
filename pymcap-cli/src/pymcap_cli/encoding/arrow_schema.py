"""Map ROS2 message definitions to pyarrow struct types.

Given an MCAP schema (name + bytes), returns a ``pa.StructType`` that matches
the message layout — with exact primitive widths (``uint8``, ``int16`` …) —
so pyarrow doesn't have to infer types from Python values. Used by
``export-duckdb`` to keep ``UTINYINT``/``REAL`` instead of promoting
everything to ``BIGINT``/``DOUBLE``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ros_parser import parse_schema_to_definitions
from ros_parser.models import Field, MessageDefinition, Type

if TYPE_CHECKING:
    import pyarrow as pa
    from small_mcap import Schema


def _ros_primitive_to_arrow(type_name: str) -> pa.DataType:
    import pyarrow as pa  # noqa: PLC0415

    mapping = {
        "bool": pa.bool_(),
        "byte": pa.uint8(),
        "char": pa.int8(),
        "int8": pa.int8(),
        "uint8": pa.uint8(),
        "int16": pa.int16(),
        "uint16": pa.uint16(),
        "int32": pa.int32(),
        "uint32": pa.uint32(),
        "int64": pa.int64(),
        "uint64": pa.uint64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "string": pa.string(),
        "wstring": pa.string(),
    }
    if type_name in mapping:
        return mapping[type_name]
    # ROS1 primitives `time` / `duration` — collapse sec+nanosec to native
    # arrow temporal types so SQL can filter/subtract directly.
    if type_name == "time":
        return pa.timestamp("ns")
    if type_name == "duration":
        return pa.duration("ns")
    raise KeyError(f"unknown ROS primitive type: {type_name}")


# ROS2 represents time/duration as nested messages rather than primitives. We
# detect them by canonical name and collapse to arrow temporal types so the
# treatment is symmetric with ROS1.
_TIMESTAMP_MESSAGE_NAMES: frozenset[str] = frozenset(
    {"builtin_interfaces/Time", "builtin_interfaces/msg/Time"}
)
_DURATION_MESSAGE_NAMES: frozenset[str] = frozenset(
    {"builtin_interfaces/Duration", "builtin_interfaces/msg/Duration"}
)


def _resolve_complex(t: Type, defs: dict[str, MessageDefinition]) -> MessageDefinition | None:
    candidates = [
        f"{t.package_name}/{t.type_name}" if t.package_name else t.type_name,
        f"{t.package_name}/msg/{t.type_name}" if t.package_name else None,
        t.type_name,
    ]
    for key in candidates:
        if key and key in defs:
            return defs[key]
    return None


def _type_to_arrow(t: Type, defs: dict[str, MessageDefinition]) -> pa.DataType:
    import pyarrow as pa  # noqa: PLC0415

    if t.is_primitive:
        inner = _ros_primitive_to_arrow(t.type_name)
    else:
        qualified = f"{t.package_name}/{t.type_name}" if t.package_name else t.type_name
        if qualified in _TIMESTAMP_MESSAGE_NAMES:
            inner = pa.timestamp("ns")
        elif qualified in _DURATION_MESSAGE_NAMES:
            inner = pa.duration("ns")
        else:
            nested = _resolve_complex(t, defs)
            # Fall back to opaque binary so we at least keep the column.
            inner = pa.binary() if nested is None else _message_to_struct(nested, defs)
    if t.is_array:
        # Fixed-size arrays (e.g. ``float64[9]``) → ``FixedSizeList<T, N>`` so
        # the DuckDB column carries the exact length. Bounded (``<=N``) and
        # dynamic arrays become variable-length ``List<T>``.
        if t.is_fixed_array and t.array_size is not None:
            return pa.list_(inner, list_size=int(t.array_size))
        return pa.list_(inner)
    return inner


def _message_to_struct(msg: MessageDefinition, defs: dict[str, MessageDefinition]) -> pa.StructType:
    import pyarrow as pa  # noqa: PLC0415

    fields: list[pa.Field] = []
    for f in msg.fields:
        if not isinstance(f, Field):
            continue
        try:
            fields.append(pa.field(f.name, _type_to_arrow(f.type, defs)))
        except (KeyError, ValueError):
            # Skip fields we can't resolve; pyarrow would reject them anyway.
            continue
    return pa.struct(fields)


class ArrowSchemaCache:
    """Cache ``pa.StructType`` per MCAP schema id.

    MCAP schema ids are stable for the lifetime of a reader, so caching by id
    avoids re-parsing the schema text for every message.
    """

    def __init__(self) -> None:
        self._cache: dict[int, pa.StructType | None] = {}

    def get(self, schema: Schema | None) -> pa.StructType | None:
        """Return the arrow struct type for *schema*, or ``None`` if unsupported."""
        if schema is None:
            return None
        sid = int(schema.id)
        if sid in self._cache:
            return self._cache[sid]
        if schema.encoding != "ros2msg":
            self._cache[sid] = None
            return None
        name = schema.name
        data = schema.data
        try:
            defs = parse_schema_to_definitions(name, data)
            root = defs.get(name) or defs.get(name.replace("/msg/", "/"))
            if root is None:
                self._cache[sid] = None
                return None
            struct = _message_to_struct(root, defs)
        except Exception:  # noqa: BLE001
            self._cache[sid] = None
            return None
        self._cache[sid] = struct
        return struct
