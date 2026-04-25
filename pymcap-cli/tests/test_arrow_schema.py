"""Tests for ArrowSchemaCache — ROS2 schemas → pyarrow struct types."""

from __future__ import annotations

import pyarrow as pa
from pymcap_cli.encoding.arrow_schema import ArrowSchemaCache
from small_mcap import Schema

_SCALAR_MSG = """\
bool ok
uint8 small_u
int8 small_s
uint16 medium_u
int32 counter
uint64 big
float32 temp
float64 precise
string label"""


def test_scalar_primitives_keep_exact_widths() -> None:
    cache = ArrowSchemaCache()
    schema = Schema(1, "pkg/msg/Scalars", "ros2msg", _SCALAR_MSG.encode())
    struct = cache.get(schema)
    assert struct is not None
    fields = {struct.field(i).name: struct.field(i).type for i in range(struct.num_fields)}
    assert fields["ok"] == pa.bool_()
    assert fields["small_u"] == pa.uint8()
    assert fields["small_s"] == pa.int8()
    assert fields["medium_u"] == pa.uint16()
    assert fields["counter"] == pa.int32()
    assert fields["big"] == pa.uint64()
    assert fields["temp"] == pa.float32()
    assert fields["precise"] == pa.float64()
    assert fields["label"] == pa.string()


_DYNAMIC_ARRAY_MSG = """\
float32[] ranges
string[] ids"""


def test_dynamic_arrays_become_list() -> None:
    cache = ArrowSchemaCache()
    struct = cache.get(Schema(2, "pkg/msg/Arr", "ros2msg", _DYNAMIC_ARRAY_MSG.encode()))
    assert struct is not None
    types = {struct.field(i).name: struct.field(i).type for i in range(struct.num_fields)}
    assert types["ranges"] == pa.list_(pa.float32())
    assert types["ids"] == pa.list_(pa.string())


_FIXED_ARRAY_MSG = """\
float64[36] covariance
uint8[4] ipv4"""


def test_fixed_arrays_become_fixed_size_list() -> None:
    cache = ArrowSchemaCache()
    struct = cache.get(Schema(3, "pkg/msg/Fixed", "ros2msg", _FIXED_ARRAY_MSG.encode()))
    assert struct is not None
    types = {struct.field(i).name: struct.field(i).type for i in range(struct.num_fields)}
    assert types["covariance"] == pa.list_(pa.float64(), list_size=36)
    assert types["ipv4"] == pa.list_(pa.uint8(), list_size=4)


_NESTED_MSG = """\
std_msgs/Header header
geometry_msgs/Point position

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z"""


def test_nested_messages_become_struct_with_timestamp_stamp() -> None:
    cache = ArrowSchemaCache()
    struct = cache.get(Schema(4, "pkg/msg/Pose", "ros2msg", _NESTED_MSG.encode()))
    assert struct is not None
    header = next(
        struct.field(i).type for i in range(struct.num_fields) if struct.field(i).name == "header"
    )
    assert pa.types.is_struct(header)
    # builtin_interfaces/Time is collapsed to timestamp(ns), not struct<sec, nanosec>.
    stamp = next(
        header.field(i).type for i in range(header.num_fields) if header.field(i).name == "stamp"
    )
    assert stamp == pa.timestamp("ns")


_TIME_ONLY_MSG = """\
builtin_interfaces/Time start
builtin_interfaces/Duration elapsed"""


def test_time_and_duration_map_to_native_temporal_types() -> None:
    cache = ArrowSchemaCache()
    struct = cache.get(Schema(7, "pkg/msg/Window", "ros2msg", _TIME_ONLY_MSG.encode()))
    assert struct is not None
    types = {struct.field(i).name: struct.field(i).type for i in range(struct.num_fields)}
    assert types["start"] == pa.timestamp("ns")
    assert types["elapsed"] == pa.duration("ns")


def test_cache_hits_on_repeat_calls() -> None:
    cache = ArrowSchemaCache()
    s = Schema(5, "pkg/msg/Scalars", "ros2msg", _SCALAR_MSG.encode())
    a = cache.get(s)
    b = cache.get(s)
    assert a is b


def test_non_ros2_schema_returns_none() -> None:
    cache = ArrowSchemaCache()
    assert cache.get(Schema(6, "whatever", "jsonschema", b"{}")) is None
    assert cache.get(None) is None
