"""Micro-benchmarks targeting specific hot paths in generated decoder/encoder code.

These benchmarks isolate pure decode/encode performance without MCAP I/O overhead.
They focus on message types where the optimizations have the most impact:
- String-heavy messages (str() vs codecs.utf_8_decode, .encode() vs codecs.utf_8_encode)
- Complex array messages (pre-allocation vs append)
- Dict-input encoding (_get_field sentinel vs try/except)

Run with:
    uv run pytest tests/test_micro_benchmark.py --benchmark-only -v
"""

import pytest
from mcap_ros2_support_fast._planner import generate_plans, optimize_plan, serialize_dynamic
from mcap_ros2_support_fast._dynamic_decoder import create_decoder
from mcap_ros2_support_fast._dynamic_encoder import create_encoder

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

# 1. String-heavy: diagnostic_msgs/KeyValue (two string fields)
KEYVALUE_SCHEMA = "string key\nstring value"

# 2. String array: custom message with a dynamic string array
STRING_ARRAY_SCHEMA = "string[] labels"

# 3. Complex array: tf2_msgs/TFMessage — array of TransformStamped
TF_MESSAGE_SCHEMA = """\
geometry_msgs/TransformStamped[] transforms

================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation

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
float64 w"""

# 4. DiagnosticArray — complex array of DiagnosticStatus, each with string arrays + KeyValue arrays
DIAGNOSTIC_ARRAY_SCHEMA = """\
std_msgs/Header header
diagnostic_msgs/DiagnosticStatus[] status

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: diagnostic_msgs/DiagnosticStatus
byte level
string name
string message
string hardware_id
diagnostic_msgs/KeyValue[] entries

================================================================================
MSG: diagnostic_msgs/KeyValue
string key
string value"""

# 5. Simple string message (std_msgs/String)
STRING_SCHEMA = "string data"


# ---------------------------------------------------------------------------
# Fixture helpers — build encoder/decoder once, reuse across rounds
# ---------------------------------------------------------------------------


def _build_codec(schema_name: str, schema_text: str):
    """Return (decoder, encoder) for a schema."""
    plan = generate_plans(schema_name, schema_text)
    opt = optimize_plan(plan)
    return create_decoder(opt), create_encoder(opt)


# ---------------------------------------------------------------------------
# Pre-built codecs (module-level so pytest-benchmark measures only hot path)
# ---------------------------------------------------------------------------

_string_dec, _string_enc = _build_codec("std_msgs/String", STRING_SCHEMA)
_kv_dec, _kv_enc = _build_codec("custom/KeyValue", KEYVALUE_SCHEMA)
_str_arr_dec, _str_arr_enc = _build_codec("custom/StringArray", STRING_ARRAY_SCHEMA)
_tf_dec, _tf_enc = _build_codec("tf2_msgs/TFMessage", TF_MESSAGE_SCHEMA)
_diag_dec, _diag_enc = _build_codec(
    "diagnostic_msgs/DiagnosticArray", DIAGNOSTIC_ARRAY_SCHEMA
)


# ---------------------------------------------------------------------------
# Test data factories — produce (message_dict, cdr_bytes) pairs
# ---------------------------------------------------------------------------


def _make_string_msg(length: int = 50) -> dict:
    return {"data": "x" * length}


def _make_keyvalue(i: int = 0) -> dict:
    return {"key": f"sensor_{i}_temperature", "value": f"{20.0 + i * 0.5:.2f}"}


def _make_string_array(n: int) -> dict:
    return {"labels": [f"item_{i}_with_some_text" for i in range(n)]}


def _make_transform(i: int = 0) -> dict:
    return {
        "header": {
            "stamp": {"sec": 1700000000 + i, "nanosec": i * 1000},
            "frame_id": f"world_{i}",
        },
        "child_frame_id": f"sensor_link_{i}",
        "transform": {
            "translation": {"x": 1.0 + i, "y": 2.0 + i, "z": 3.0 + i},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }


def _make_tf_message(n: int) -> dict:
    return {"transforms": [_make_transform(i) for i in range(n)]}


def _make_diagnostic_status(i: int, n_values: int) -> dict:
    return {
        "level": 0,
        "name": f"component_{i}_health_monitor",
        "message": f"Status report for component {i}: all systems nominal",
        "hardware_id": f"hw-{i:04d}-rev-b",
        "entries": [_make_keyvalue(j) for j in range(n_values)],
    }


def _make_diagnostic_array(n_status: int, n_values: int) -> dict:
    return {
        "header": {
            "stamp": {"sec": 1700000000, "nanosec": 500000},
            "frame_id": "diagnostics",
        },
        "status": [_make_diagnostic_status(i, n_values) for i in range(n_status)],
    }


# Pre-encode CDR bytes for decode benchmarks
_string_cdr = bytes(_string_enc(_make_string_msg()))
_kv_cdr = bytes(_kv_enc(_make_keyvalue()))
_str_arr_10_cdr = bytes(_str_arr_enc(_make_string_array(10)))
_str_arr_100_cdr = bytes(_str_arr_enc(_make_string_array(100)))
_tf_5_cdr = bytes(_tf_enc(_make_tf_message(5)))
_tf_50_cdr = bytes(_tf_enc(_make_tf_message(50)))
_diag_5x5_cdr = bytes(_diag_enc(_make_diagnostic_array(5, 5)))
_diag_10x10_cdr = bytes(_diag_enc(_make_diagnostic_array(10, 10)))

# Pre-build message dicts for encode benchmarks
_string_msg = _make_string_msg()
_kv_msg = _make_keyvalue()
_str_arr_10_msg = _make_string_array(10)
_str_arr_100_msg = _make_string_array(100)
_tf_5_msg = _make_tf_message(5)
_tf_50_msg = _make_tf_message(50)
_diag_5x5_msg = _make_diagnostic_array(5, 5)
_diag_10x10_msg = _make_diagnostic_array(10, 10)


# ===========================================================================
# DECODE benchmarks
# ===========================================================================


@pytest.mark.benchmark(group="decode-string")
@pytest.mark.parametrize(
    ("name", "decoder", "cdr"),
    [
        pytest.param("String", _string_dec, _string_cdr, id="String"),
        pytest.param("KeyValue", _kv_dec, _kv_cdr, id="KeyValue"),
        pytest.param("string[10]", _str_arr_dec, _str_arr_10_cdr, id="string_arr_10"),
        pytest.param("string[100]", _str_arr_dec, _str_arr_100_cdr, id="string_arr_100"),
    ],
)
def test_decode_strings(benchmark, name, decoder, cdr):
    """Benchmark string decoding: str(buf, 'utf-8') vs codecs.utf_8_decode."""
    benchmark(decoder, cdr)


@pytest.mark.benchmark(group="decode-complex-array")
@pytest.mark.parametrize(
    ("name", "decoder", "cdr"),
    [
        pytest.param("TF x5", _tf_dec, _tf_5_cdr, id="TF_5"),
        pytest.param("TF x50", _tf_dec, _tf_50_cdr, id="TF_50"),
        pytest.param("Diag 5x5", _diag_dec, _diag_5x5_cdr, id="Diag_5x5"),
        pytest.param("Diag 10x10", _diag_dec, _diag_10x10_cdr, id="Diag_10x10"),
    ],
)
def test_decode_complex_arrays(benchmark, name, decoder, cdr):
    """Benchmark complex array decoding: [None]*N + index vs append."""
    benchmark(decoder, cdr)


# ===========================================================================
# ENCODE benchmarks
# ===========================================================================


@pytest.mark.benchmark(group="encode-string")
@pytest.mark.parametrize(
    ("name", "encoder", "msg"),
    [
        pytest.param("String", _string_enc, _string_msg, id="String"),
        pytest.param("KeyValue", _kv_enc, _kv_msg, id="KeyValue"),
        pytest.param("string[10]", _str_arr_enc, _str_arr_10_msg, id="string_arr_10"),
        pytest.param("string[100]", _str_arr_enc, _str_arr_100_msg, id="string_arr_100"),
    ],
)
def test_encode_strings(benchmark, name, encoder, msg):
    """Benchmark string encoding: .encode() vs codecs.utf_8_encode."""
    benchmark(encoder, msg)


@pytest.mark.benchmark(group="encode-complex-array")
@pytest.mark.parametrize(
    ("name", "encoder", "msg"),
    [
        pytest.param("TF x5", _tf_enc, _tf_5_msg, id="TF_5"),
        pytest.param("TF x50", _tf_enc, _tf_50_msg, id="TF_50"),
        pytest.param("Diag 5x5", _diag_enc, _diag_5x5_msg, id="Diag_5x5"),
        pytest.param("Diag 10x10", _diag_enc, _diag_10x10_msg, id="Diag_10x10"),
    ],
)
def test_encode_complex_arrays(benchmark, name, encoder, msg):
    """Benchmark complex array encoding with dict inputs (_get_field sentinel)."""
    benchmark(encoder, msg)


# ===========================================================================
# ENCODE with dataclass inputs (tests _get_field fast path)
# ===========================================================================

# Decode CDR to get dataclass instances, then re-encode them
_string_dc = _string_dec(_string_cdr)
_kv_dc = _kv_dec(_kv_cdr)
_str_arr_10_dc = _str_arr_dec(_str_arr_10_cdr)
_str_arr_100_dc = _str_arr_dec(_str_arr_100_cdr)
_tf_5_dc = _tf_dec(_tf_5_cdr)
_tf_50_dc = _tf_dec(_tf_50_cdr)
_diag_5x5_dc = _diag_dec(_diag_5x5_cdr)
_diag_10x10_dc = _diag_dec(_diag_10x10_cdr)


@pytest.mark.benchmark(group="encode-dataclass")
@pytest.mark.parametrize(
    ("name", "encoder", "msg"),
    [
        pytest.param("String", _string_enc, _string_dc, id="String"),
        pytest.param("KeyValue", _kv_enc, _kv_dc, id="KeyValue"),
        pytest.param("string[10]", _str_arr_enc, _str_arr_10_dc, id="string_arr_10"),
        pytest.param("string[100]", _str_arr_enc, _str_arr_100_dc, id="string_arr_100"),
        pytest.param("TF x5", _tf_enc, _tf_5_dc, id="TF_5"),
        pytest.param("TF x50", _tf_enc, _tf_50_dc, id="TF_50"),
        pytest.param("Diag 5x5", _diag_enc, _diag_5x5_dc, id="Diag_5x5"),
        pytest.param("Diag 10x10", _diag_enc, _diag_10x10_dc, id="Diag_10x10"),
    ],
)
def test_encode_dataclass(benchmark, name, encoder, msg):
    """Benchmark encoding from dataclass inputs (_get_field getattr path)."""
    benchmark(encoder, msg)
