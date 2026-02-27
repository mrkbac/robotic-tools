"""Benchmark: Phase 2 optimizations for decoder & encoder.

Measures the combined impact of:
  Decoder: Inline LE dispatcher (Opt 1), Skip memoryview (Opt 2),
           Static offset tracking, Return directly (Opt 5)
  Encoder: Simplified _get_field (Opt 3), Static offset tracking (Opt 4)

Usage:
    uv run python scripts/bench_split_decode.py
"""

import struct
import time
from collections.abc import Callable
from typing import Any

from mcap_ros2_support_fast._dynamic_decoder import DecoderGeneratorFactory, create_decoder
from mcap_ros2_support_fast._dynamic_encoder import EncoderGeneratorFactory, create_encoder
from mcap_ros2_support_fast._planner import generate_plans, optimize_plan

# ruff: noqa: T201

COMMAND_MSG_SCHEMA = """\
std_msgs/Header header
float64 velocity
float64 steering
float64 arm
float64 shovel
bool handbreak
bool safety
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
"""


def build_cdr_payload(frame_id: str = "base_link") -> bytes:
    """Build a CDR-encoded CommandMsg payload."""
    payload = bytearray()
    # CDR header (little-endian)
    payload += b"\x00\x01\x00\x00"
    # Time: sec (int32) + nanosec (uint32)
    payload += struct.pack("<iI", 1700000000, 500000)
    # frame_id: uint32 length (including null) + data + null
    frame_bytes = frame_id.encode("utf-8")
    payload += struct.pack("<I", len(frame_bytes) + 1)
    payload += frame_bytes + b"\x00"
    # Align to 8 bytes for float64 (offset from start of data, after CDR header)
    data_offset = len(payload) - 4  # subtract CDR header
    while data_offset % 8 != 0:
        payload += b"\x00"
        data_offset += 1
    # 4x float64 + 2x bool
    payload += struct.pack("<4d2?", 1.5, -0.3, 0.0, 2.7, True, False)
    return bytes(payload)


def make_decoder_namespace(opt_plan: tuple) -> dict[str, Any]:
    """Build a shared exec namespace with struct patterns and message classes."""
    _, fields = opt_plan
    header_plan = fields[0].plan
    time_class = header_plan[1][0].plan[0]
    return {
        "__builtins__": {"memoryview": memoryview, "str": str},
        "_d_le_iI": struct.Struct("<iI").unpack_from,
        "_d_le_I": struct.Struct("<I").unpack_from,
        "_d_le_ddddboolbool": struct.Struct("<4d2?").unpack_from,
        "builtin_interfaces_Time": time_class,
        "std_msgs_Header": header_plan[0],
        "custom_type_CommandMsg": opt_plan[0],
    }


# Old-style generated code: _offset tracking throughout, memoryview, dispatcher overhead
DECODER_BASELINE_CODE = """\
def decoder_baseline(_raw):
    _data = memoryview(_raw)[4:]
    _offset = 0
    _v3 = builtin_interfaces_Time(*_d_le_iI(_data, _offset))
    _offset += 8
    _str_size, = _d_le_I(_data, _offset)
    _offset += 4
    if _str_size > 1:
        _v4 = str(_data[_offset:_offset + _str_size - 1], "utf-8")
    else:
        _v4 = ""
    _offset += _str_size
    _v2 = std_msgs_Header(_v3, _v4)
    _offset = (_offset + 7) & ~7
    _v5, _v6, _v7, _v8, _v9, _v10 = _d_le_ddddboolbool(_data, _offset)
    _offset += 34
    _v1 = custom_type_CommandMsg(_v2, _v5, _v6, _v7, _v8, _v9, _v10)
    return _v1
"""


def bench(
    func: Callable[[bytes], object], data: bytes, iterations: int = 500_000, repeats: int = 7
) -> float:
    """Run benchmark and return best average time in nanoseconds."""
    for _ in range(10_000):
        func(data)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            func(data)
        elapsed = time.perf_counter_ns() - start
        avg = elapsed / iterations
        best = min(best, avg)
    return best


def bench_encoder(
    func: Callable[[object], object], msg: object, iterations: int = 500_000, repeats: int = 7
) -> float:
    """Run encoder benchmark and return best average time in nanoseconds."""
    for _ in range(10_000):
        func(msg)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            func(msg)
        elapsed = time.perf_counter_ns() - start
        avg = elapsed / iterations
        best = min(best, avg)
    return best


def main() -> None:
    print("=" * 70)
    print("Phase 2 Benchmark: Decoder & Encoder Optimizations")
    print("=" * 70)

    # Build test data
    cdr_short = build_cdr_payload("b")
    cdr_medium = build_cdr_payload("base_link")
    cdr_long = build_cdr_payload("very_long_frame_id_string_that_tests_alignment")

    # Build optimized plan
    plan = generate_plans("custom_type/CommandMsg", COMMAND_MSG_SCHEMA)
    opt_plan = optimize_plan(plan)

    # ─── DECODER BENCHMARK ──────────────────────────────────────────────
    print("\n── Decoder ─────────────────────────────────────────────────────")

    # Build baseline (old-style) decoder via exec
    base_ns = make_decoder_namespace(opt_plan)
    exec(DECODER_BASELINE_CODE, base_ns)  # noqa: S102
    baseline = base_ns["decoder_baseline"]

    # Build optimized decoder via the code generator (full create_decoder)
    optimized_decoder = create_decoder(opt_plan, comments=False)

    # Also show the generated LE code
    factory = DecoderGeneratorFactory(opt_plan, comments=False, endianness="<")
    optimized_code = factory.generate_decoder_code("decoder_optimized", be_fallback="decoder_be")

    print("\nBaseline (old-style) decoder:")
    print("-" * 70)
    print(DECODER_BASELINE_CODE.strip())
    print("\nOptimized decoder (LE, inlined):")
    print("-" * 70)
    print(optimized_code)
    print("-" * 70)

    # Verify correctness
    for label, data in [("short", cdr_short), ("medium", cdr_medium), ("long", cdr_long)]:
        r1 = baseline(data)
        r2 = optimized_decoder(data)
        assert r1 == r2, f"Mismatch for {label}: {r1} != {r2}"
    print("\nCorrectness: PASSED\n")

    iterations = 500_000
    repeats = 7
    print(f"Benchmarking ({iterations:,} iterations x {repeats} repeats, best-of-{repeats})...\n")
    print(f"{'frame_id':<50} {'baseline':>10} {'optimized':>10} {'speedup':>10}")
    print("-" * 82)

    for label, data in [
        ("short (1 char)", cdr_short),
        ("medium ('base_link')", cdr_medium),
        ("long (47 chars)", cdr_long),
    ]:
        t_base = bench(baseline, data, iterations, repeats)
        t_opt = bench(optimized_decoder, data, iterations, repeats)
        speedup = (t_base - t_opt) / t_base * 100
        print(f"{label:<50} {t_base:>8.0f}ns {t_opt:>8.0f}ns {speedup:>+8.1f}%")

    # ─── ENCODER BENCHMARK ──────────────────────────────────────────────
    print("\n── Encoder ─────────────────────────────────────────────────────")

    # Build encoder
    optimized_encoder = create_encoder(opt_plan, comments=False)

    # Show generated encoder code
    enc_factory = EncoderGeneratorFactory(opt_plan, comments=False, endianness="<")
    encoder_code = enc_factory.generate_encoder_code("encoder_optimized")
    print("\nOptimized encoder:")
    print("-" * 70)
    print(encoder_code)
    print("-" * 70)

    # Create test messages by decoding the CDR payloads
    msg_short = optimized_decoder(cdr_short)
    msg_medium = optimized_decoder(cdr_medium)
    msg_long = optimized_decoder(cdr_long)

    # Also test with dict input
    msg_dict = {
        "header": {
            "stamp": {"sec": 1700000000, "nanosec": 500000},
            "frame_id": "base_link",
        },
        "velocity": 1.5,
        "steering": -0.3,
        "arm": 0.0,
        "shovel": 2.7,
        "handbreak": True,
        "safety": False,
    }

    # Verify encoder correctness (encode → decode roundtrip)
    for label, msg in [("short", msg_short), ("medium", msg_medium), ("long", msg_long)]:
        encoded = optimized_encoder(msg)
        roundtrip = optimized_decoder(encoded)
        assert msg == roundtrip, f"Encoder roundtrip mismatch for {label}"
    print("\nEncoder roundtrip: PASSED\n")

    print(f"Benchmarking ({iterations:,} iterations x {repeats} repeats, best-of-{repeats})...\n")
    print(f"{'input':<50} {'time':>10}")
    print("-" * 62)

    for label, msg in [
        ("object (short frame_id)", msg_short),
        ("object (medium frame_id)", msg_medium),
        ("object (long frame_id)", msg_long),
        ("dict (medium frame_id)", msg_dict),
    ]:
        t = bench_encoder(optimized_encoder, msg, iterations, repeats)
        print(f"{label:<50} {t:>8.0f}ns")

    print()


if __name__ == "__main__":
    main()
