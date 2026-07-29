"""Repeatable release-build throughput benchmark for representative CloudINI payloads.

Run from the repository root after building a release wheel:

    uv run --isolated --no-project --with /path/to/pureini.whl \
        python pureini/benchmarks/benchmark_matrix.py
"""

from __future__ import annotations

import argparse
import gc
import math
import struct
import time
from dataclasses import dataclass
from importlib.metadata import version
from statistics import median
from typing import TYPE_CHECKING

import pureini

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    data: bytes
    info: pureini.EncodingInfo
    is_exact: bool


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    raw_size: int
    encoded_size: int
    ratio: float
    encode_seconds: float
    decode_seconds: float
    encode_p95_seconds: float
    decode_p95_seconds: float

    @property
    def encode_mib_per_second(self) -> float:
        return self.raw_size / (1024 * 1024) / self.encode_seconds

    @property
    def decode_mib_per_second(self) -> float:
        return self.raw_size / (1024 * 1024) / self.decode_seconds


def create_info(
    *,
    point_count: int,
    point_step: int,
    fields: list[pureini.PointField],
    version: int,
    encoding: pureini.EncodingOptions = pureini.EncodingOptions.LOSSY,
    compression: pureini.CompressionOption = pureini.CompressionOption.NONE,
) -> pureini.EncodingInfo:
    return pureini.EncodingInfo(
        fields=fields,
        width=point_count,
        height=1,
        point_step=point_step,
        encoding_opt=encoding,
        compression_opt=compression,
        version=version,
    )


def sequential_u32(point_count: int) -> bytes:
    data = bytearray(point_count * 4)
    for index in range(point_count):
        struct.pack_into("<I", data, index * 4, index)
    return bytes(data)


def random_u32(point_count: int, maximum: int | None = None) -> bytes:
    data = bytearray(point_count * 4)
    state = 0x1234_5678
    for index in range(point_count):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        value = state if maximum is None else state % (maximum + 1)
        struct.pack_into("<I", data, index * 4, value)
    return bytes(data)


def xyz_f32(point_count: int) -> bytes:
    data = bytearray(point_count * 12)
    for index in range(point_count):
        angle = index * 0.001
        struct.pack_into(
            "<fff",
            data,
            index * 12,
            80.0 * math.cos(angle),
            80.0 * math.sin(angle),
            (index % 128) * 0.025,
        )
    return bytes(data)


def mixed_lidar(point_count: int) -> bytes:
    data = bytearray(point_count * 16)
    for index in range(point_count):
        angle = index * 0.001
        struct.pack_into(
            "<fffHH",
            data,
            index * 16,
            80.0 * math.cos(angle),
            80.0 * math.sin(angle),
            (index % 128) * 0.025,
            (index // 64) % 4096,
            index % 128,
        )
    return bytes(data)


def create_cases(point_count: int, version: int) -> list[BenchmarkCase]:
    sequential = sequential_u32(point_count)
    random_small = random_u32(point_count, 999)
    random_full = random_u32(point_count)
    xyz = xyz_f32(point_count)
    mixed = mixed_lidar(point_count)
    xyz_fields = [
        pureini.PointField("x", 0, pureini.FieldType.FLOAT32, 0.001),
        pureini.PointField("y", 4, pureini.FieldType.FLOAT32, 0.001),
        pureini.PointField("z", 8, pureini.FieldType.FLOAT32, 0.001),
    ]
    return [
        BenchmarkCase(
            "sequential-u32",
            sequential,
            create_info(
                point_count=point_count,
                point_step=4,
                fields=[pureini.PointField("index", 0, pureini.FieldType.UINT32)],
                version=version,
            ),
            True,
        ),
        BenchmarkCase(
            "random-u32-0-999",
            random_small,
            create_info(
                point_count=point_count,
                point_step=4,
                fields=[pureini.PointField("value", 0, pureini.FieldType.UINT32)],
                version=version,
            ),
            True,
        ),
        BenchmarkCase(
            "random-u32-full",
            random_full,
            create_info(
                point_count=point_count,
                point_step=4,
                fields=[pureini.PointField("value", 0, pureini.FieldType.UINT32)],
                version=version,
            ),
            True,
        ),
        BenchmarkCase(
            "xyz-lossy-none",
            xyz,
            create_info(
                point_count=point_count,
                point_step=12,
                fields=xyz_fields,
                version=version,
            ),
            False,
        ),
        BenchmarkCase(
            "xyz-lossy-lz4",
            xyz,
            create_info(
                point_count=point_count,
                point_step=12,
                fields=xyz_fields,
                version=version,
                compression=pureini.CompressionOption.LZ4,
            ),
            False,
        ),
        BenchmarkCase(
            "xyz-lossy-zstd",
            xyz,
            create_info(
                point_count=point_count,
                point_step=12,
                fields=xyz_fields,
                version=version,
                compression=pureini.CompressionOption.ZSTD,
            ),
            False,
        ),
        BenchmarkCase(
            "xyz-lossless-none",
            xyz,
            create_info(
                point_count=point_count,
                point_step=12,
                version=version,
                fields=[
                    pureini.PointField("x", 0, pureini.FieldType.FLOAT32),
                    pureini.PointField("y", 4, pureini.FieldType.FLOAT32),
                    pureini.PointField("z", 8, pureini.FieldType.FLOAT32),
                ],
                encoding=pureini.EncodingOptions.LOSSLESS,
            ),
            True,
        ),
        BenchmarkCase(
            "mixed-lidar",
            mixed,
            create_info(
                point_count=point_count,
                point_step=16,
                version=version,
                fields=[
                    *xyz_fields,
                    pureini.PointField("intensity", 12, pureini.FieldType.UINT16),
                    pureini.PointField("ring", 14, pureini.FieldType.UINT16),
                ],
            ),
            False,
        ),
    ]


def percentile_95(durations: list[float]) -> float:
    return sorted(durations)[math.ceil(len(durations) * 0.95) - 1]


def measure(operation: Callable[[], None], iterations: int) -> tuple[float, float]:
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - start)
    return median(durations), percentile_95(durations)


def benchmark_case(case: BenchmarkCase, iterations: int, warmups: int) -> BenchmarkResult:
    encoder = pureini.PointcloudEncoder(case.info)
    decoder = pureini.PointcloudDecoder()
    encoded = encoder.encode(case.data)
    decoded, decoded_info = decoder.decode(encoded)
    assert decoded_info == case.info
    assert len(decoded) == len(case.data)
    if case.is_exact:
        assert decoded == case.data

    for _ in range(warmups):
        encoder.encode(case.data)
        decoder.decode(encoded)

    encode_result = encoded
    decode_result = decoded

    def run_encode() -> None:
        nonlocal encode_result
        encode_result = encoder.encode(case.data)

    def run_decode() -> None:
        nonlocal decode_result
        decode_result, _ = decoder.decode(encoded)

    encode_seconds, encode_p95_seconds = measure(run_encode, iterations)
    decode_seconds, decode_p95_seconds = measure(run_decode, iterations)
    assert len(encode_result) == len(encoded)
    assert len(decode_result) == len(case.data)
    return BenchmarkResult(
        name=case.name,
        raw_size=len(case.data),
        encoded_size=len(encoded),
        ratio=len(case.data) / len(encoded),
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        encode_p95_seconds=encode_p95_seconds,
        decode_p95_seconds=decode_p95_seconds,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        f"{'case':<20} {'raw MiB':>8} {'ratio':>7} "
        f"{'enc MiB/s':>10} {'dec MiB/s':>10} {'enc p95':>9} {'dec p95':>9}"
    )
    for result in results:
        print(
            f"{result.name:<20} {result.raw_size / (1024 * 1024):>8.2f} "
            f"{result.ratio:>7.2f} {result.encode_mib_per_second:>10.0f} "
            f"{result.decode_mib_per_second:>10.0f} "
            f"{result.encode_p95_seconds * 1000:>8.2f}ms "
            f"{result.decode_p95_seconds * 1000:>8.2f}ms"
        )


def installed_backend_name() -> str:
    try:
        from pureini import _pureini as native_backend
    except ImportError:
        return "python"
    return native_backend.__implementation__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=500_000)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--version", type=int, default=5)
    parser.add_argument("--case", action="append", dest="cases")
    args = parser.parse_args()
    if args.points <= 0 or args.iterations <= 0 or args.warmups < 0:
        parser.error("points and iterations must be positive; warmups must be non-negative")

    gc.disable()
    try:
        cases = create_cases(args.points, args.version)
        if args.cases:
            selected = set(args.cases)
            cases = [case for case in cases if case.name in selected]
            missing = selected.difference(case.name for case in cases)
            if missing:
                parser.error(f"unknown cases: {', '.join(sorted(missing))}")
        results = [benchmark_case(case, args.iterations, args.warmups) for case in cases]
    finally:
        gc.enable()
    print(f"pureini {version('pureini')} ({installed_backend_name()}), V{args.version}")
    print_results(results)


if __name__ == "__main__":
    main()
