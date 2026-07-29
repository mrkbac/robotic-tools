"""Compare fused Rust preprocessing with the previous Python staging pipeline."""

from __future__ import annotations

import argparse
import gc
import struct
import time
from dataclasses import dataclass, field
from statistics import median
from typing import TYPE_CHECKING

import numba as nb  # ty: ignore[unresolved-import]
import numpy as np
import pureini
from pointcloud2 import PointCloud2, PointField

if TYPE_CHECKING:
    from collections.abc import Callable

    from pointcloud2 import Pointcloud2Msg
    from pointcloud2.messages import Stamp


@dataclass
class BenchmarkStamp:
    sec: int = 0
    nanosec: int = 0


@dataclass
class Header:
    frame_id: str = ""
    stamp: Stamp = field(default_factory=BenchmarkStamp)


@nb.njit(cache=True, fastmath=False, nogil=True)
def _legacy_compact_group(
    data: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    keys: np.ndarray,
    output: np.ndarray,
    buckets: np.ndarray,
    counts: np.ndarray,
) -> int:
    for index in range(x_values.shape[0]):
        x_value = x_values[index]
        y_value = y_values[index]
        z_value = z_values[index]
        if (
            x_value != x_value
            or y_value != y_value
            or z_value != z_value
            or (x_value == 0.0 and y_value == 0.0 and z_value == 0.0)
        ):
            buckets[index] = -1
        else:
            bucket = keys[index]
            buckets[index] = bucket
            counts[bucket + 1] += 1
    for bucket in range(1, counts.shape[0]):
        counts[bucket] += counts[bucket - 1]
    total = counts[-1]
    for index in range(x_values.shape[0]):
        bucket = buckets[index]
        if bucket < 0:
            continue
        destination = counts[bucket]
        counts[bucket] += 1
        source_offset = index * 2
        destination_offset = destination * 2
        output[destination_offset : destination_offset + 2] = data[
            source_offset : source_offset + 2
        ]
    return total


@nb.njit(cache=True, fastmath=False, nogil=True)
def _legacy_filter(
    data: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    output: np.ndarray,
) -> int:
    kept = 0
    for index in range(x_values.shape[0]):
        x_value = x_values[index]
        y_value = y_values[index]
        z_value = z_values[index]
        if not (
            x_value != x_value
            or y_value != y_value
            or z_value != z_value
            or (x_value == 0.0 and y_value == 0.0 and z_value == 0.0)
        ):
            source_offset = index * 2
            destination_offset = kept * 2
            for word_offset in range(2):
                output[destination_offset + word_offset] = data[source_offset + word_offset]
            kept += 1
    return kept


def legacy_preprocess(cloud: PointCloud2, *, sort_field: str | None) -> PointCloud2:
    point_count = cloud.width * cloud.height
    x_values = np.ndarray((point_count,), "<f4", cloud.data, 0, (cloud.point_step,))
    y_values = np.ndarray((point_count,), "<f4", cloud.data, 4, (cloud.point_step,))
    z_values = np.ndarray((point_count,), "<f4", cloud.data, 8, (cloud.point_step,))
    data = np.frombuffer(cloud.data, dtype=np.uint64)
    output = np.empty(data.shape[0], dtype=np.uint64)
    if sort_field is None:
        kept = _legacy_filter(data, x_values, y_values, z_values, output)
    else:
        key_view = np.ndarray((point_count,), "u1", cloud.data, 12, (cloud.point_step,))
        keys = np.ascontiguousarray(key_view)
        buckets = np.empty(point_count, dtype=np.int16)
        bucket_count = int(keys.max()) + 1 if keys.size else 1
        counts = np.zeros(bucket_count + 1, dtype=np.int64)
        kept = _legacy_compact_group(
            data,
            x_values,
            y_values,
            z_values,
            keys,
            output,
            buckets,
            counts,
        )
    return PointCloud2(
        header=cloud.header,
        height=1,
        width=kept,
        fields=cloud.fields,
        is_bigendian=False,
        point_step=cloud.point_step,
        row_step=kept * cloud.point_step,
        data=output[: kept * 2].tobytes(),
        is_dense=True,
    )


def create_cloud(point_count: int) -> PointCloud2:
    data = bytearray(point_count * 16)
    for index in range(point_count):
        if index % 5:
            struct.pack_into(
                "<fffB",
                data,
                index * 16,
                index * 0.01 + 1.0,
                index * 0.02 + 1.0,
                index * 0.03 + 1.0,
                index % 128,
            )
        else:
            data[index * 16 + 12] = index % 128
    fields = [
        PointField("x", 0, PointField.FLOAT32),
        PointField("y", 4, PointField.FLOAT32),
        PointField("z", 8, PointField.FLOAT32),
        PointField("line", 12, PointField.UINT8),
    ]
    return PointCloud2(
        header=Header(),
        height=1,
        width=point_count,
        fields=fields,
        is_bigendian=False,
        point_step=16,
        row_step=point_count * 16,
        data=bytes(data),
        is_dense=False,
    )


def create_encoder(cloud: Pointcloud2Msg) -> pureini.PointcloudEncoder:
    return pureini.PointcloudEncoder(
        pureini.EncodingInfo(
            fields=[
                pureini.PointField("x", 0, pureini.FieldType.FLOAT32, 0.01),
                pureini.PointField("y", 4, pureini.FieldType.FLOAT32, 0.01),
                pureini.PointField("z", 8, pureini.FieldType.FLOAT32, 0.01),
                pureini.PointField("line", 12, pureini.FieldType.UINT8),
            ],
            width=cloud.width,
            height=cloud.height,
            point_step=cloud.point_step,
            encoding_opt=pureini.EncodingOptions.LOSSY,
            compression_opt=pureini.CompressionOption.ZSTD,
        )
    )


def measure(operation: Callable[[], bytes], iterations: int) -> float:
    durations: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - start)
    return median(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=500_000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument(
        "--minimum-speedup",
        type=float,
        default=0.0,
        help="Exit unsuccessfully when fused filter+group encoding is below this ratio",
    )
    args = parser.parse_args()

    cloud = create_cloud(args.points)
    cleaned_reference = legacy_preprocess(cloud, sort_field="line")
    filtered_reference = legacy_preprocess(cloud, sort_field=None)
    staged_encoder = create_encoder(cleaned_reference)
    filtered_encoder = create_encoder(filtered_reference)
    fused_encoder = create_encoder(cloud)

    def staged() -> bytes:
        cleaned = legacy_preprocess(cloud, sort_field="line")
        return staged_encoder.encode(cleaned.data)

    def fused() -> bytes:
        return fused_encoder.encode(
            cloud.data,
            drop_invalid=True,
            sort_field="line",
        )

    def staged_filter() -> bytes:
        cleaned = legacy_preprocess(cloud, sort_field=None)
        return filtered_encoder.encode(cleaned.data)

    def fused_filter() -> bytes:
        return fused_encoder.encode(
            cloud.data,
            drop_invalid=True,
            sort_field=None,
        )

    def staged_preprocess() -> bytes:
        return legacy_preprocess(cloud, sort_field="line").data

    def fused_preprocess() -> bytes:
        return fused_encoder.preprocess(
            cloud.data,
            drop_invalid=True,
            sort_field="line",
        )[0]

    def encode_only() -> bytes:
        return staged_encoder.encode(cleaned_reference.data)

    def filter_encode_only() -> bytes:
        return filtered_encoder.encode(filtered_reference.data)

    expected = staged()
    actual = fused()
    expected_raw, expected_info = pureini.PointcloudDecoder().decode(expected)
    actual_raw, actual_info = pureini.PointcloudDecoder().decode(actual)
    assert actual_info == expected_info
    assert actual_raw == expected_raw
    expected_filter_raw, expected_filter_info = pureini.PointcloudDecoder().decode(staged_filter())
    actual_filter_raw, actual_filter_info = pureini.PointcloudDecoder().decode(fused_filter())
    assert actual_filter_info == expected_filter_info
    assert actual_filter_raw == expected_filter_raw
    assert fused_preprocess() == staged_preprocess()

    for _ in range(args.warmups):
        staged()
        fused()
        staged_filter()
        fused_filter()
        encode_only()
        filter_encode_only()
        staged_preprocess()
        fused_preprocess()

    gc.disable()
    try:
        staged_seconds = measure(staged, args.iterations)
        fused_seconds = measure(fused, args.iterations)
        staged_filter_seconds = measure(staged_filter, args.iterations)
        fused_filter_seconds = measure(fused_filter, args.iterations)
        encode_only_seconds = measure(encode_only, args.iterations)
        filter_encode_only_seconds = measure(filter_encode_only, args.iterations)
        staged_preprocess_seconds = measure(staged_preprocess, args.iterations)
        fused_preprocess_seconds = measure(fused_preprocess, args.iterations)
    finally:
        gc.enable()

    mib = len(cloud.data) / (1024 * 1024)
    print(f"points: {args.points:,}")
    print(
        f"encode only floor:      {encode_only_seconds * 1000:.3f} ms "
        f"({mib / encode_only_seconds:.1f} MiB/s)"
    )
    print(
        f"staged Python + encode: {staged_seconds * 1000:.3f} ms ({mib / staged_seconds:.1f} MiB/s)"
    )
    print(
        f"fused Rust encode:      {fused_seconds * 1000:.3f} ms ({mib / fused_seconds:.1f} MiB/s)"
    )
    speedup = staged_seconds / fused_seconds
    print(f"speedup:                {speedup:.2f}x")
    print(
        f"preprocessing only:     {staged_preprocess_seconds / fused_preprocess_seconds:.2f}x "
        f"({staged_preprocess_seconds * 1000:.3f} vs "
        f"{fused_preprocess_seconds * 1000:.3f} ms)"
    )
    print()
    print(f"filter encode floor:    {filter_encode_only_seconds * 1000:.3f} ms")
    print(f"filter-only staged:     {staged_filter_seconds * 1000:.3f} ms")
    print(f"filter-only fused:      {fused_filter_seconds * 1000:.3f} ms")
    print(f"filter-only speedup:    {staged_filter_seconds / fused_filter_seconds:.2f}x")
    if speedup < args.minimum_speedup:
        raise SystemExit(
            f"fused preprocessing speedup {speedup:.2f}x is below "
            f"the required {args.minimum_speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
