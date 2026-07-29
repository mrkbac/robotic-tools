# pureini

CloudINI-compatible point-cloud compression for Python, implemented as a
Rust/PyO3 extension.

- Reads CloudINI V2, V3, V4, and V5 payloads.
- Writes the current V5 format by default, with V2-V4 output available through
  `EncodingInfo.version`.
- Supports no second-stage compression, LZ4, and Zstandard.
- Has no Python runtime dependencies and does not require NumPy, Numba, or
  LLVM.
- Uses Python's stable ABI for CPython 3.10 and newer.

```python
import pureini

info = pureini.EncodingInfo(
    width=1,
    height=1,
    point_step=4,
    fields=[
        pureini.PointField(
            name="x",
            offset=0,
            type=pureini.FieldType.FLOAT32,
            resolution=0.001,
        )
    ],
)

encoded = pureini.PointcloudEncoder(info).encode(b"\0\0\0\0")
decoded, decoded_info = pureini.PointcloudDecoder().decode(encoded)
```

The public API is deliberately limited to `PointField`, `EncodingInfo`,
`FieldType`, `EncodingOptions`, `CompressionOption`, `PointcloudEncoder`, and
`PointcloudDecoder`. Wire-format primitives and header helpers are internal.

Point-cloud cleanup can be fused into the native encode call. This removes
invalid XYZ returns, stably groups retained points by a discrete field, and
returns the updated layout together with the payload:

```python
encoded, output_point_count, did_filter_invalid_xyz = pureini.PointcloudEncoder(info).encode(
    cloud_data,
    drop_invalid=True,
    sort_field="line",
    return_metadata=True,
)
```

`return_metadata=True` opts into the tuple result. Without it, `encode` returns
bytes directly even when preprocessing is enabled. `output_point_count` is
`None` when preprocessing leaves the input layout unchanged. `point_step`
never changes, and transformed clouds are one row high.

Source point data is little-endian by default. Pass `is_bigendian=True` to
normalize big-endian fields in Rust before preprocessing or encoding.

Native preprocessing is opt-in and releases the GIL; larger counting-sort
workloads run in parallel.

Add it to a project with `uv add pureini`.

## Distribution

Release artifacts use CPython's stable ABI and cover Python 3.10 and newer:

- manylinux and musllinux wheels for x86-64 and AArch64;
- macOS wheels for Intel and Apple Silicon;
- Windows x86-64 wheels;
- a source distribution containing the complete Rust crate.

CI tests native wheels on CPython 3.10 and 3.14 and independently rebuilds and
tests the source distribution.

## Benchmarking

Benchmark a release wheel rather than the editable debug build:

```shell
uv build --package pureini --out-dir dist
uv run --isolated --no-project --with dist/pureini-0.8.0-*.whl \
  python pureini/benchmarks/benchmark_matrix.py
```

The matrix uses warmed median timings for structured and random integers,
lossy and lossless XYZ, LZ4, Zstandard, and mixed LiDAR fields. Adjust
`--points`, `--iterations`, and `--warmups` for the target machine.

Compare the fused Rust preprocessing path against the previous warmed Numba
pipeline and require a material win:

```shell
uv run --with numba python pureini/benchmarks/benchmark_preprocessing.py \
  --points 500000 --iterations 30 --warmups 8 --minimum-speedup 1.25
```

The benchmark validates decoded bytes and layout before timing. Its default
fixture contains 20% invalid XYZ placeholders and 128 interleaved `line`
groups.
