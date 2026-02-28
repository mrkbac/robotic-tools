"""
Benchmark suite for pureini encoders.

Run with: uv run python benchmarks/benchmark_encoders.py
Or with profiling: uv run pyinstrument benchmarks/benchmark_encoders.py
"""

import random
import struct
import time

from pureini import CompressionOption, EncodingInfo, EncodingOptions, FieldType, PointField
from pureini.decoder import PointcloudDecoder
from pureini.encoder import PointcloudEncoder


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def format_size(bytes_count: int) -> str:
    """Format size in human-readable format."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.2f} KB"
    return f"{bytes_count / (1024 * 1024):.2f} MB"


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(
        self,
        name: str,
        num_points: int,
        encode_time: float,
        decode_time: float,
        original_size: int,
        compressed_size: int,
    ) -> None:
        self.name = name
        self.num_points = num_points
        self.encode_time = encode_time
        self.decode_time = decode_time
        self.original_size = original_size
        self.compressed_size = compressed_size

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.original_size / self.compressed_size if self.compressed_size > 0 else 0

    @property
    def encode_throughput(self) -> float:
        """Calculate encoding throughput in MB/s."""
        return (
            (self.original_size / (1024 * 1024)) / self.encode_time if self.encode_time > 0 else 0
        )

    @property
    def decode_throughput(self) -> float:
        """Calculate decoding throughput in MB/s."""
        return (
            (self.original_size / (1024 * 1024)) / self.decode_time if self.decode_time > 0 else 0
        )

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print(f"\n{'=' * 70}")
        print(f"Benchmark: {self.name}")
        print(f"{'=' * 70}")
        print(f"Points:              {self.num_points:,}")
        print(f"Original size:       {format_size(self.original_size)}")
        print(f"Compressed size:     {format_size(self.compressed_size)}")
        print(f"Compression ratio:   {self.compression_ratio:.2f}x")
        print()
        print(f"Encode time:         {format_time(self.encode_time)}")
        print(f"Encode throughput:   {self.encode_throughput:.2f} MB/s")
        print(f"Decode time:         {format_time(self.decode_time)}")
        print(f"Decode throughput:   {self.decode_throughput:.2f} MB/s")
        print(f"Total time:          {format_time(self.encode_time + self.decode_time)}")
        print(f"{'=' * 70}")


def benchmark_int_field(num_points: int = 100_000) -> BenchmarkResult:
    """
    Benchmark integer field encoding/decoding.

    Args:
        num_points: Number of points to encode

    Returns:
        BenchmarkResult with timing information
    """
    random.seed(42)

    # Generate random uint32 values (0-999)
    input_data = [random.randint(0, 999) for _ in range(num_points)]  # noqa: S311

    # Create point cloud data (4 bytes per point = uint32)
    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<I", value))

    # Create encoding info
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [PointField(name="value", offset=0, type=FieldType.UINT32, resolution=None)]

    # Benchmark encoding
    encoder = PointcloudEncoder(info)
    start = time.perf_counter()
    compressed = encoder.encode(bytes(point_cloud))
    encode_time = time.perf_counter() - start

    # Benchmark decoding
    decoder = PointcloudDecoder()
    start = time.perf_counter()
    decompressed, _decoded_info = decoder.decode(compressed)
    decode_time = time.perf_counter() - start

    # Verify correctness
    output_data = []
    for i in range(num_points):
        offset = i * 4
        value = struct.unpack_from("<I", decompressed, offset)[0]
        output_data.append(value)

    assert len(output_data) == len(input_data), "Size mismatch"
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i], f"Data mismatch at {i}"

    return BenchmarkResult(
        name="Integer Field (UINT32)",
        num_points=num_points,
        encode_time=encode_time,
        decode_time=decode_time,
        original_size=len(point_cloud),
        compressed_size=len(compressed),
    )


def benchmark_timestamps(num_points: int = 100_000) -> BenchmarkResult:
    """
    Benchmark UINT64 timestamp encoding (realistic robotics pattern).

    Simulates ROS2 timestamps with regular intervals (60Hz).
    """
    random.seed(42)

    # Generate realistic microsecond timestamps (60Hz = 16ms intervals)
    start_timestamp = 1730000000000000  # Nov 2024 in microseconds
    frame_interval = 16000  # 60Hz
    input_data = [start_timestamp + i * frame_interval for i in range(num_points)]

    # Create point cloud data (8 bytes per point = uint64)
    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<Q", value))

    # Create encoding info
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 8
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [PointField(name="timestamp", offset=0, type=FieldType.UINT64, resolution=None)]

    # Benchmark encoding
    encoder = PointcloudEncoder(info)
    start = time.perf_counter()
    compressed = encoder.encode(bytes(point_cloud))
    encode_time = time.perf_counter() - start

    # Benchmark decoding
    decoder = PointcloudDecoder()
    start = time.perf_counter()
    decompressed, _decoded_info = decoder.decode(compressed)
    decode_time = time.perf_counter() - start

    # Verify correctness
    output_data = []
    for i in range(num_points):
        offset = i * 8
        value = struct.unpack_from("<Q", decompressed, offset)[0]
        output_data.append(value)

    assert len(output_data) == len(input_data), "Size mismatch"
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i], f"Data mismatch at {i}"

    return BenchmarkResult(
        name="Timestamps (UINT64, 60Hz pattern)",
        num_points=num_points,
        encode_time=encode_time,
        decode_time=decode_time,
        original_size=len(point_cloud),
        compressed_size=len(compressed),
    )


def benchmark_scan_rows(num_points: int = 100_000) -> BenchmarkResult:
    """
    Benchmark sequential row indices (optimal compression pattern).

    Simulates lidar/camera scan line indices.
    """
    random.seed(42)

    # Generate sequential row indices
    input_data = list(range(num_points))

    # Create point cloud data (4 bytes per point = uint32)
    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<I", value))

    # Create encoding info
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [PointField(name="row", offset=0, type=FieldType.UINT32, resolution=None)]

    # Benchmark encoding
    encoder = PointcloudEncoder(info)
    start = time.perf_counter()
    compressed = encoder.encode(bytes(point_cloud))
    encode_time = time.perf_counter() - start

    # Benchmark decoding
    decoder = PointcloudDecoder()
    start = time.perf_counter()
    decompressed, _decoded_info = decoder.decode(compressed)
    decode_time = time.perf_counter() - start

    # Verify correctness
    output_data = []
    for i in range(num_points):
        offset = i * 4
        value = struct.unpack_from("<I", decompressed, offset)[0]
        output_data.append(value)

    assert len(output_data) == len(input_data), "Size mismatch"
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i], f"Data mismatch at {i}"

    return BenchmarkResult(
        name="Scan Rows (UINT32, sequential)",
        num_points=num_points,
        encode_time=encode_time,
        decode_time=decode_time,
        original_size=len(point_cloud),
        compressed_size=len(compressed),
    )


def benchmark_labels(num_points: int = 100_000) -> BenchmarkResult:
    """
    Benchmark object detection labels (clustered pattern).

    Simulates semantic segmentation with repeated class IDs.
    """
    random.seed(42)

    # Generate realistic label distribution (clustered objects)
    input_data = []
    current_label = 0
    for _ in range(num_points):
        if random.random() < 0.95:  # 95% chance to stay in same object  # noqa: S311
            input_data.append(current_label)
        else:  # 5% chance to switch to new object
            current_label = random.randint(0, 20)  # 20 common classes  # noqa: S311
            input_data.append(current_label)

    # Create point cloud data (4 bytes per point = uint32)
    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<I", value))

    # Create encoding info
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [PointField(name="label", offset=0, type=FieldType.UINT32, resolution=None)]

    # Benchmark encoding
    encoder = PointcloudEncoder(info)
    start = time.perf_counter()
    compressed = encoder.encode(bytes(point_cloud))
    encode_time = time.perf_counter() - start

    # Benchmark decoding
    decoder = PointcloudDecoder()
    start = time.perf_counter()
    decompressed, _decoded_info = decoder.decode(compressed)
    decode_time = time.perf_counter() - start

    # Verify correctness
    output_data = []
    for i in range(num_points):
        offset = i * 4
        value = struct.unpack_from("<I", decompressed, offset)[0]
        output_data.append(value)

    assert len(output_data) == len(input_data), "Size mismatch"
    for i in range(len(input_data)):
        assert output_data[i] == input_data[i], f"Data mismatch at {i}"

    return BenchmarkResult(
        name="Labels (UINT32, clustered 0-20)",
        num_points=num_points,
        encode_time=encode_time,
        decode_time=decode_time,
        original_size=len(point_cloud),
        compressed_size=len(compressed),
    )


def benchmark_float_lossy(num_points: int = 1_000_000) -> BenchmarkResult:
    """
    Benchmark lossy float compression (main bottleneck).

    Args:
        num_points: Number of points to encode

    Returns:
        BenchmarkResult with timing information
    """
    resolution = 0.01
    tolerance = resolution * 1.0001

    random.seed(42)

    # Generate random float values (0.0 to ~10.0)
    input_data = [0.001 * random.randint(0, 10000) for _ in range(num_points)]  # noqa: S311

    # Insert NaN values at specific indices
    input_data[1] = float("nan")
    input_data[15] = float("nan")
    input_data[16] = float("nan")

    # Create point cloud data
    point_cloud = bytearray()
    for value in input_data:
        point_cloud.extend(struct.pack("<f", value))

    # Create encoding info
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 4
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [
        PointField(name="the_float", offset=0, type=FieldType.FLOAT32, resolution=resolution)
    ]

    # Benchmark encoding
    encoder = PointcloudEncoder(info)
    start = time.perf_counter()
    compressed = encoder.encode(bytes(point_cloud))
    encode_time = time.perf_counter() - start

    # Benchmark decoding
    decoder = PointcloudDecoder()
    start = time.perf_counter()
    decompressed, _decoded_info = decoder.decode(compressed)
    decode_time = time.perf_counter() - start

    # Verify correctness
    import math

    output_data = []
    max_diff = 0.0

    for i in range(num_points):
        offset = i * 4
        value = struct.unpack_from("<f", decompressed, offset)[0]
        output_data.append(value)

        if math.isnan(input_data[i]):
            assert math.isnan(value), f"NaN not preserved at index {i}"
        else:
            diff = abs(value - input_data[i])
            max_diff = max(max_diff, diff)
            assert diff <= tolerance, f"Tolerance exceeded at {i}: {diff} > {tolerance}"

    return BenchmarkResult(
        name="Float Lossy (FLOAT32, 1M points)",
        num_points=num_points,
        encode_time=encode_time,
        decode_time=decode_time,
        original_size=len(point_cloud),
        compressed_size=len(compressed),
    )


def benchmark_float_lossy_xyz(num_points: int = 100_000) -> BenchmarkResult:
    """
    Benchmark FloatN lossy encoding for XYZ fields.

    Args:
        num_points: Number of points to encode

    Returns:
        BenchmarkResult with timing information
    """
    resolution = 0.001
    tolerance = resolution * 1.0001

    random.seed(42)

    # Generate random XYZ values
    point_cloud = bytearray()
    input_data = []
    for _ in range(num_points):
        x = random.uniform(-100.0, 100.0)  # noqa: S311
        y = random.uniform(-100.0, 100.0)  # noqa: S311
        z = random.uniform(-100.0, 100.0)  # noqa: S311
        input_data.append((x, y, z))
        point_cloud.extend(struct.pack("<fff", x, y, z))

    # Create encoding info
    info = EncodingInfo()
    info.width = num_points
    info.height = 1
    info.point_step = 12  # 3 floats
    info.encoding_opt = EncodingOptions.LOSSY
    info.compression_opt = CompressionOption.NONE
    info.fields = [
        PointField(name="x", offset=0, type=FieldType.FLOAT32, resolution=resolution),
        PointField(name="y", offset=4, type=FieldType.FLOAT32, resolution=resolution),
        PointField(name="z", offset=8, type=FieldType.FLOAT32, resolution=resolution),
    ]

    # Benchmark encoding
    encoder = PointcloudEncoder(info)
    start = time.perf_counter()
    compressed = encoder.encode(bytes(point_cloud))
    encode_time = time.perf_counter() - start

    # Benchmark decoding
    decoder = PointcloudDecoder()
    start = time.perf_counter()
    decompressed, _decoded_info = decoder.decode(compressed)
    decode_time = time.perf_counter() - start

    # Verify correctness
    for i in range(num_points):
        offset = i * 12
        x, y, z = struct.unpack_from("<fff", decompressed, offset)
        orig_x, orig_y, orig_z = input_data[i]

        assert abs(x - orig_x) <= tolerance, f"X tolerance exceeded at {i}"
        assert abs(y - orig_y) <= tolerance, f"Y tolerance exceeded at {i}"
        assert abs(z - orig_z) <= tolerance, f"Z tolerance exceeded at {i}"

    return BenchmarkResult(
        name="FloatN Lossy (XYZ, 100K points)",
        num_points=num_points,
        encode_time=encode_time,
        decode_time=decode_time,
        original_size=len(point_cloud),
        compressed_size=len(compressed),
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    print("Starting pureini benchmarks...")
    print("=" * 70)

    results = []

    # Integer benchmark (random data - worst case)
    print("\nRunning integer field benchmark (random 0-999 - worst case)...")
    result = benchmark_int_field(num_points=100_000)
    result.print_summary()
    results.append(result)

    # Realistic integer benchmarks
    print("\nRunning timestamps benchmark (realistic robotics data)...")
    result = benchmark_timestamps(num_points=100_000)
    result.print_summary()
    results.append(result)

    print("\nRunning scan rows benchmark (sequential indices)...")
    result = benchmark_scan_rows(num_points=100_000)
    result.print_summary()
    results.append(result)

    print("\nRunning labels benchmark (clustered classes)...")
    result = benchmark_labels(num_points=100_000)
    result.print_summary()
    results.append(result)

    # Float lossy benchmark (main bottleneck - 1M points)
    print("\nRunning float lossy benchmark (1M points - main bottleneck)...")
    result = benchmark_float_lossy(num_points=1_000_000)
    result.print_summary()
    results.append(result)

    # FloatN XYZ benchmark
    print("\nRunning FloatN XYZ benchmark...")
    result = benchmark_float_lossy_xyz(num_points=100_000)
    result.print_summary()
    results.append(result)

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_time = sum(r.encode_time + r.decode_time for r in results)
    print(f"Total benchmark time: {format_time(total_time)}")
    print(f"Number of benchmarks: {len(results)}")
    print("=" * 70)
