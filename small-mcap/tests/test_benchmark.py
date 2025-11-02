"""Benchmark tests comparing small_mcap vs mcap library for various reading scenarios."""

import io
from dataclasses import dataclass
from pathlib import Path

import pytest
from mcap.reader import make_reader
from small_mcap import include_topics, read_message

# Test file path - nuScenes MCAP file with 30,900 messages, 19.15s duration, 560 zstd chunks
TEST_MCAP_FILE = (
    Path(__file__).parent.parent.parent
    / "data"
    / "data"
    / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
)

# Topics with different message rates for filtering tests
TEST_TOPICS = [
    "/diagnostics",  # 22,019 msgs (high frequency)
    "/odom",  # 937 msgs (medium frequency)
    "/CAM_FRONT/image_rect_compressed",  # 224 msgs (camera data)
    "/LIDAR_TOP",  # 382 msgs (sensor data)
]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark test scenario."""

    id: str
    seekable: bool
    topics: list[str] | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None


# Test configurations for all benchmark scenarios
# Note: File timestamps are Unix time (2018-07-24 02:42:07 to 02:42:26)
# Time range: 1532402927647 to 1532402946797 (ms), duration: 19,149 ms
TEST_CONFIGS = [
    BenchmarkConfig(id="full-seekable", seekable=True),
    BenchmarkConfig(id="full-nonseekable", seekable=False),
    BenchmarkConfig(
        id="time-seekable",
        seekable=True,
        start_time_ms=1532402927647,
        end_time_ms=1532402932647,
    ),
    BenchmarkConfig(
        id="time-nonseekable",
        seekable=False,
        start_time_ms=1532402927647,
        end_time_ms=1532402932647,
    ),
    BenchmarkConfig(id="topic-seekable", seekable=True, topics=TEST_TOPICS),
    BenchmarkConfig(id="topic-nonseekable", seekable=False, topics=TEST_TOPICS),
]


# ============================================================================
# Generic wrapper functions
# ============================================================================


def read_small_mcap(config: BenchmarkConfig) -> int:
    """Generic wrapper for small_mcap reading with configurable options."""
    count = 0

    # Open stream based on seekable flag
    if config.seekable:
        stream = TEST_MCAP_FILE.open("rb")
    else:
        data = TEST_MCAP_FILE.read_bytes()
        stream = io.BytesIO(data)

    try:
        # Build read_message arguments based on config
        kwargs = {}

        # Add topic filter if specified
        if config.topics is not None:
            kwargs["should_include"] = include_topics(config.topics)

        # Add time filter if specified (convert ms to ns)
        if config.start_time_ms is not None:
            kwargs["start_time_ns"] = config.start_time_ms * 1_000_000
        if config.end_time_ms is not None:
            kwargs["end_time_ns"] = config.end_time_ms * 1_000_000

        # Read and count messages
        for _schema, _channel, _message in read_message(stream, **kwargs):
            count += 1

    finally:
        if hasattr(stream, "close"):
            stream.close()

    return count


def read_mcap(config: BenchmarkConfig) -> int:
    """Generic wrapper for mcap library reading with configurable options."""
    count = 0

    # Open stream based on seekable flag
    if config.seekable:
        stream = TEST_MCAP_FILE.open("rb")
    else:
        data = TEST_MCAP_FILE.read_bytes()
        stream = io.BytesIO(data)

    try:
        reader = make_reader(stream)

        # Build iter_messages arguments based on config
        kwargs = {}

        # Add topic filter if specified
        if config.topics is not None:
            kwargs["topics"] = config.topics

        # Add time filter if specified (convert ms to ns)
        if config.start_time_ms is not None:
            kwargs["start_time"] = config.start_time_ms * 1_000_000
            kwargs["log_time_order"] = True
        if config.end_time_ms is not None:
            kwargs["end_time"] = config.end_time_ms * 1_000_000
            kwargs["log_time_order"] = True

        # Read and count messages
        for _schema, _channel, _message in reader.iter_messages(**kwargs):
            count += 1

    finally:
        if hasattr(stream, "close"):
            stream.close()

    return count


# ============================================================================
# Benchmark tests
# ============================================================================


@pytest.mark.parametrize(
    ("library", "reader_func"),
    [
        pytest.param("small_mcap", read_small_mcap, id="small_mcap"),
        pytest.param("mcap", read_mcap, id="mcap"),
    ],
)
@pytest.mark.parametrize("config", TEST_CONFIGS, ids=lambda c: c.id)
def test_benchmark_read(benchmark, library, reader_func, config):
    """Benchmark reading with various configurations.

    Each config creates its own benchmark group for easy comparison between libraries.
    """
    # Set benchmark group dynamically based on config
    benchmark.group = config.id

    result = benchmark(reader_func, config)
    assert result > 0, f"{library} should read at least one message"
