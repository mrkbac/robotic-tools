# small-mcap

Lightweight Python library for reading and writing MCAP files.

## Installation

```bash
uv add small-mcap

# With compression support
uv add small-mcap[compression]  # ZSTD + LZ4
uv add small-mcap[zstd]         # ZSTD only
uv add small-mcap[lz4]          # LZ4 only
```

## Reader

### Basic read

```python
from small_mcap import read_message

with open("input.mcap", "rb") as f:
    for schema, channel, message in read_message(f):
        print(f"{channel.topic}: {message.data}")
```

### Read multiple inputs

```python
from small_mcap import read_message

with open("recording1.mcap", "rb") as f1, \
     open("recording2.mcap", "rb") as f2, \
     open("recording3.mcap", "rb") as f3:
    for schema, channel, message in read_message([f1, f2, f3]):
        print(f"{channel.topic}: {message.log_time}")
```

### Read with topic filtering

```python
from small_mcap import read_message, include_topics

with open("input.mcap", "rb") as f:
    topics = ["/camera/image", "/lidar/points"]
    for schema, channel, message in read_message(f, should_include=include_topics(topics)):
        print(f"{channel.topic}: {len(message.data)} bytes")
```

### Read with time range

```python
from small_mcap import read_message

with open("input.mcap", "rb") as f:
    start = 1000000000  # nanoseconds
    end = 2000000000
    for schema, channel, message in read_message(f, start_time_ns=start, end_time_ns=end):
        print(f"{channel.topic} at {message.log_time}")
```

### Read decoded messages

```python
from small_mcap import read_message_decoded
import json

class JsonDecoderFactory:
    def decoder_for(self, schema):
        if schema.encoding == "json":
            return lambda data: json.loads(data)
        return None

with open("input.mcap", "rb") as f:
    for msg in read_message_decoded(f, decoder_factories=[JsonDecoderFactory()]):
        print(f"{msg.channel.topic}: {msg.decoded_message}")
```

### Read summary/metadata

```python
from small_mcap import get_summary, get_header

with open("input.mcap", "rb") as f:
    summary = get_summary(f)
    print(f"Messages: {summary.statistics.message_count}")
    print(f"Duration: {summary.statistics.message_start_time} - {summary.statistics.message_end_time}")

    for channel in summary.channels.values():
        print(f"  {channel.topic}: {channel.message_encoding}")
```

## Writer

### Basic write

```python
from small_mcap import McapWriter

with open("output.mcap", "wb") as f:
    writer = McapWriter(f)
    writer.start(profile="", library="my-app")

    # Add schema
    schema_id = 1
    writer.add_schema(schema_id, "MySchema", "json", b'{"type": "object"}')

    # Add channel
    channel_id = 1
    writer.add_channel(channel_id, "/my/topic", "json", schema_id)

    # Add messages
    for i in range(100):
        writer.add_message(
            channel_id,
            log_time=i * 1000000,  # nanoseconds
            data=b'{"value": 42}',
            publish_time=i * 1000000
        )

    writer.finish()
```

### Write with compression

```python
from small_mcap import McapWriter, CompressionType

with open("output.mcap", "wb") as f:
    writer = McapWriter(
        f,
        compression=CompressionType.ZSTD,
        chunk_size=1024 * 1024  # 1MB chunks
    )
    writer.start(profile="", library="my-app")

    schema_id = 1
    writer.add_schema(schema_id, "MySchema", "json", b"{}")
    channel_id = 1
    writer.add_channel(channel_id, "/topic", "json", schema_id)

    for i in range(1000):
        writer.add_message(channel_id, log_time=i*1000, data=b"data", publish_time=i*1000)

    writer.finish()
```

### Write with encoder factory

```python
from small_mcap import McapWriter
import json

class JsonEncoderFactory:
    """Implements EncoderFactoryProtocol for JSON messages."""
    profile = ""
    encoding = "jsonschema"
    message_encoding = "json"

    def encoder_for(self, schema):
        return lambda msg: json.dumps(msg).encode()

with open("output.mcap", "wb") as f:
    writer = McapWriter(f, encoder_factory=JsonEncoderFactory())
    writer.start(profile="", library="my-app")

    schema_id = 1
    writer.add_schema(schema_id, "SensorData", "jsonschema", b'{"type": "object"}')

    channel_id = 1
    writer.add_channel(channel_id, "/sensor/data", "json", schema_id)

    for i in range(100):
        msg = {"timestamp": i, "value": i * 2}
        writer.add_message_encode(channel_id, i * 1000, msg, publish_time=i * 1000)

    writer.finish()
```

## Features

- Zero dependencies for core functionality
- Optional compression support (ZSTD, LZ4)
- Lazy chunk loading for efficient memory usage
- Topic and time-range filtering
- Automatic schema/channel registration
- CRC validation
- Fast summary/metadata access

## Performance

`small-mcap` is optimized for high-performance MCAP file reading with zero-copy operations and lazy chunk loading:

**Key Optimizations:**

- **Zero-copy memory access**: Uses `memoryview` to avoid unnecessary data copies
- **Lazy chunk loading**: Only decompresses chunks when needed
- **Parallel chunk decompression**: `num_workers` threads decompress chunks ahead of the reader (zstd/lz4 release the GIL)
- **Binary search**: Efficient time-range filtering using chunk indexes
- **Heap-based merging**: Optimal multi-file reading with automatic ID remapping

### Parallel Prefetch (`num_workers`)

Pass `num_workers` to `read_message` to decompress chunks in parallel using a thread pool. The main thread reads raw bytes sequentially while worker threads decompress ahead.

```python
with open("large.mcap", "rb") as f:
    for schema, channel, message in read_message(f, num_workers=4):
        ...
```

Benchmarked on the included nuScenes MCAP file (431 MB, 560 zstd chunks, 30,900 messages; median of 5 runs):

| Workers | Median time (s) | Msg/s   | Speedup |
|---------|------------------|---------|---------|
| 0       | 0.3878           | 79,675  | 1.00x   |
| 2       | 0.2017           | 153,223 | 1.92x   |
| 4       | 0.1357           | 227,727 | 2.86x   |
| 8       | 0.0920           | 335,839 | 4.22x   |

**Comparison with other libraries:**

| Feature              | small-mcap | mcap (official) | rosbags  | pybag    |
| -------------------- | ---------- | --------------- | -------- | -------- |
| Performance          | Fastest    | Fast            | Fast     | Moderate |
| Zero dependencies    | Yes        | No              | No       | No       |
| Non-seekable streams | Yes        | Yes             | No       | No       |
| Multi-file reading   | Yes        | No              | Yes      | Yes      |
| ROS1 support         | No         | No              | Yes      | No       |
| SQLite3 backend      | No         | No              | Yes      | No       |

## Benchmarks

Median runtime from `pytest-benchmark` on the included nuScenes dataset (`data/data/nuScenes-v1.0-mini-scene-0061-ros2.mcap`, 30,900 messages, 19.15s duration, 560 zstd chunks):

| Scenario | small-mcap | mcap (official) | rosbags | pybag |
|----------|------------|-----------------|---------|-------|
| Full read (seekable) | 399.4 ms | 493.4 ms | 429.9 ms | 521.4 ms |
| Full read (non-seekable) | 405.9 ms | 495.2 ms | - | - |
| Time-range filter (seekable) | 106.1 ms | 127.7 ms | 426.3 ms | 131.1 ms |
| Time-range filter (non-seekable) | 125.0 ms | 146.6 ms | - | - |
| Topic filter (seekable) | 375.9 ms | 458.9 ms | 397.6 ms | 451.9 ms |
| Topic filter (non-seekable) | 396.4 ms | 470.0 ms | - | - |

Note: `rosbags` and `pybag` require seekable streams and are skipped for the non-seekable cases.

**Summary:**

- `small-mcap` was fastest in all six scenarios
- **1.17-1.24x faster** than mcap (official) across all scenarios
- **1.06-4.02x faster** than rosbags where rosbags supports the scenario
- **1.20-1.31x faster** than pybag on seekable streams

## Links

- [MCAP Specification](https://mcap.dev/)
