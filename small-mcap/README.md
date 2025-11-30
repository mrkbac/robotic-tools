# small-mcap

Lightweight Python library for reading and writing MCAP files.

## Installation

```bash
pip install small-mcap

# With compression support
pip install small-mcap[compression]  # ZSTD + LZ4
pip install small-mcap[zstd]         # ZSTD only
pip install small-mcap[lz4]          # LZ4 only
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
    schema_id = writer.add_schema("MySchema", "json", b'{"type": "object"}')

    # Add channel
    channel_id = writer.add_channel("/my/topic", "json", schema_id=schema_id)

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

    schema_id = writer.add_schema("MySchema", "json", b"{}")
    channel_id = writer.add_channel("/topic", "json", schema_id=schema_id)

    for i in range(1000):
        writer.add_message(channel_id, log_time=i*1000, data=b"data", publish_time=i*1000)

    writer.finish()
```

### Write with encoder factory

```python
from small_mcap import McapWriter, EncoderFactory
import json

class JsonEncoder(EncoderFactory):
    def get_schema_encoding(self, schema_name):
        return "json", b'{"type": "object"}'

    def get_channel_encoding(self, topic):
        return "json"

    def encode(self, topic, msg):
        return json.dumps(msg).encode()

with open("output.mcap", "wb") as f:
    writer = McapWriter(f)
    writer.start(profile="", library="my-app")

    encoder = JsonEncoder()

    # Encoder automatically registers schemas and channels
    for i in range(100):
        msg = {"timestamp": i, "value": i * 2}
        writer.add_message_encoded("/sensor/data", i * 1000, msg, encoder, publish_time=i * 1000)

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
- **Binary search**: Efficient time-range filtering using chunk indexes
- **Heap-based merging**: Optimal multi-file reading with automatic ID remapping

**Comparison with other libraries:**

| Feature              | small-mcap | mcap (official) | rosbags | pybag       |
| -------------------- | ---------- | --------------- | ------- | ----------- |
| Performance          | ⚡ Fastest | ⚡ Fast         | ⚡ Fast | 🐌 Moderate |
| Zero dependencies    | ✅         | ❌              | ❌      | ❌          |
| Non-seekable streams | ✅         | ✅              | ❌      | ❌          |
| Multi-file reading   | ✅         | ❌              | ✅      | ✅          |
| ROS1 support         | ❌         | ❌              | ✅      | ❌          |
| SQLite3 backend      | ❌         | ❌              | ✅      | ❌          |

## Benchmarks

Benchmark results comparing small-mcap against mcap (official), rosbags, and pybag libraries using a nuScenes dataset (30,900 messages, 19.15s duration, 560 zstd chunks).

### Full File Read (Seekable)

```txt
----------------------------------------------------------------------------------------- benchmark 'full-seekable': 4 tests -----------------------------------------------------------------------------------------
Name (time in ms)                                       Min                   Max                  Mean             StdDev                Median                IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_read[full-seekable-small_mcap]      442.7987 (1.0)        455.6817 (1.0)        448.6568 (1.0)       4.7916 (1.0)        448.9002 (1.0)       6.6608 (1.0)           2;0  2.2288 (1.0)           5           1
test_benchmark_read[full-seekable-rosbags]         502.2698 (1.13)       523.9009 (1.15)       510.3689 (1.14)      8.6200 (1.80)       506.1880 (1.13)     11.7877 (1.77)          1;0  1.9594 (0.88)          5           1
test_benchmark_read[full-seekable-pybag]           559.9649 (1.26)       596.3682 (1.31)       578.5393 (1.29)     13.1715 (2.75)       581.2666 (1.29)     15.2660 (2.29)          2;0  1.7285 (0.78)          5           1
test_benchmark_read[full-seekable-mcap]            574.9254 (1.30)       614.3929 (1.35)       594.9063 (1.33)     15.2217 (3.18)       593.9697 (1.32)     21.7823 (3.27)          2;0  1.6809 (0.75)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Full File Read (Non-seekable Stream)

```txt
----------------------------------------------------------------------------------------- benchmark 'full-nonseekable': 2 tests ------------------------------------------------------------------------------------------
Name (time in ms)                                          Min                   Max                  Mean             StdDev                Median                IQR            Outliers     OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_read[full-nonseekable-small_mcap]       423.7839 (1.0)        454.8048 (1.0)        433.9779 (1.0)      12.7063 (1.0)        428.9654 (1.0)      14.9051 (1.0)           1;0  2.3043 (1.0)           5           1
test_benchmark_read[full-nonseekable-mcap]             595.2403 (1.40)       639.9618 (1.41)       616.2232 (1.42)     19.9259 (1.57)       607.5073 (1.42)     34.9823 (2.35)          2;0  1.6228 (0.70)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

Note: rosbags and pybag require seekable streams and are skipped for non-seekable tests.

### Time-Range Filtered Read (Seekable)

```txt
---------------------------------------------------------------------------------------- benchmark 'time-seekable': 4 tests ----------------------------------------------------------------------------------------
Name (time in ms)                                      Min                 Max                Mean            StdDev              Median               IQR            Outliers     OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_read[time-seekable-small_mcap]     119.7578 (1.0)      128.1060 (1.0)      123.7998 (1.0)      2.5372 (1.0)      123.7509 (1.0)      2.7058 (1.0)           3;0  8.0776 (1.0)           9           1
test_benchmark_read[time-seekable-pybag]          140.4680 (1.17)     159.2642 (1.24)     146.3533 (1.18)     6.4415 (2.54)     143.5709 (1.16)     6.2359 (2.30)          1;1  6.8328 (0.85)          7           1
test_benchmark_read[time-seekable-mcap]           146.3309 (1.22)     155.6906 (1.22)     150.6005 (1.22)     3.9600 (1.56)     150.2616 (1.21)     7.3775 (2.73)          2;0  6.6401 (0.82)          7           1
test_benchmark_read[time-seekable-rosbags]        509.0745 (4.25)     521.1191 (4.07)     512.3522 (4.14)     4.9971 (1.97)     510.4790 (4.13)     4.5978 (1.70)          1;1  1.9518 (0.24)          5           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

### Topic-Filtered Read (Seekable)

```txt
----------------------------------------------------------------------------------------- benchmark 'topic-seekable': 4 tests -----------------------------------------------------------------------------------------
Name (time in ms)                                       Min                 Max                Mean             StdDev              Median                IQR            Outliers     OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_read[topic-seekable-small_mcap]     442.4237 (1.0)      454.9705 (1.0)      446.8667 (1.0)       4.9801 (1.05)     445.4813 (1.0)       6.3691 (1.0)           1;0  2.2378 (1.0)           5           1
test_benchmark_read[topic-seekable-rosbags]        502.9512 (1.14)     514.8851 (1.13)     508.7413 (1.14)      4.7252 (1.0)      507.7358 (1.14)      7.3330 (1.15)          2;0  1.9656 (0.88)          5           1
test_benchmark_read[topic-seekable-pybag]          507.1222 (1.15)     536.2468 (1.18)     520.2659 (1.16)     12.1789 (2.58)     517.0470 (1.16)     20.4743 (3.21)          2;0  1.9221 (0.86)          5           1
test_benchmark_read[topic-seekable-mcap]           548.7598 (1.24)     560.8708 (1.23)     554.8000 (1.24)      5.2638 (1.11)     554.2846 (1.24)      9.4890 (1.49)          2;0  1.8025 (0.81)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

**Summary:**

- **1.13-1.42x faster** than mcap (official) across all scenarios
- **1.14-4.14x faster** than rosbags (especially for time-range filtering)
- **1.16-1.29x faster** than pybag for seekable streams

## Links

- [MCAP Specification](https://mcap.dev/)
