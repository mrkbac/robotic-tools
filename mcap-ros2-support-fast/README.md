# mcap-ros2-support-fast

High-performance pure Python ROS2 message serialization and deserialization for MCAP files.

No dependencies on ROS2 or a ROS2 environment.

## Installation

```bash
uv add mcap-ros2-support-fast
```

## Usage

### Decoding Messages

```python
from small_mcap import read_message_decoded
from mcap_ros2_support_fast.decoder import DecoderFactory

decoder_factory = DecoderFactory()

with open("recording.mcap", "rb") as f:
    for msg in read_message_decoded(f, decoder_factories=[decoder_factory]):
        print(f"{msg.channel.topic}: {msg.decoded_message}")
```

### Encoding Messages

```python
from small_mcap import McapWriter
from mcap_ros2_support_fast import ROS2EncoderFactory

encoder_factory = ROS2EncoderFactory()

with open("output.mcap", "wb") as f:
    writer = McapWriter(f, encoder_factory=encoder_factory)
    writer.start(profile="ros2")

    schema_id = 1
    writer.add_schema(
        schema_id,
        "geometry_msgs/msg/Point",
        "ros2msg",
        b"float64 x\nfloat64 y\nfloat64 z",
    )
    channel_id = 1
    writer.add_channel(channel_id, "/point", "cdr", schema_id)

    point = {"x": 1.0, "y": 2.0, "z": 3.0}
    writer.add_message_encode(channel_id, log_time=0, data=point, publish_time=0)

    writer.finish()
```

## Benchmarks

Median runtime from the non-`slow` `pytest-benchmark` suite on the included nuScenes dataset (`data/data/nuScenes-v1.0-mini-scene-0061-ros2.mcap`, 30,900 messages).

### Read Performance

| Messages | mcap-ros2-support-fast | rosbags | pybag | mcap-ros2 |
|----------|-------------------------|---------|-------|-----------|
| 10 | 5.46 ms | 7.47 ms | 25.56 ms | 365.70 ms |
| 100 | 36.44 ms | 67.04 ms | 110.68 ms | 1324.51 ms |
| 1000 | 127.68 ms | 234.15 ms | - | 4210.95 ms |

Note: `pybag` is skipped at 1000 messages because it cannot decode `diagnostic_msgs/DiagnosticArray`.

**Read Summary (median time):**

- **33-67x faster** than reference mcap-ros2
- **1.37-1.84x faster** than rosbags
- **3.04-4.68x faster** than pybag (10-100 msgs; pybag skipped at 1000 due to decoder limitations)

### Write Performance (Read + Write)

| Messages | mcap-ros2-support-fast | rosbags | pybag | mcap-ros2 |
|----------|-------------------------|---------|-------|-----------|
| 10 | 17.36 ms | 18.33 ms | 56.32 ms | 928.81 ms |
| 100 | 119.05 ms | 129.59 ms | 253.93 ms | 3132.39 ms |
| 1000 | 397.07 ms | 460.15 ms | - | 9193.86 ms |

**Read+Write Summary (median time):**

- **23.2-53.5x faster** than reference mcap-ros2
- **1.06-1.16x faster** than rosbags
- **2.13-3.24x faster** than pybag (10-100 msgs; pybag skipped at 1000 due to decoder limitations)

### Write-Only Performance

| Messages | mcap-ros2-support-fast | rosbags | pybag | mcap-ros2 |
|----------|-------------------------|---------|-------|-----------|
| 10 | 11.34 ms | 10.91 ms | 30.67 ms | 576.93 ms |
| 100 | 66.75 ms | 53.50 ms | 104.61 ms | 794.56 ms |
| 1000 | 224.47 ms | 177.82 ms | - | 1357.78 ms |

**Write-Only Summary (median time):**

- **6.0-50.9x faster** than reference mcap-ros2
- **0.96-1.26x** similar to rosbags (rosbags remains slightly faster for write-only)
- **1.57-2.71x faster** than pybag (10-100 msgs; pybag skipped at 1000 due to decoder limitations)
