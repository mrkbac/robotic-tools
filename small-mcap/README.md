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

## Links

- [MCAP Specification](https://mcap.dev/)
