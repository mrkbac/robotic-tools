# websocket-bridge

A Python library implementing the Foxglove WebSocket protocol for streaming robotics data.

## Installation

```bash
uv add websocket-bridge
```

## Features

- **WebSocketBridgeServer**: Async server for publishing robotics data over WebSocket
- **WebSocketBridgeClient**: Async client for subscribing to data streams
- Full support for Foxglove WebSocket protocol messages (advertise, subscribe, publish, etc.)

## Usage

### Server Example

```python
import asyncio
from websocket_bridge import WebSocketBridgeServer, Channel

async def main():
    server = WebSocketBridgeServer(name="my-server")

    # Define a channel
    channel = Channel(
        id=1,
        topic="/sensor/data",
        encoding="json",
        schema_name="SensorData",
        schema='{"type": "object"}'
    )

    # Start server and advertise channel
    await server.start("0.0.0.0", 8765)
    server.set_channels([channel])

    # Publish messages
    while True:
        await server.send_message(channel.id, timestamp_ns, data)
        await asyncio.sleep(0.1)

asyncio.run(main())
```

### Client Example

```python
import asyncio
from websocket_bridge import WebSocketBridgeClient

async def main():
    client = WebSocketBridgeClient("ws://localhost:8765")

    @client.on_message
    async def handle_message(channel, timestamp, data):
        print(f"{channel['topic']}: {data}")

    @client.on_advertise
    async def handle_advertise(channel):
        await client.subscribe(channel["id"])

    await client.connect()
    await client.run_forever()

asyncio.run(main())
```

## Protocol Reference

See the [Foxglove WebSocket Protocol](https://docs.foxglove.dev/docs/connecting-to-data/frameworks/ws-protocol/) for protocol details.
