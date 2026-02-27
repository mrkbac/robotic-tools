# robo-ws-bridge

A Python library implementing the Foxglove WebSocket protocol for streaming robotics data.

## Installation

```bash
uv add robo-ws-bridge
```

## Features

- **WebSocketBridgeServer**: Async server for publishing robotics data over WebSocket
- **WebSocketBridgeClient**: Async client for subscribing to data streams
- Full support for Foxglove WebSocket protocol messages (advertise, subscribe, publish, etc.)

## Usage

### Server Example

```python
import asyncio
from robo_ws_bridge.server import Channel
from robo_ws_bridge import WebSocketBridgeServer

async def main():
    server = WebSocketBridgeServer(host="0.0.0.0", port=8765, name="my-server")

    # Define and advertise a channel
    channel = Channel(
        id=1,
        topic="/sensor/data",
        encoding="json",
        schema_name="SensorData",
        schema='{"type": "object"}',
    )

    await server.start()
    await server.advertise_channel(channel)

    # Publish messages
    while True:
        data = b'{"temperature": 22.5}'
        await server.publish_message(channel.id, data)
        await asyncio.sleep(0.1)

asyncio.run(main())
```

### Client Example

```python
import asyncio
from robo_ws_bridge import WebSocketBridgeClient

async def main():
    client = WebSocketBridgeClient("ws://localhost:8765")

    async def handle_message(channel, timestamp, data):
        print(f"{channel['topic']}: {data}")

    async def handle_advertise(channel):
        await client.subscribe(channel["topic"])

    client.on_message(handle_message)
    client.on_advertised_channel(handle_advertise)

    await client.connect()

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await client.disconnect()

asyncio.run(main())
```

## Protocol Reference

See the [Foxglove WebSocket Protocol](https://docs.foxglove.dev/docs/connecting-to-data/frameworks/ws-protocol/) for protocol details.
