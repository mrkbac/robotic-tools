# robo-ws-bridge

A Python library implementing the Foxglove WebSocket protocol for streaming robotics data.

## Installation

```bash
uv add robo-ws-bridge
```

## Features

- **WebSocketBridgeServer**: Async server for publishing robotics data over WebSocket
- **WebSocketBridgeClient**: Async client for subscribing to data streams
- Native channel streaming, time synchronization, playback control, subscriptions, and status messages
- Extensible handlers for parameters, services, client publishing, connection graphs, and assets

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

### Playback control

Playback control uses our native async server implementation; it does not depend on the Foxglove SDK.
Supplying the recording time range advertises the `playbackControl` capability; the server accepts
the `foxglove.sdk.v1` subprotocol by default.

```python
from robo_ws_bridge import (
    PlaybackCommand,
    PlaybackControlRequest,
    PlaybackState,
    PlaybackStatus,
    WebSocketBridgeServer,
)

server = WebSocketBridgeServer(playback_time_range=(start_time_ns, end_time_ns))

async def control(request: PlaybackControlRequest) -> PlaybackState:
    if request.seek_time is not None:
        await recording.seek(request.seek_time)
    await recording.set_speed(request.playback_speed)
    if request.playback_command is PlaybackCommand.PLAY:
        await recording.play()
    else:
        await recording.pause()
    return PlaybackState(
        status=PlaybackStatus.PLAYING if recording.is_playing else PlaybackStatus.PAUSED,
        current_time=recording.current_time,
        playback_speed=recording.speed,
        did_seek=request.seek_time is not None,
    )

server.on_playback_control(control)
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
