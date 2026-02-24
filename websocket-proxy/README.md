# websocket-proxy

A proxy for [Foxglove Bridge](https://github.com/foxglove/foxglove-sdk/tree/main/ros/src/foxglove_bridge) that converts messages just-in-time and forwards them via the Foxglove Bridge protocol.

## Installation

```bash
# Run directly without installing
uvx --from websocket-proxy bridge ws://localhost:8765

# Or add to your project
uv add websocket-proxy
```

## Features

- Converts `sensor_msgs/msg/CompressedImage` and `sensor_msgs/msg/Image` to `foxglove_msgs/msg/CompressedVideo`
  - Using ffmpeg: h264, h265, vp9, av1
  - Downscaling support
- Converts `sensor_msgs/msg/PointCloud2` to `point_cloud_interfaces/msg/CompressedPointCloud2` using [cloudini](https://github.com/facontidavide/cloudini)
  - Downsampling support (dropping random points, voxelize grid)
- Throttling support for all topics
- Prevents flooding traffic by awaiting Ping/Pong

## Usage

```bash
bridge <source-ws>
```

This starts the bridge connecting to `<source-ws>` and opens the proxy at `ws://0.0.0.0:8766`.

Default settings:
- Image downsampling to HD and h264 compression
- Cloudini point cloud compression
- Throttling to 1Hz
