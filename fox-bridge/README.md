# FoxBridge

FoxBridge is a simple proxy of [Foxglove Bridge](https://github.com/foxglove/foxglove-sdk/tree/main/ros/src/foxglove_bridge)
which converts messages just in time and forwards via Foxglove Bridge compatible protocol.

## Features

- Converts `sensor_msgs/msg/CompressedImage` and `sensor_msgs/msg/Image` to [`foxglove_msgs/msg/CompressedVideo`](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video)
  - Using ffmpeg `h264`, `h265`, `vp9`, `av1`
  - Downscaling support
- Converts `sensor_msgs/msg/PointCloud2` to `point_cloud_interfaces/msg/CompressedPointCloud2` using [cloudini](https://github.com/facontidavide/cloudini)
  - Downsampling support (dropping random points, voxelize grid)
- Throttling support for all topics
- Prevent flooding traffic by awaiting for Ping/Pong

## usage

`uv run bridge <source-ws>`

This will start the bridge connecting to the bridge at `<source-ws>` and opening the proxy at `ws://0.0.0.0:8766`
Image downsampling to HD and h264 compression, cloudini point cloud compression and throttling to 1Hz is enabled by default.
