# Robotic Tools

A workspace of pure-Python tools for working with MCAP files, ROS 1/2 messages, and robotics data — built around a fast, batteries-included CLI.

## pymcap-cli

A high-performance Python CLI for slicing, dicing, recovering, and exporting MCAP files. Pure Python, no ROS runtime required.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="pymcap-cli/vhs/info-light.gif">
  <source media="(prefers-color-scheme: dark)" srcset="pymcap-cli/vhs/info.gif">
  <img src="pymcap-cli/vhs/info.gif" alt="pymcap-cli info" />
</picture>

```sh
# Try it without installing
uvx pymcap-cli info data.mcap

# Or add to your project
uv add pymcap-cli
```

**Why pymcap-cli?**

- **Advanced recovery** — repairs corrupt MCAP files via chunk-level recovery and MessageIndex validation
- **Smart chunk copying** — up to 10× faster filtering by copying compressed chunks without decompressing
- **Unified `process` command** — recovery, filtering, and recompression in a single optimized pass
- **Precise filtering** — regex topics, time ranges, and content-type filters with deferred schema/channel writing
- **Broad format coverage** — converts ROS 1 `.bag` and ROS 2 `.db3` to MCAP; exports to NDJSON, CSV, Parquet, PCD, GeoJSON / KML / GPX, image folders, and MP4
- **Rich terminal output** — colored topics, Unicode distribution histograms, tree views, responsive layouts

See the full command reference in [pymcap-cli/README.md](pymcap-cli/README.md).

## Supporting packages

The workspace also ships the libraries pymcap-cli is built on. Each is usable standalone.

| Package | What it does |
|---|---|
| [`small-mcap`](small-mcap/) | Lightweight MCAP reader/writer. Zero hard deps, high performance. |
| [`mcap-ros2-support-fast`](mcap-ros2-support-fast/) | Pure-Python ROS 2 CDR (de)serialization. 30–60× faster than the reference implementation. |
| [`ros-parser`](ros-parser/) | Parser for ROS 1 / ROS 2 message definitions and Foxglove message path syntax. |
| [`mcap-codec-support`](mcap-codec-support/) | Reusable MCAP encoder/decoder factories for video and point-cloud codecs. |
| [`pointcloud2`](pointcloud2/) | Helpers for `sensor_msgs/PointCloud2` access and conversion. |
| [`pureini`](pureini/) | Pure-Python implementation of the Cloudini point-cloud compression format. |
| [`robo-ws-bridge`](robo-ws-bridge/) | Python implementation of the Foxglove WebSocket protocol for streaming robotics data. |
| [`websocket-proxy`](websocket-proxy/) | Proxy in front of [Foxglove Bridge](https://github.com/foxglove/foxglove-sdk/tree/main/ros/src/foxglove_bridge) with just-in-time message conversion, image compression, and point-cloud compression. |

## License & credits

These tools draw inspiration from (and aim to interoperate with):

- <https://github.com/foxglove/mcap>
- <https://github.com/foxglove/ros-typescript>
