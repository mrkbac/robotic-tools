# Digitalis

Terminal-based visualization tool for robotics development.

![Digitalis](./screenshot.svg)

## Installation

```sh
# Run directly without installing
uvx digitalis recording.mcap

# Or add to your project
uv add digitalis
```

## Features

- Browse and visualize MCAP files and live WebSocket streams
- Specialized panels for common ROS message types:
  - Images (CompressedImage, Image)
  - Point clouds (PointCloud2)
  - TF transforms
  - NavSatFix (GPS)
  - Occupancy grids
  - Diagnostics
  - Raw message data (JSON)
- Time-based playback controls
- Topic filtering and search
- SSH-optimized mode (auto-detected)

## Data Sources

- Local MCAP files
- HTTP/HTTPS URLs to MCAP files
- WebSocket streams (Foxglove WebSocket protocol)

## Usage

```sh
uvx digitalis <path or websocket URL>
```

Examples:
```sh
# Open a local MCAP file
uvx digitalis recording.mcap

# Open from URL
uvx digitalis https://example.com/data.mcap

# Connect to a WebSocket stream
uvx digitalis ws://localhost:8765
```

## Controls

- `q` - Quit
- Arrow keys / mouse - Navigate topics and panels
- Space - Play/pause playback
- `/` - Search topics

## Debug

```sh
uv run textual run --dev digitalis.app:main
```
