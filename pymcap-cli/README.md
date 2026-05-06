# pymcap-cli

A high-performance Python CLI for MCAP file processing with advanced recovery, filtering, and optimization capabilities.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="vhs/info-light.gif">
  <source media="(prefers-color-scheme: dark)" srcset="vhs/info.gif">
  <img src="vhs/info.gif" alt="pymcap-cli info" />
</picture>

## Installation

```bash
# Run directly without installing
uvx pymcap-cli info data.mcap

# Or add to your project
uv add pymcap-cli

# With video support (for video and roscompress commands)
uv add pymcap-cli[video]
```

## Why pymcap-cli over the official Go CLI?

- **Advanced Recovery** — handles corrupt MCAP files with intelligent chunk-level recovery and MessageIndex validation
- **Smart Chunk Copying** — fast chunk copying without decompression when possible, up to 10x faster for filtering operations
- **Unified Processing** — single `process` command combines recovery + filtering + compression in one optimized pass
- **Precise Filtering** — regex topic filtering, time range filtering, and content type filtering with deferred schema/channel writing
- **Broad Format Coverage** — converts ROS 1 `.bag` and ROS 2 `.db3` to MCAP, exports to NDJSON, CSV, Parquet, PCD, GeoJSON/KML/GPX, and image/video files
- **Rich Terminal Output** — colored topics, Unicode distribution histograms, tree views, and responsive layouts
- **Robust Error Handling** — graceful degradation with detailed error reporting and recovery statistics

## Commands

### `info` — File Information

Display detailed MCAP file information including schemas, channels, message counts, time ranges, and per-topic distribution histograms.

```bash
pymcap-cli info data.mcap
```

Use `--tree` to group topics into a hierarchical tree view:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="vhs/info-tree-light.gif">
  <source media="(prefers-color-scheme: dark)" srcset="vhs/info-tree.gif">
  <img src="vhs/info-tree.gif" alt="pymcap-cli info --tree" />
</picture>

```bash
# Multiple files
pymcap-cli info file1.mcap file2.mcap file3.mcap

# JSON output
pymcap-cli info-json data.mcap
```

### `cat` — Stream Messages

Stream MCAP messages to stdout. Outputs as Rich tables when interactive, JSONL when piped.

Use `--query` to extract nested fields from deeply structured ROS messages with JSONPath-like syntax:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="vhs/cat-query-light.gif">
  <source media="(prefers-color-scheme: dark)" srcset="vhs/cat-query.gif">
  <img src="vhs/cat-query.gif" alt="pymcap-cli cat --query" />
</picture>

```bash
# Display messages in a table
pymcap-cli cat recording.mcap

# Filter specific topics
pymcap-cli cat recording.mcap --topics /camera/image

# Filter by time range
pymcap-cli cat recording.mcap --start-secs 10 --end-secs 20

# Limit output
pymcap-cli cat recording.mcap --limit 100

# Query specific field using message path
pymcap-cli cat recording.mcap --query '/odom.pose.position.x'

# Filter array elements
pymcap-cli cat recording.mcap --query '/detections.objects[:]{confidence>0.8}'

# Pipe to file as JSONL
pymcap-cli cat recording.mcap > messages.jsonl

# Write to file with progress bar
pymcap-cli cat recording.mcap -o messages.jsonl

# Control binary field serialization
pymcap-cli cat recording.mcap --bytes base64   # base64-encoded
pymcap-cli cat recording.mcap --bytes skip     # omit binary fields
```

### `tftree` — TF Transform Tree

Visualize the ROS TF transform tree with colored static/dynamic transforms, translation and rotation values.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="vhs/tftree-light.gif">
  <source media="(prefers-color-scheme: dark)" srcset="vhs/tftree.gif">
  <img src="vhs/tftree.gif" alt="pymcap-cli tftree" />
</picture>

```bash
# Show complete TF tree (both /tf and /tf_static)
pymcap-cli tftree data.mcap

# Show only static transforms
pymcap-cli tftree data.mcap --static-only
```

### `diag` — ROS2 Diagnostics

Inspect ROS2 diagnostics with per-component health overview, sparkline timelines, frequency stats, and time-in-state tracking.

```bash
# Show components with issues (WARN/ERROR/STALE)
pymcap-cli diag recording.mcap

# Show all components including OK
pymcap-cli diag recording.mcap --all

# Detailed inspection of specific components
pymcap-cli diag recording.mcap --inspect "encoder"

# Hierarchical tree view
pymcap-cli diag recording.mcap --tree

# JSON output for scripting
pymcap-cli diag recording.mcap --json
```

### `plot` — Time-Series Visualization

Plot message fields over time using Plotly. Supports named labels, LTTB downsampling, XY trajectory mode, and saves to interactive HTML. Requires the `plot` extra (`uv add pymcap-cli[plot]`).

```bash
# Plot a single field
pymcap-cli plot recording.mcap /odom.pose.position.x

# Named series
pymcap-cli plot recording.mcap "Vel X=/odom.twist.twist.linear.x"

# XY trajectory plot
pymcap-cli plot recording.mcap --xy /odom.pose.position.x /odom.pose.position.y

# Downsample to 1000 points and save to file
pymcap-cli plot recording.mcap /odom.pose.position.x -d 1000 -o plot.html
```

### `process` — Unified Processing

The most powerful command — combines recovery, filtering, and optimization in a single pass.

```bash
# Filter by topic regex
pymcap-cli process data.mcap -o filtered.mcap -y "/camera.*" -y "/lidar.*"

# Time range filtering (nanoseconds or RFC3339)
pymcap-cli process data.mcap -o subset.mcap -S "2022-01-01T00:00:00Z" -E "2022-01-01T01:00:00Z"

# Exclude topics and metadata
pymcap-cli process data.mcap -o clean.mcap -n "/debug.*" --exclude-metadata

# Change compression with filtering
pymcap-cli process zstd.mcap -o lz4.mcap --compression lz4 -y "/important.*"

# Recovery mode with filtering (handles corrupt files)
pymcap-cli process corrupt.mcap -o recovered.mcap -y "/camera.*" --recovery-mode
```

### `recover` — Advanced Recovery

Recover data from potentially corrupt MCAP files with intelligent error handling.

```bash
# Basic recovery
pymcap-cli recover corrupt.mcap -o fixed.mcap

# Force chunk decoding for maximum recovery
pymcap-cli recover corrupt.mcap -o fixed.mcap --always-decode-chunk
```

### `recover-inplace` — In-Place Recovery

Rebuild an MCAP file's summary and footer in place without creating a new file.

```bash
# Rebuild summary/footer in place
pymcap-cli recover-inplace data.mcap

# With exact size calculation
pymcap-cli recover-inplace data.mcap --exact-sizes

# Skip confirmation prompt
pymcap-cli recover-inplace data.mcap --force
```

### `merge` — Merge Files

Merge multiple MCAP files chronologically into a single output file.

```bash
# Merge two files
pymcap-cli merge recording1.mcap recording2.mcap -o combined.mcap

# Merge with compression
pymcap-cli merge *.mcap -o all_recordings.mcap --compression lz4

# Exclude metadata/attachments
pymcap-cli merge file1.mcap file2.mcap -o merged.mcap --metadata exclude
```

### `convert` — Convert DB3 to MCAP

Convert ROS2 DB3 (SQLite) bag files to MCAP format.

```bash
# Basic conversion
pymcap-cli convert input.db3 -o output.mcap

# Specify ROS distro
pymcap-cli convert input.db3 -o output.mcap --distro jazzy

# With custom message definitions
pymcap-cli convert input.db3 -o output.mcap --extra-path /path/to/msgs
```

### `bag2mcap` — Convert ROS 1 Bag to MCAP

Convert ROS 1 `.bag` files to MCAP using the `ros1` profile. Message bytes are
preserved as raw ROS 1 serialization and schemas use `ros1msg` encoding with
the full message definition from the bag.

```bash
# Basic conversion
pymcap-cli bag2mcap recording.bag -o recording.mcap

# Pick a different compression / chunk size
pymcap-cli bag2mcap recording.bag -o recording.mcap --compression lz4 --chunk-size 8388608
```

### `split` — Split into Segments

Split an MCAP file into multiple output segments by duration, explicit
timestamps, or value-change of a message-path expression.

```bash
# Split every 60 seconds
pymcap-cli split data.mcap --duration 60s -t "out_{index:03d}.mcap"

# Split at specific RFC3339 timestamps
pymcap-cli split data.mcap --split-at "2024-01-01T10:00:00Z" --split-at "2024-01-01T10:30:00Z"

# Start a new segment when /gps/fix.status.status changes value
pymcap-cli split data.mcap -E "/gps/fix.status.status"

# Predicate trigger — split on match/no-match transitions
pymcap-cli split data.mcap -E "/detections.objects[:]{confidence>0.8}"
```

### `rechunk` — Topic-Based Rechunking

Reorganize MCAP messages into separate chunk groups based on topic patterns for optimized playback.

```bash
# Group camera and lidar topics into separate chunks
pymcap-cli rechunk data.mcap -o rechunked.mcap -p "/camera.*" -p "/lidar.*"

# Multiple patterns — each gets its own chunk group
pymcap-cli rechunk data.mcap -o rechunked.mcap \
  -p "/camera/front.*" \
  -p "/camera/rear.*" \
  -p "/lidar.*" \
  -p "/radar.*"
```

### `filter` — Topic Filtering

Filter messages by topic patterns (simpler version of `process`).

```bash
# Include specific topics
pymcap-cli filter data.mcap -o filtered.mcap --include-topics "/camera/image" "/lidar/points"

# Exclude topics
pymcap-cli filter data.mcap -o filtered.mcap --exclude-topics "/debug.*" "/test.*"
```

### `compress` — Compression Tool

Change MCAP file compression.

```bash
pymcap-cli compress input.mcap -o output.mcap --compression zstd
pymcap-cli compress input.mcap -o output.mcap --compression lz4
```

### `du` — Disk Usage Analysis

Analyze MCAP file size breakdown by chunks, schemas, channels, and message counts.

```bash
pymcap-cli du large.mcap
```

### `list` — List Records

List various record types in an MCAP file.

```bash
pymcap-cli list channels data.mcap
pymcap-cli list chunks data.mcap
pymcap-cli list schemas data.mcap
pymcap-cli list attachments data.mcap
pymcap-cli list metadata data.mcap
```

### `diff` — Compare Files

Compare MCAP files using summary and message-index timestamps. Reads through the
footer/summary first and falls back to rebuilding metadata from the data section
when the summary is missing.

```bash
# Compare two recordings
pymcap-cli diff a.mcap b.mcap

# Hide channels with identical timestamps
pymcap-cli diff a.mcap b.mcap --skip-identical

# Show more timestamp ranges per channel
pymcap-cli diff a.mcap b.mcap --max-ranges 10
```

### `duplicates` — Find Duplicate Recordings

Scan files and directories for likely duplicate MCAP recordings using summary
and message-index fingerprints.

```bash
# Scan a directory tree
pymcap-cli duplicates /data/recordings

# Include singleton groups
pymcap-cli duplicates /data/recordings --all

# Rebuild summaries for files missing them
pymcap-cli duplicates /data/recordings --rebuild-missing
```

### `records` — Raw Record Dump

Print every MCAP record in file order using its `repr`. Useful for inspecting
raw file structure when debugging readers/writers.

```bash
pymcap-cli records data.mcap
```

### `topic-chunks` — Topic/Chunk Layout

Show which topics appear in which chunks, sorted by chunk count and percentage
of total chunks. Helps identify topics that would benefit from `rechunk`.

```bash
pymcap-cli topic-chunks data.mcap
```

### `video` — Video Generation

Generate MP4 videos from image topics using hardware-accelerated encoding. Requires the `video` extra.

```bash
# Basic video generation
pymcap-cli video data.mcap --topic /camera/front --output front.mp4

# With quality preset
pymcap-cli video data.mcap --topic /camera/rear --output rear.mp4 --quality high

# Use specific codec and encoder
pymcap-cli video data.mcap --topic /lidar/image --output lidar.mp4 --codec h265 --encoder videotoolbox
```

### `roscompress` — ROS Image Compression

Compress ROS MCAP files by converting CompressedImage/Image topics to CompressedVideo format and PointCloud2 topics to Cloudini or Draco compressed point clouds.

```bash
# Basic compression
pymcap-cli roscompress data.mcap -o compressed.mcap

# Specify quality and codec
pymcap-cli roscompress data.mcap -o compressed.mcap --quality 28 --codec h265

# Draco point cloud compression using the Foxglove compressed point cloud schema
pymcap-cli roscompress data.mcap -o compressed.mcap --pc-format draco --pc-schema foxglove
```

### `rosdecompress` — ROS Decompression

Decompress CompressedVideo, CompressedPointCloud2, and Foxglove CompressedPointCloud topics back to standard ROS formats.

```bash
# Decompress to CompressedImage (JPEG)
pymcap-cli rosdecompress input.mcap output.mcap

# Decompress to raw Image
pymcap-cli rosdecompress input.mcap output.mcap --video-format raw

# Skip point cloud decompression
pymcap-cli rosdecompress input.mcap output.mcap --no-pointcloud
```

### `export-images` — Image Files

Export image topics to per-topic folders of image files. `CompressedImage`
payloads keep their original encoding by default (`--format native`); set
`--format` to an `imagecodecs` encoder name (e.g. `jpeg`, `png`, `webp`) to
re-encode. Raw `Image` messages always use `--raw-format` (default `png`).
Requires the `image` extra.

```bash
# Native passthrough for CompressedImage; PNG for raw Image
pymcap-cli export-images data.mcap -o ./images -t /camera/front

# Force re-encoding to JPEG for everything
pymcap-cli export-images data.mcap -o ./images --format jpeg
```

### `export-csv` — CSV Files

Export an MCAP file to a directory of CSV files (one per topic). Nested fields
are flattened with dot notation (`pose.position.x`); arrays remain JSON
strings to preserve row counts. Schemas with raw media payloads (`Image`,
`CompressedImage`, …) are skipped unless `--include-blobs` is set.

```bash
pymcap-cli export-csv data.mcap -o ./csv
pymcap-cli export-csv data.mcap -o ./csv -t /odom -t /imu
```

### `export-json` — NDJSON / Per-Message JSON

Export an MCAP file to NDJSON (one line per message) or per-message JSON
files. Default writes one `<topic>.ndjson` per topic; with `--per-message`
each topic gets a directory of `<log_time_ns>.json` files — handy for
downstream tools that expect one record per file.

```bash
# One NDJSON per topic
pymcap-cli export-json data.mcap -o ./ndjson

# One JSON file per message
pymcap-cli export-json data.mcap -o ./json --per-message
```

### `export-parquet` — Parquet Files

Export an MCAP file to a directory of Parquet files (one per topic). Requires
the `parquet` extra.

```bash
pymcap-cli export-parquet data.mcap -o ./parquet
pymcap-cli export-parquet data.mcap -o ./parquet --compression snappy
```

### `export-pcd` — Point Cloud Files

Export `sensor_msgs/PointCloud2` topics to ASCII PCD v0.7 files
(`<output>/<safe_topic>/<log_time_ns>.pcd`) — readable by `pcl_viewer`,
Open3D, and CloudCompare. Requires the `pointcloud` extra.

```bash
pymcap-cli export-pcd data.mcap -o ./pcd
pymcap-cli export-pcd data.mcap -o ./pcd -t /lidar/points
```

### `export-geo` — Map Formats

Export geographic topics (`NavSatFix`, `geographic_msgs/*`) to GeoJSON, KML,
or GPX. GeoJSON writes one `<topic>.geojson` per topic; KML and GPX produce
a single `export.{kml,gpx}` covering all topics. Local-frame poses
(`Odometry`, `geometry_msgs/Pose*`) are out of scope — they need a datum.

```bash
# Default GeoJSON, track + points per topic
pymcap-cli export-geo data.mcap -o ./geo

# GPX track every 5th sample
pymcap-cli export-geo data.mcap -o ./geo --format gpx --mode track --stride 5

# Keep NO_FIX samples too
pymcap-cli export-geo data.mcap -o ./geo --include-no-fix
```

### Shell Autocompletion

```bash
# Automatically install completion for your current shell
pymcap-cli --install-completion

# Or manually for a specific shell
eval "$(pymcap-cli --show-completion bash)"   # bash
eval "$(pymcap-cli --show-completion zsh)"    # zsh
pymcap-cli --show-completion fish | source    # fish
```

## Common Use Cases

```bash
# Remove debug topics and compress
pymcap-cli process raw.mcap -o clean.mcap \
  -n "/debug.*" -n "/test.*" --exclude-metadata --compression zstd

# Extract camera data with time range
pymcap-cli process full_log.mcap -o camera.mcap \
  -y "/camera.*" -S "2024-01-01T10:00:00Z" -E "2024-01-01T11:00:00Z"

# Recover corrupt file and compress in one pass
pymcap-cli process corrupt.mcap -o recovered.mcap --recovery-mode --compression lz4

# Fast filtering with chunk copying (up to 10x faster)
pymcap-cli process 100gb_file.mcap -o filtered.mcap \
  -y "/lidar.*" --chunk-copying --compression zstd

# Optimize for topic-specific playback
pymcap-cli rechunk robot_log.mcap -o optimized.mcap \
  -p "/camera.*" -p "/lidar.*" -p "/imu.*" -p "/gps.*"
```

## Technical Details

- **Smart Chunk Processing** — automatically chooses between fast chunk copying and individual record processing based on filter criteria
- **MessageIndex Validation** — validates and rebuilds MessageIndexes when necessary for data integrity
- **Deferred Schema Writing** — only writes schemas and channels that are actually used by included messages
- **Compression Support** — zstd, lz4, and uncompressed formats with configurable chunk sizes
- **Memory Efficient** — streams processing with configurable buffer sizes for handling large files
- **Error Recovery** — multiple fallback strategies for handling corrupt or incomplete MCAP files

## Development

```bash
# Setup development environment
uv sync

# Run locally during development
uv run pymcap-cli --help

# Format and lint code
uv run pre-commit run --all-files

# Run tests
uv run pytest pymcap-cli/tests
```
