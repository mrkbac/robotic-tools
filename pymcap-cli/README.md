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

# With video support (for video generation and ROS image compression)
uv add "pymcap-cli[video]"

# With ROS image and point-cloud compression support
uv add "pymcap-cli[video,pointcloud]"

# Add Draco point-cloud compression support
uv add "pymcap-cli[video,pointcloud,draco]"
```

The base install includes the CLI framework, MCAP compression support, ROS
schema parsing/decoding, configuration paths, and YAML handling because those
are used across the core inspect and transform commands. Feature-specific
binary and web stacks remain optional:

| Extra | Enables | Why it is optional |
| --- | --- | --- |
| `bridge` | Foxglove WebSocket client, playback, and serving | Network-specific workflow |
| `bridge-codecs` | JIT video and point-cloud bridge presets | Composes `bridge`, `video`, and `pointcloud` |
| `bridge-proxy` | Live video and point-cloud transforming proxy | Reuses `bridge-codecs` |
| `video` | Video export, compression, and decompression | PyAV and Pillow binary wheels |
| `pointcloud` | PCD export and CloudINI processing | Native CloudINI codec |
| `draco` | Draco point-cloud processing | DracoPy and NumPy binary wheels |
| `image` | Image export | Pillow is only needed by image workflows |
| `plot` | Interactive and static plots | Plotly and Kaleido |
| `xxhash` | Stable index fingerprints | Only index and hashing features require it |
| `lite` | Image, Draco, bridge, and index features | Compact bundle without video, CloudINI, or plotting |
| `all` | Every supported optional feature | Full feature set |

Each extra is tested from the built wheel in an isolated environment. Adding a
new optional dependency requires assigning it to a feature module in the import
contracts and adding its promised command to that wheel matrix.

## Why pymcap-cli over the official Go CLI?

- **Advanced Recovery** — handles corrupt MCAP files with intelligent chunk-level recovery and MessageIndex validation
- **Smart Chunk Copying** — fast chunk copying without decompression when possible, up to 10x faster for filtering operations
- **Unified Processing** — single `process` command combines recovery + filtering + compression in one optimized pass
- **Precise Filtering** — regex topic filtering, time range filtering, and content type filtering with deferred schema/channel writing
- **Broad Format Coverage** — converts ROS 1 `.bag` and ROS 2 `.db3` to MCAP, exports to NDJSON, CSV, PCD, GeoJSON/KML/GPX, and image/video files
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

### Common Message Filters

File-reading commands use the same selectors, including `cat`, `filter`,
`process`, `roscompress`, the exporters, and the diagnostic/TF readers:

```bash
# Exact topics (repeatable)
pymcap-cli export-json data.mcap -o ./json -t /odom -t /imu

# Regex selectors use full-match semantics; exclusions always win
pymcap-cli cat data.mcap -t '/camera/.*' -x '.*/debug'

# Inclusive start, exclusive end; recording-relative values are supported
pymcap-cli export-images data.mcap -o ./images --start @10s --end end-5s
```

Every `--topic` / `--exclude-topic` value is a regular expression evaluated
with full-match semantics. Thus `/camera/front` matches only that topic,
`/camera/.*` matches the camera namespace, and `.*camera.*` performs a
substring-style match. Escape regex metacharacters in arbitrary non-ROS MCAP
topic names when you mean them literally. See each command's `--help` for
domain-specific options.

#### Time-filter cheat sheet

`--start` / `-S` is inclusive. `--end` / `-E` is exclusive.

| Input | Meaning |
|---|---|
| `1234567890` | Absolute timestamp in nanoseconds |
| `20ns` | Absolute 20 nanoseconds |
| `500us` | Absolute 500 microseconds |
| `250ms` | Absolute 250 milliseconds |
| `20s` | Absolute 20 seconds |
| `5m` | Absolute 5 minutes |
| `1h` | Absolute 1 hour |
| `2026-07-13T12:00:00Z` | Absolute RFC3339 timestamp |
| `+1m` | One minute after recording start |
| `-1m` | One minute before recording end |
| `@1m` | Alias for `+1m` |
| `start+1m` | Explicitly one minute after recording start |
| `end-1m` | Explicitly one minute before recording end |

```bash
# Keep [10s, 20s) relative to recording start
-S +10s -E +20s

# Keep everything except the final 30 seconds
-E=-30s

# Keep the final minute
-S=-1m

# Absolute RFC3339 window
-S 2026-07-13T12:00:00Z -E 2026-07-13T12:10:00Z
```

Use `=` with negative shorthand so it is not mistaken for another option:

```bash
--start=-1m
--end=-30s
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

# Filter an exact topic
pymcap-cli cat recording.mcap --topic /camera/image

# Filter by recording-relative time range
pymcap-cli cat recording.mcap --start @10s --end @20s

# Limit output
pymcap-cli cat recording.mcap --limit 100

# Query specific field using message path
pymcap-cli cat recording.mcap --query '/odom.pose.position.x'

# Query multiple fields from one topic; labels become output keys
pymcap-cli cat recording.mcap \
  --query 'x=/odom.pose.position.x' \
  --query 'y=/odom.pose.position.y'

# Filter array elements
pymcap-cli cat recording.mcap --query '/detections.objects[:]{confidence>0.8}'

# Reduce an array in each message to one scalar
pymcap-cli cat recording.mcap --query '/joint_states.position.@max'

# Supply MessagePath variables on the command line
pymcap-cli cat recording.mcap --query '/temperature{>=$minimum}' --var minimum=-40

# Or reuse variables from the environment; --var overrides matching names
PYMCAP_VAR_minimum=-40 PYMCAP_VAR_maximum=125 \
  pymcap-cli cat recording.mcap --query '/temperature{>=$minimum && <=$maximum}'

# Pipe to file as JSONL
pymcap-cli cat recording.mcap > messages.jsonl

# Write to file with progress bar
pymcap-cli cat recording.mcap -o messages.jsonl

# Control binary field serialization
pymcap-cli cat recording.mcap --bytes base64   # base64-encoded
pymcap-cli cat recording.mcap --bytes skip     # omit binary fields
```

When multiple queries select the same topic, their values are combined into one
object per source message. Without labels, the relative expressions (for example
`.pose.position.x`) are used as output keys.

### `check` — Recording Contract Validation

Check the topics, schemas, encodings, timing, and decoded values in a recording
against a strict versioned YAML spec. Topic selectors are case-insensitive
regular expressions matched against the whole topic name. Warnings are shown
without causing a non-zero exit; errors exit with status 1.

```bash
pymcap-cli check recording.mcap --spec recording.yaml
```

```yaml
version: 1

topics:
  imu:
    topic: /imu
    schema:
      name: sensor_msgs/msg/Imu
      encoding: ros2msg
    message_encoding: cdr
    frequency:
      min: 95
      max: 105
      tolerance: 0.05
      window: 1s
    timeout: 50ms
    values:
      - '.linear_acceleration.@norm{<=30}'
      - '.header.frame_id{=="imu_link"}'
    live:
      publishers:
        min: 1
        max: 1
        node: /imu_driver
      subscribers:
        min: 1

  forbidden_front_radar:
    topic: /RADAR_FRONT
    expected: false
    severity: error

live:
  nodes:
    localization:
      node: /localization
      expected: true
```

The spec format is described by
[`schemas/mcap_check_spec.json`](schemas/mcap_check_spec.json); point your
editor at it for validation and completion:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/mrkbac/robotic-tools/main/pymcap-cli/schemas/mcap_check_spec.json
```

`expected` defaults to `true`, `severity` defaults to `error`, and frequency
tolerance defaults to zero. Predicate-ending MessagePaths are the preferred
value-check form: a matching value passes and an empty result fails. Mapping
rules with inclusive `min`/`max`, `equals`, or `one_of` remain available when a
predicate is not convenient or when reports need the rejected scalar value:

```yaml
values:
  - '.fields[:]{name == "z"}.@length{==1}'
  - '.@product(width, height){>=1000 && <=100000}'
  - path: .temperature
    min: -40
    max: 85
```

Cross-message modifiers use `@@`. Their state is isolated per concrete topic:

```yaml
values:
  - '.temperature.@@mean{>=15 && <=35}'
  - '.status{=="OK"}.@@timedelta.@@max{<=0.5}'
  - '.header.stamp.@to_nsec.@@unchanged_for.@@max{<=0.5}'
  - '.@@timedelta.@@stddev{<=0.005}'
```

Recorded stream timing uses MCAP log time; live checks use monotonic local
arrival time. Select a timestamp field and use `@@delta` when checking that
clock instead, such as
`.header.stamp.@to_nsec.@@delta.@@max{<=200000000}`.

Checks may reference `$log_time_ns`, `$publish_time_ns`,
`$recording_start_ns`, and `$recording_end_ns` as evaluation variables.
On files without summary statistics (e.g. torn, in-progress recordings) the
recording end is unknown while messages stream by, so rules referencing
`$recording_end_ns` report an evaluation error instead of a guessed value.

The repository includes a complete contract for its nuScenes fixture. From the
workspace root, run:

```bash
pymcap-cli check data/data/nuScenes-v1.0-mini-scene-0061-ros2.mcap \
  --spec pymcap-cli/examples/check/nuscenes.yaml
```

Use the same contract as a live preflight before recording:

```bash
pymcap-cli bridge check localhost --spec recording.yaml --duration 5
```

The recording command validates the shared topic rules and skips `live` constraints.
`bridge check` validates advertised topics, schemas, publisher/subscriber counts and
node identities, and samples only topics with frequency, timeout, or value rules.
Live graph constraints require the bridge `connectionGraph` capability. A node is
considered present when it publishes, subscribes, or provides a service in that graph.

### `doctor` — MCAP Container Validation

Check an MCAP file structure against the MCAP container specification, with
summary, index, chunk, message-order, and advisory findings.

```bash
pymcap-cli doctor data.mcap
pymcap-cli doctor data.mcap --strict-message-order --show-all
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

### `tf-get` — TF Transform Lookup

Resolve the transform from a source frame into a target frame using `/tf_static`
and `/tf`. Without `--at`, dynamic edges use their latest sample.

```bash
pymcap-cli tf-get data.mcap map base_link
pymcap-cli tf-get data.mcap odom base_link --at 2024-01-01T10:00:00Z
```

### `tf-export` — TF Tree to URDF / SDF / JSON

Reconstruct robot description files from `/tf_static` (and optionally `/tf` at a
snapshot timestamp). Useful when the original `.urdf` is missing — Foxglove
Studio and rviz can render the static skeleton from the exported file.

```bash
# Write a URDF for the static tree
pymcap-cli tf-export data.mcap -o robot.urdf

# SDF or JSON instead
pymcap-cli tf-export data.mcap --format sdf -o robot.sdf
pymcap-cli tf-export data.mcap --format json

# Capture a dynamic snapshot from /tf at a given time
pymcap-cli tf-export data.mcap --include-dynamic-at 2024-01-01T10:00:00Z -o snapshot.urdf

# Pick a subtree when the recording has multiple disconnected roots
pymcap-cli tf-export data.mcap --root base_link -o robot.urdf
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

### `plot` — Time-Series And Distribution Visualization

Plot message fields as time series, value histograms, categorical frequency bars, or XY trajectories using Plotly. Supports named labels, LTTB downsampling, interactive HTML, and static image output. Requires the `plot` extra (`uv add pymcap-cli[plot]`).

```bash
# Plot a single field
pymcap-cli plot recording.mcap /odom.pose.position.x

# Named series
pymcap-cli plot recording.mcap "Vel X=/odom.twist.twist.linear.x"

# XY trajectory plot
pymcap-cli plot recording.mcap --kind xy /odom.pose.position.x /odom.pose.position.y

# Numeric histogram with at most 40 bins
pymcap-cli plot recording.mcap /imu.linear_acceleration.x \
  --kind histogram --bins 40

# Categorical frequencies as probabilities
pymcap-cli plot recording.mcap /system.mode \
  --kind histogram --normalize probability

# Downsample to 1000 points and save to file
pymcap-cli plot recording.mcap /odom.pose.position.x -d 1000 -o plot.html
```

### `process` — Unified Processing

The most powerful command — combines recovery, filtering, and optimization in a single pass.

```bash
# Filter by topic regex
pymcap-cli process data.mcap -o filtered.mcap \
  -t '/camera/.*' -t '/lidar/.*'

# Filter decoded messages. Repeated predicates for one topic are ORed.
pymcap-cli process data.mcap -o alerts.mcap \
  --where '/diagnostics.status[:]{level >= 2}' \
  --where '/diagnostics.status[:]{message == "stale"}'

# Put AND conditions inside one MessagePath predicate.
pymcap-cli process data.mcap -o confident-cars.mcap \
  --where '/detections.objects[:]{confidence >= 0.8 && label == "car"}'

# Time range filtering (nanoseconds or RFC3339)
pymcap-cli process data.mcap -o subset.mcap -S "2022-01-01T00:00:00Z" -E "2022-01-01T01:00:00Z"

# Exclude topics and metadata
pymcap-cli process data.mcap -o clean.mcap \
  -x '/debug/.*' --metadata exclude

# Change compression with filtering
pymcap-cli process zstd.mcap -o lz4.mcap --compression lz4 \
  -t '/important/.*'

# Convert Jazzy QoS policy names to Humble-compatible integer codes
pymcap-cli process jazzy.mcap -o humble.mcap --qos-format numeric

# Embed standard ROS 2 per-topic QoS overrides
pymcap-cli process data.mcap -o qos-fixed.mcap --qos-override qos.yaml

# Apply repeatable regex overrides, then convert the result for Humble
pymcap-cli process data.mcap -o qos-fixed.mcap \
  --qos-set '/camera/.*:reliability=best_effort' \
  --qos-set '/camera/front:depth=3' --qos-format numeric

# Recovery mode with filtering (handles corrupt files)
pymcap-cli process corrupt.mcap -o recovered.mcap \
  -t '/camera/.*' --recovery-mode
```

`--where` only filters the topic named by its MessagePath; topics without a
predicate pass through unchanged. Use `--topic` when the output should contain
only the filtered topic. Predicate paths may reference `--var` values. Stream
modifiers (`@@`) are not supported by `--where`.

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

# Drop duplicate messages (same channel, log_time, payload) from overlapping inputs
pymcap-cli merge a.mcap b.mcap -o merged.mcap --dedup-identical
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
timestamps, paired events, value-change of a message-path expression, or a byte
budget per segment.

```bash
# Split every 60 seconds
pymcap-cli split data.mcap --duration 60s -t "out_{index:03d}.mcap"

# Split at specific RFC3339 timestamps
pymcap-cli split data.mcap --split-at "2024-01-01T10:00:00Z" --split-at "2024-01-01T10:30:00Z"

# Start a new segment when /gps/fix.status.status changes value
pymcap-cli split data.mcap -E "/gps/fix.status.status"

# Predicate trigger — split on match/no-match transitions
pymcap-cli split data.mcap -E "/detections.objects[:]{confidence>0.8}"

# Omit neutral runs and name files with the typed expression value
pymcap-cli split data.mcap \
  -E '/sensor/aramine/drive_state.drive_direction' \
  --skip-value 0 \
  -t 'drive_{value:+d}_{index:03d}.mcap'

# Split when each output reaches roughly 1 GB
pymcap-cli split data.mcap --max-size 1G -t "shard_{index:03d}.mcap"

# Keep inclusive windows between matching start and stop events
pymcap-cli split data.mcap \
  --window-start '/events/start{data == true}' \
  --window-end '/events/stop{data == true}' \
  --min-window 100ms --max-window 30s \
  -t 'event_{index:03d}_{window_start}_{window_end}.mcap'
```

Expression extractors must resolve to a primitive (`bool`, `int`, `float`, or
`str`). Filter expressions normalize to a boolean match/no-match value. Output
templates accept normal Python format specifications for typed fields such as
`{value:+d}` and `{index:03d}`.

Paired windows use a discovery pass before opening outputs and reject malformed
or out-of-range pairs by default. `--orphan-stop`, `--nested-start`,
`--unclosed-window`, and `--invalid-window` make each alternative policy
explicit. The local source must remain unchanged through the output pass.

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

### `filter` — Message Filtering

Filter messages by topic and time (simpler version of `process`).

```bash
# Include specific topics
pymcap-cli filter data.mcap -o filtered.mcap \
  -t /camera/image -t /lidar/points

# Exclude topics
pymcap-cli filter data.mcap -o filtered.mcap \
  -x '/debug/.*' -x '/test/.*'
```

### `compress` — Compression Tool

Change MCAP file compression.

```bash
pymcap-cli compress input.mcap -o output.mcap --compression zstd
pymcap-cli compress input.mcap -o output.mcap --compression lz4

# Compress in place: write to a temp file, validate it, then replace the source
pymcap-cli compress input.mcap --in-place --compression zstd

# Trade a little ratio for throughput: --fast (zstd fast mode), or pick a level
pymcap-cli compress input.mcap -o output.mcap --fast
pymcap-cli compress input.mcap -o output.mcap --compression-level -5
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
pymcap-cli list schema data.mcap --name sensor_msgs/msg/Image
pymcap-cli list attachments data.mcap
pymcap-cli list metadata data.mcap
```

### `msg` — ROS2 Message Definitions

Resolve, hash, search, list, and browse ROS2 `.msg` definitions. `msg def` prints
complete definitions including dependencies; `msg hash` prints the RIHS01
interface hash; `msg search` finds definitions in local/cache data (or an
explicit all-distro scan); `msg list` lists package message types; and `msg serve`
starts a local browser UI.

```bash
# Resolve a standard ROS2 message
pymcap-cli msg def sensor_msgs/msg/Image --distro humble

# Print only the root definition, with comments and constants removed
pymcap-cli msg def sensor_msgs/msg/Image --root-only --compact

# Include custom package roots before AMENT_PREFIX_PATH and the user cache
pymcap-cli msg def my_robot_msgs/msg/Status -I ./install/share

# Calculate the ROS interface hash
pymcap-cli msg hash sensor_msgs/msg/PointCloud2 --distro jazzy

# Find and display a definition from local/cache sources
pymcap-cli msg search pointcloud2 --show-definition

# Search every package in a distro when local/cache sources are insufficient
pymcap-cli msg search PointCloud2 --remote --show-definition

# List messages in a package or browse definitions locally
pymcap-cli msg list sensor_msgs --distro jazzy
pymcap-cli msg serve --distro jazzy --no-browser
```

Missing standard packages are resolved from rosdistro/GitHub and cached under
the `pymcap_cli_msg_def` user cache. Bare `msg search` queries search supplied
paths, `AMENT_PREFIX_PATH`, and that cache without downloading every package;
use `--remote` for an explicit all-package scan.

### `get` — Extract Attachments and Metadata

Extract a single attachment's bytes or a metadata record's key/value map.

```bash
# Write attachment bytes to a file (or pipe stdout)
pymcap-cli get attachment --name calib.bin --output calib.bin data.mcap
pymcap-cli get attachment -n calib.bin data.mcap > calib.bin

# Disambiguate when multiple attachments share a name
pymcap-cli get attachment --name notes.txt --offset 1234 -o notes.txt data.mcap

# Print a metadata record as JSON (records sharing a name are merged)
pymcap-cli get metadata --name session data.mcap
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

### `hash` — Stable Recording Fingerprint

Print a compression-independent structural hash. It covers normalized schemas,
channels, counts, and every message log timestamp, while reusing the MCAP
summary and message indexes when available.

```bash
pymcap-cli hash recording.mcap
# mcap-index-v1:7f…
```

Compressed, uncompressed, and unchunked rewrites of the same recording have the
same hash.
Message payload bytes, publish timestamps, sequences, attachment contents, and
metadata bodies are not hashed; use `diff --compare-payloads` when payload
equality must be verified.

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

### `index` — Sidecar Catalog

Maintain a sidecar SQLite catalog of MCAP summaries for fast lookup across
large recording trees. Requires the `xxhash` extra.

```bash
# Scan a tree and skip unchanged files on later runs
pymcap-cli index scan /data/recordings

# Coverage and directory-level rollups
pymcap-cli index status /data/recordings
pymcap-cli index tree /data/recordings --max-depth 3

# Query by topic/schema/time and inspect catalog-wide topics
pymcap-cli index query /data/recordings --topic /camera/front --format json
pymcap-cli index topics /camera --sort-by messages

# Compute one reducer result per matching file (all matches by default)
pymcap-cli index stats /data/recordings \
  --query 'maximum=/pressure/front.value.@@max'

# Machine-readable output keeps each path paired with its reducer values
pymcap-cli index stats /data/recordings \
  --query 'maximum=/pressure/front.value.@@max' \
  --format json

# Apply pending schema migrations to an existing catalog
pymcap-cli index migrate
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

Generate one MP4 per image topic using hardware-accelerated encoding. Requires
the `video` extra.

```bash
# Basic video generation
pymcap-cli video data.mcap --topic /camera/front --output ./videos

# With quality preset
pymcap-cli video data.mcap --topic /camera/rear --output ./videos --quality high

# Use specific codec and encoder
pymcap-cli video data.mcap --topic /lidar/image --output ./videos --codec h265 --encoder videotoolbox
```

### `roscompress` — ROS Image and Point-Cloud Compression

Compress ROS MCAP files by converting CompressedImage/Image topics to
CompressedVideo format and PointCloud2 topics to Cloudini or Draco compressed
point clouds. Requires the `video` and `pointcloud` extras; Draco compression
also requires the `draco` extra.

```bash
# Basic compression
pymcap-cli roscompress data.mcap -o compressed.mcap

# Specify quality and codec
pymcap-cli roscompress data.mcap -o compressed.mcap --quality 28 --codec h265

# Draco point cloud compression using the Foxglove compressed point cloud schema
pymcap-cli roscompress data.mcap -o compressed.mcap --pc-format draco --pc-schema foxglove

# Override inherited point-cloud settings for one topic (repeatable)
pymcap-cli roscompress data.mcap -o compressed.mcap \
  --pointcloud-topic-options '/LIDAR_TOP/points:resolution=0.02,pc-compression=lz4'

# Override inherited video settings; the selector is a topic regex
pymcap-cli roscompress data.mcap -o compressed.mcap \
  --video-topic-options '/CAM_FRONT/image:quality=24,scale=1280' \
  --video-topic-options '/CAM_(BACK|BACK_LEFT|BACK_RIGHT)/image:codec=h265,quality=28'

# Keep selected topics byte-for-byte while compressing the rest
pymcap-cli roscompress data.mcap -o compressed.mcap \
  --pointcloud-topic-options '/LIDAR_LEFT/points:mode=keep' \
  --video-topic-options '/CAM_REAR/image:mode=keep'

# Append raw ffmpeg output arguments globally and for matching topics
pymcap-cli roscompress data.mcap -o compressed.mcap --backend ffmpeg-cli \
  --ffmpeg-args='-preset medium' \
  --video-topic-ffmpeg-args '/CAM_.*/image:-tune film -threads 4'

# Delete the local source only after validating the completed output
pymcap-cli roscompress data.mcap -o compressed.mcap --delete-source

# Recursively compress a directory as independent resumable transactions
uvx 'pymcap-cli[video,pointcloud]' roscompress recordings \
  --batch --output-dir compressed --continue-on-error
```

Batch discovery is recursive and sorted. Each job uses a uniquely named adjacent
partial, lightly validates it for a readable MCAP header, summary, and per-topic
message counts, then atomically renames it. Counts are compared per topic when
the source provides them; `--topic` limits the topics that must be preserved,
while `--exclude-topic` explicitly permits loss on matching topics. A source
without counts skips this comparison, but an output without counts is rejected.
The JSONL archive records the recipe, source size and modification time, and
output size; unchanged work is validated again before reporting
`verified-resumed`. Existing unverified outputs remain collisions unless
`--force` is explicit. Batch mode does not support URLs, a single-file output,
or `--delete-source`.

Topic profiles inherit every unspecified global option. The selector is a topic
regex matched in full and case-insensitively, exactly like `--topic` /
`--exclude-topic`, so a plain topic name selects just that topic. When patterns
overlap, the first profile that matches a topic wins, and any topic no profile
claims uses the global options. A pattern that matches no topic is reported as a
warning rather than silently ignored.

Both profile types accept `mode=keep` by itself to copy matching topics without
decoding, cleanup, sorting, schema/channel changes, or message reserialization.
Those topics are excluded from the catch-all compressor. `mode=default` resets
an earlier profile for the same pattern to the global roscompress settings.
Point-cloud keys: `mode`, `resolution`, `pc-format`, `pc-schema`, `pc-encoding`,
`pc-compression`, `draco-compression-level`. Video keys: `mode`, `quality`,
`codec`, `encoder`, `scale`, `backend`; `encoder=auto` and `scale=original` clear
global values. `pc-encoding=none` disables Cloudini's internal field encoding;
it does not keep the source topic unchanged. FFmpeg arguments use shell-style quoting without invoking a shell.
Per-topic arguments append to `--ffmpeg-args` and require `ffmpeg-cli`; use
`PATTERN:none` to clear them for the matching topics.

### `rosdecompress` — ROS Decompression

Decompress CompressedVideo, CompressedPointCloud2, and Foxglove CompressedPointCloud topics back to standard ROS formats. Requires the `video` and `pointcloud` extras.

```bash
# Decompress to CompressedImage (JPEG)
pymcap-cli rosdecompress input.mcap output.mcap

# Decompress to raw Image
pymcap-cli rosdecompress input.mcap output.mcap --video-format raw

# Skip point cloud decompression
pymcap-cli rosdecompress input.mcap output.mcap --no-pointcloud

# Delete the local source only after validating the completed output
pymcap-cli rosdecompress input.mcap output.mcap --delete-source
```

### `export-images` — Image Files

Export image topics to per-topic folders of image files. `CompressedImage`
payloads keep their original encoding by default (`--format native`); set
`--format` to a Pillow format (e.g. `jpeg`, `png`, `webp`) to re-encode. Raw
`Image` messages always use `--raw-format` (default `png`). Requires the
`image` extra.

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

### `bridge` — Live Foxglove Bridge

Inspect, monitor, stream, or record live topics from a Foxglove WebSocket bridge.
Requires the `bridge` extra.

```bash
# Inspect advertised channels
pymcap-cli bridge localhost:8765

# Validate the live system before recording
pymcap-cli bridge check localhost --spec recording.yaml --duration 5

# Stream decoded messages
pymcap-cli bridge cat localhost:8765 --topic /tf --limit 10

# Monitor one or more topic rates
pymcap-cli bridge hz localhost:8765 -t /imu/data -t '/camera/.*'

# Monitor received payload bandwidth
pymcap-cli bridge bw localhost:8765 -t /camera/image

# Combined rate, payload bandwidth, and corrected message delay
pymcap-cli bridge stats localhost:8765 --all

# Record all advertised topics to MCAP
pymcap-cli bridge record localhost:8765 --all -o live.mcap

# Chronologically merge and play MCAP files into an existing bridge
pymcap-cli bridge play first.mcap second.mcap --target localhost --speed 2

# The bridge target can come from the environment
PYMCAP_BRIDGE=localhost pymcap-cli bridge play recording.mcap -t '/camera/.*'

# Host an MCAP through the minimal Foxglove launcher
pymcap-cli bridge serve recording.mcap

# Browse a directory and open one or more recordings in Foxglove
pymcap-cli bridge serve /data/recordings --port 9090

# Compress images and point clouds just in time while serving (no temporary MCAP)
uvx "pymcap-cli[bridge-codecs]" bridge serve recording.mcap --preset fast

# Publish a compressed recording as standard JPEG images and PointCloud2 messages
pymcap-cli bridge play compressed.mcap --target localhost --transform rosdecompress

# Avoid JIT work for topics without consumers (target must support connectionGraph)
pymcap-cli bridge play recording.mcap --target localhost \
  --transform roscompress --only-subscribed
```

`bridge hz`, `bridge bw`, and `bridge stats` subscribe once and receive selected
messages continuously. They print the heading once, then append one compact,
timestamped row per topic every second so each update remains in terminal
scrollback. Measurements use a 10-second rolling time window by default; use
`--interval` and `--window` to change those independent values. They run until
Ctrl+C unless `--duration` is set. With `--json`, they emit one JSON object per
interval.
`bridge bw` reports Foxglove message payload bytes, excluding WebSocket, TLS, and
other transport framing. `bridge stats` corrects message timestamp age using bridge
time frames; delay is unavailable when no clock-offset samples arrive. Use
`bridge delay` for detailed bridge-clock and decoded ROS `header.stamp`
measurements.

`bridge serve` only transforms messages on channels that currently have subscribers,
and releases per-channel codec state after the last subscriber leaves. For
`bridge play`, `--only-subscribed` uses the target's `connectionGraph` capability to
follow consumers dynamically and pauses playback while no selected topic has one.
`bridge serve` always serves the same minimal recording launcher. Passing explicit
files shows only those files; passing a directory discovers recordings beneath it.
Opening one file or a multi-file selection launches Foxglove Desktop through
`foxglove://open`. Pass `--no-browser` for headless operation. Playback loops by
default; pass `--no-loop` to play each connection once. Its default port is 8766,
avoiding the official Foxglove bridge default on 8765.

The launcher loads indexed start/end time, duration, message count, and channel
count after the file list appears. This reads and caches each MCAP footer/summary
by file size and modification time; it does not scan message data, so the work
scales with the number and summary sizes of files rather than their total payload
size. An unindexed or incomplete file shows `No indexed summary` instead of
triggering a potentially multi-gigabyte recovery scan.

The launcher's links contain the raw WebSocket URL. Clicking one opens Foxglove,
while copying its address produces a connection URL that can be shared or pasted
directly into Foxglove:

Foxglove connects to `/ws` with one repeated `file` parameter per selected
recording, for example:

```text
ws://localhost:9090/ws?file=run-1.mcap&file=run-2.mcap
```

An optional `preset` overrides the server's transform configuration for that
connection only. Without it, the server's CLI configuration remains the default:

```text
# Original messages, even when the server has a compression preset
ws://localhost:9090/ws?file=run.mcap&preset=none

# Full-resolution or reduced-bandwidth JIT compression
wss://recordings.example/ws?file=run.mcap&preset=compress
wss://recordings.example/ws?file=run.mcap&preset=fast
wss://recordings.example/ws?file=run.mcap&preset=low

# Decode roscompress messages for Foxglove
ws://localhost:9090/ws?file=run.mcap&preset=decompress
```

Files selected within one connection are merged chronologically. Every Foxglove
connection receives an independent playhead, controls, and MCAP cache. The playback
clock advances even when no topics are subscribed, while message transforms and
publishing remain subscriber-aware. Closing the Foxglove connection closes its session.
The playback clock remains authoritative at high speeds: messages more than
100 ms late are counted and dropped before further JIT transform work instead
of building a backlog.
The server has no authentication, so expose it remotely only through a
trusted network or an authenticated reverse proxy.

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
  -x '/debug/.*' -x '/test/.*' \
  --metadata exclude --compression zstd

# Extract camera data with time range
pymcap-cli process full_log.mcap -o camera.mcap \
  -t '/camera/.*' \
  -S "2024-01-01T10:00:00Z" -E "2024-01-01T11:00:00Z"

# Recover corrupt file and compress in one pass
pymcap-cli process corrupt.mcap -o recovered.mcap --recovery-mode --compression lz4

# Fast filtering with automatic chunk copying when possible
pymcap-cli process 100gb_file.mcap -o filtered.mcap \
  -t '/lidar/.*' --compression zstd

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
uv sync --all-groups --all-extras --all-packages

# Run locally during development
uv run pymcap-cli --help

# Format and lint code
pre-commit run --all-files

# Run tests
uv run pytest pymcap-cli/tests -m "not benchmark" --no-cov -q
```
