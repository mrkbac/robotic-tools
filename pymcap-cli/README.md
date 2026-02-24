# pymcap-cli

A high-performance Python CLI for MCAP file processing with advanced recovery, filtering, and optimization capabilities.

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

- **Advanced Recovery**: Handles corrupt MCAP files with intelligent chunk-level recovery and MessageIndex validation
- **Smart Chunk Copying**: Fast chunk copying without decompression when possible - up to 10x faster for filtering operations
- **Unified Processing**: Single `process` command combines recovery + filtering + compression in one optimized pass
- **Precise Filtering**: Regex topic filtering, time range filtering, and content type filtering with deferred schema/channel writing
- **Rich Progress Display**: Beautiful progress bars with transfer speeds and time estimates using Rich console output
- **Robust Error Handling**: Graceful degradation with detailed error reporting and recovery statistics

## Quick Start

```bash
# Get file information
pymcap-cli info data.mcap

# Recover a corrupt MCAP file
pymcap-cli recover corrupted.mcap -o fixed.mcap

# Filter messages by topic with compression
pymcap-cli process large.mcap -o filtered.mcap -y "/camera.*" --compression zstd

# Rechunk by topic patterns
pymcap-cli rechunk data.mcap -o rechunked.mcap -p "/camera.*" -p "/lidar.*"

# Show disk usage breakdown
pymcap-cli du large.mcap

# Compress files
pymcap-cli compress input.mcap -o compressed.mcap --compression lz4
```

## Available Commands

### `info` - File Information

Display detailed MCAP file information including schemas, channels, message counts, and time ranges.

```bash
# Single file
pymcap-cli info data.mcap

# Multiple files (displays each file separately)
pymcap-cli info file1.mcap file2.mcap file3.mcap

# JSON output for single file
pymcap-cli info-json data.mcap

# JSON output for multiple files (returns array)
pymcap-cli info-json file1.mcap file2.mcap
```

### `cat` - Stream Messages

Stream MCAP messages to stdout. Outputs as Rich tables when interactive, JSONL when piped.

```bash
# Display messages in a table (interactive)
pymcap-cli cat recording.mcap

# Pipe to file as JSONL
pymcap-cli cat recording.mcap > messages.jsonl

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
```

### `recover` - Advanced Recovery

Recover data from potentially corrupt MCAP files with intelligent error handling.

```bash
# Basic recovery
pymcap-cli recover corrupt.mcap -o fixed.mcap

# Force chunk decoding for maximum recovery
pymcap-cli recover corrupt.mcap -o fixed.mcap --always-decode-chunk
```

### `recover-inplace` - In-Place Recovery

Rebuild an MCAP file's summary and footer in place without creating a new file.

```bash
# Rebuild summary/footer in place
pymcap-cli recover-inplace data.mcap

# With exact size calculation
pymcap-cli recover-inplace data.mcap --exact-sizes

# Skip confirmation prompt
pymcap-cli recover-inplace data.mcap --force
```

### `process` - Unified Processing

The most powerful command - combines recovery, filtering, and optimization in a single pass.

```bash
# Filter by topic regex
pymcap-cli process data.mcap -o filtered.mcap -y "/camera.*" -y "/lidar.*"

# Time range filtering (nanoseconds or RFC3339)
pymcap-cli process data.mcap -o subset.mcap -S 1640995200000000000 -E 1640995260000000000
pymcap-cli process data.mcap -o subset.mcap -S "2022-01-01T00:00:00Z" -E "2022-01-01T01:00:00Z"

# Exclude topics and metadata
pymcap-cli process data.mcap -o clean.mcap -n "/debug.*" --exclude-metadata

# Change compression with filtering
pymcap-cli process zstd.mcap -o lz4.mcap --compression lz4 -y "/important.*"

# Recovery mode with filtering (handles corrupt files)
pymcap-cli process corrupt.mcap -o recovered.mcap -y "/camera.*" --recovery-mode
```

### `merge` - Merge Files

Merge multiple MCAP files chronologically into a single output file.

```bash
# Merge two files
pymcap-cli merge recording1.mcap recording2.mcap -o combined.mcap

# Merge with compression
pymcap-cli merge *.mcap -o all_recordings.mcap --compression lz4

# Exclude metadata/attachments
pymcap-cli merge file1.mcap file2.mcap -o merged.mcap --metadata exclude
```

### `convert` - Convert DB3 to MCAP

Convert ROS2 DB3 (SQLite) bag files to MCAP format.

```bash
# Basic conversion
pymcap-cli convert input.db3 -o output.mcap

# Specify ROS distro
pymcap-cli convert input.db3 -o output.mcap --distro jazzy

# With custom message definitions
pymcap-cli convert input.db3 -o output.mcap --extra-path /path/to/msgs

# Multiple custom paths
pymcap-cli convert input.db3 -o output.mcap \
    --extra-path /path/to/msgs1 \
    --extra-path /path/to/msgs2
```

### `rechunk` - Topic-Based Rechunking

Reorganize MCAP messages into separate chunk groups based on topic patterns. This is useful for optimizing file layout when different topics are accessed independently, improving playback performance for topic-specific queries.

```bash
# Group camera and lidar topics into separate chunks
pymcap-cli rechunk data.mcap -o rechunked.mcap -p "/camera.*" -p "/lidar.*"

# Multiple patterns - each gets its own chunk group
pymcap-cli rechunk data.mcap -o rechunked.mcap \
  -p "/camera/front.*" \
  -p "/camera/rear.*" \
  -p "/lidar.*" \
  -p "/radar.*"

# With custom chunk size and compression
pymcap-cli rechunk data.mcap -o rechunked.mcap \
  -p "/high_freq.*" \
  --chunk-size 8388608 \
  --compression lz4
```

**How it works:**
- Messages are grouped by the first matching pattern (first match wins)
- Each pattern gets its own chunk group that can span multiple chunks
- Topics not matching any pattern go into a separate "unmatched" group
- Messages within each group preserve their original order
- Useful for optimizing access patterns when different topics are read independently

### `filter` - Topic Filtering

Filter messages by topic patterns (simpler version of `process`).

```bash
# Include specific topics
pymcap-cli filter data.mcap -o filtered.mcap --include-topics "/camera/image" "/lidar/points"

# Exclude topics
pymcap-cli filter data.mcap -o filtered.mcap --exclude-topics "/debug.*" "/test.*"
```

### `compress` - Compression Tool

Change MCAP file compression.

```bash
# Compress with different algorithms
pymcap-cli compress input.mcap -o output.mcap --compression zstd
pymcap-cli compress input.mcap -o output.mcap --compression lz4
```

### `du` - Disk Usage Analysis

Analyze MCAP file size breakdown by chunks, schemas, channels, and message counts.

```bash
pymcap-cli du large.mcap
```

### `list` - List Records

List various record types in an MCAP file with detailed information.

```bash
# List channels
pymcap-cli list channels data.mcap

# List chunks
pymcap-cli list chunks data.mcap

# List schemas
pymcap-cli list schemas data.mcap

# List attachments
pymcap-cli list attachments data.mcap

# List metadata
pymcap-cli list metadata data.mcap
```

### `info-json` - JSON Statistics

Output comprehensive MCAP file statistics as JSON, including message distribution, channel rates, and compression stats.

```bash
# Basic JSON output
pymcap-cli info-json data.mcap

# Rebuild from scratch with exact sizes
pymcap-cli info-json data.mcap --rebuild --exact-sizes
```

### `tftree` - TF Transform Tree

Display ROS TF transform tree from MCAP files with visual hierarchy.

```bash
# Show complete TF tree (both /tf and /tf_static)
pymcap-cli tftree data.mcap

# Show only static transforms
pymcap-cli tftree data.mcap --static-only
```

### `video` - Video Generation

Generate MP4 videos from image topics (CompressedImage or Image) using hardware-accelerated encoding. Requires the `video` extra.

```bash
# Basic video generation
pymcap-cli video data.mcap --topic /camera/front --output front.mp4

# With quality preset
pymcap-cli video data.mcap --topic /camera/rear --output rear.mp4 --quality high

# Use specific codec and encoder
pymcap-cli video data.mcap --topic /lidar/image --output lidar.mp4 --codec h265 --encoder videotoolbox

# Manual CRF quality control
pymcap-cli video data.mcap --topic /camera/debug --output debug.mp4 --crf 18
```

### `roscompress` - ROS Image Compression

Compress ROS MCAP files by converting CompressedImage/Image topics to CompressedVideo format. Requires the `video` extra.

```bash
# Basic compression
pymcap-cli roscompress data.mcap -o compressed.mcap

# Specify quality (CRF: lower = better, 0-51)
pymcap-cli roscompress data.mcap -o compressed.mcap --quality 28

# Specify codec
pymcap-cli roscompress data.mcap -o compressed.mcap --codec h265

# Force specific encoder
pymcap-cli roscompress data.mcap -o compressed.mcap --encoder libx264
```

### Shell Autocompletion

pymcap-cli supports automatic shell completion for bash, zsh, fish, and PowerShell using Typer's built-in completion system.

#### Quick Install (recommended)
```bash
# Automatically install completion for your current shell
pymcap-cli --install-completion

# That's it! Restart your shell or source your config file
```

#### Manual Install
```bash
# Bash - add to ~/.bashrc
eval "$(pymcap-cli --show-completion bash)"

# Zsh - add to ~/.zshrc
eval "$(pymcap-cli --show-completion zsh)"

# Fish - add to ~/.config/fish/config.fish
pymcap-cli --show-completion fish | source

# PowerShell - add to your profile
Invoke-Expression (& pymcap-cli --show-completion powershell)
```

## Advanced Usage

### Performance Optimization

```bash
# Fast chunk copying (default) - up to 10x faster for large files
pymcap-cli process data.mcap -o filtered.mcap -y "/camera.*" --chunk-copying

# Disable chunk copying for maximum compatibility
pymcap-cli process data.mcap -o filtered.mcap -y "/camera.*" --no-chunk-copying

# Always decode chunks (slower but handles edge cases)
pymcap-cli process data.mcap -o filtered.mcap -y "/camera.*" --always-decode-chunk
```

### Recovery Modes

```bash
# Graceful error handling (default)
pymcap-cli process corrupt.mcap -o recovered.mcap --recovery-mode

# Strict mode - fail on any errors
pymcap-cli process data.mcap -o output.mcap --no-recovery
```

### Time Filtering Options

```bash
# Using seconds (automatically converted to nanoseconds)
pymcap-cli process data.mcap -o subset.mcap --start-secs 1640995200 --end-secs 1640995260

# Using nanoseconds directly
pymcap-cli process data.mcap -o subset.mcap --start-nsecs 1640995200000000000

# Using RFC3339 timestamps (most readable)
pymcap-cli process data.mcap -o subset.mcap -S "2022-01-01T00:00:00Z" -E "2022-01-01T01:00:00Z"
```

## Common Use Cases

### Clean Up Debug Data

```bash
# Remove debug topics and metadata to reduce file size
pymcap-cli process raw_data.mcap -o clean_data.mcap \
  -n "/debug.*" -n "/test.*" --exclude-metadata --compression zstd
```

### Extract Camera Data

```bash
# Extract only camera topics with time filtering
pymcap-cli process full_log.mcap -o camera_only.mcap \
  -y "/camera.*" -S "2024-01-01T10:00:00Z" -E "2024-01-01T11:00:00Z"
```

### Recover and Compress

```bash
# Recover corrupt file and compress in one pass
pymcap-cli process corrupt.mcap -o recovered.mcap \
  --recovery-mode --compression lz4
```

### High-Performance Filtering

```bash
# Fast filtering with chunk copying for maximum performance
pymcap-cli process 100gb_file.mcap -o filtered.mcap \
  -y "/lidar.*" --chunk-copying --compression zstd
```

### Optimize for Topic-Specific Playback

```bash
# Rechunk by sensor type for faster topic-specific access
pymcap-cli rechunk robot_log.mcap -o optimized.mcap \
  -p "/camera.*" \
  -p "/lidar.*" \
  -p "/imu.*" \
  -p "/gps.*"
```

## Technical Details

- **Smart Chunk Processing**: Automatically chooses between fast chunk copying and individual record processing based on filter criteria
- **MessageIndex Validation**: Validates and rebuilds MessageIndexes when necessary for data integrity
- **Deferred Schema Writing**: Only writes schemas and channels that are actually used by included messages
- **Compression Support**: zstd, lz4, and uncompressed formats with configurable chunk sizes
- **Memory Efficient**: Streams processing with configurable buffer sizes for handling large files
- **Error Recovery**: Multiple fallback strategies for handling corrupt or incomplete MCAP files

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
