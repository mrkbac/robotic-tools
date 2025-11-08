# 🚀 pymcap-cli

**A high-performance Python CLI for MCAP file processing with advanced recovery, filtering, and optimization capabilities.**

## Why pymcap-cli over the official Go CLI?

- **🔧 Advanced Recovery**: Handles corrupt MCAP files with intelligent chunk-level recovery and MessageIndex validation
- **⚡ Smart Chunk Copying**: Fast chunk copying without decompression when possible - up to 10x faster for filtering operations
- **🔄 Unified Processing**: Single `process` command combines recovery + filtering + compression in one optimized pass
- **🎯 Precise Filtering**: Regex topic filtering, time range filtering, and content type filtering with deferred schema/channel writing
- **📊 Rich Progress Display**: Beautiful progress bars with transfer speeds and time estimates using Rich console output
- **🛡️ Robust Error Handling**: Graceful degradation with detailed error reporting and recovery statistics

## ⚡ Quick Start

```bash
# Get file information
uv run pymcap_cli info data.mcap

# Recover a corrupt MCAP file
uv run pymcap_cli recover corrupted.mcap -o fixed.mcap

# Filter messages by topic with compression
uv run pymcap_cli process large.mcap -o filtered.mcap -y "/camera.*" --compression zstd

# Rechunk by topic patterns
uv run pymcap_cli rechunk data.mcap -o rechunked.mcap -p "/camera.*" -p "/lidar.*"

# Show disk usage breakdown
uv run pymcap_cli du large.mcap

# Compress files
uv run pymcap_cli compress input.mcap -o compressed.mcap --compression lz4
```

## 📋 Available Commands

### `info` - File Information

Display detailed MCAP file information including schemas, channels, message counts, and time ranges.

```bash
uv run pymcap_cli info data.mcap
```

### `recover` - Advanced Recovery

Recover data from potentially corrupt MCAP files with intelligent error handling.

```bash
# Basic recovery
uv run pymcap_cli recover corrupt.mcap -o fixed.mcap

# Force chunk decoding for maximum recovery
uv run pymcap_cli recover corrupt.mcap -o fixed.mcap --always-decode-chunk
```

### `process` - Unified Processing

The most powerful command - combines recovery, filtering, and optimization in a single pass.

```bash
# Filter by topic regex
uv run pymcap_cli process data.mcap -o filtered.mcap -y "/camera.*" -y "/lidar.*"

# Time range filtering (nanoseconds or RFC3339)
uv run pymcap_cli process data.mcap -o subset.mcap -S 1640995200000000000 -E 1640995260000000000
uv run pymcap_cli process data.mcap -o subset.mcap -S "2022-01-01T00:00:00Z" -E "2022-01-01T01:00:00Z"

# Exclude topics and metadata
uv run pymcap_cli process data.mcap -o clean.mcap -n "/debug.*" --exclude-metadata

# Change compression with filtering
uv run pymcap_cli process zstd.mcap -o lz4.mcap --compression lz4 -y "/important.*"

# Recovery mode with filtering (handles corrupt files)
uv run pymcap_cli process corrupt.mcap -o recovered.mcap -y "/camera.*" --recovery-mode
```

### `rechunk` - Topic-Based Rechunking

Reorganize MCAP messages into separate chunk groups based on topic patterns. This is useful for optimizing file layout when different topics are accessed independently, improving playback performance for topic-specific queries.

```bash
# Group camera and lidar topics into separate chunks
uv run pymcap_cli rechunk data.mcap -o rechunked.mcap -p "/camera.*" -p "/lidar.*"

# Multiple patterns - each gets its own chunk group
uv run pymcap_cli rechunk data.mcap -o rechunked.mcap \
  -p "/camera/front.*" \
  -p "/camera/rear.*" \
  -p "/lidar.*" \
  -p "/radar.*"

# With custom chunk size and compression
uv run pymcap_cli rechunk data.mcap -o rechunked.mcap \
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
uv run pymcap_cli filter data.mcap -o filtered.mcap --include-topics "/camera/image" "/lidar/points"

# Exclude topics
uv run pymcap_cli filter data.mcap -o filtered.mcap --exclude-topics "/debug.*" "/test.*"
```

### `compress` - Compression Tool

Change MCAP file compression.

```bash
# Compress with different algorithms
uv run pymcap_cli compress input.mcap -o output.mcap --compression zstd
uv run pymcap_cli compress input.mcap -o output.mcap --compression lz4
```

### `du` - Disk Usage Analysis

Analyze MCAP file size breakdown by chunks, schemas, channels, and message counts.

```bash
uv run pymcap_cli du large.mcap
```

### `list` - List Records

List various record types in an MCAP file with detailed information.

```bash
# List channels
uv run pymcap_cli list channels data.mcap

# List chunks
uv run pymcap_cli list chunks data.mcap

# List schemas
uv run pymcap_cli list schemas data.mcap

# List attachments
uv run pymcap_cli list attachments data.mcap

# List metadata
uv run pymcap_cli list metadata data.mcap
```

### `info-json` - JSON Statistics

Output comprehensive MCAP file statistics as JSON, including message distribution, channel rates, and compression stats.

```bash
# Basic JSON output
uv run pymcap_cli info-json data.mcap

# Rebuild from scratch with exact sizes
uv run pymcap_cli info-json data.mcap --rebuild --exact-sizes
```

### `tftree` - TF Transform Tree

Display ROS TF transform tree from MCAP files with visual hierarchy.

```bash
# Show complete TF tree (both /tf and /tf_static)
uv run pymcap_cli tftree data.mcap

# Show only static transforms
uv run pymcap_cli tftree data.mcap --static-only
```

### `video` - Video Generation

Generate MP4 videos from image topics (CompressedImage or Image) using hardware-accelerated encoding.

```bash
# Basic video generation
uv run pymcap_cli video data.mcap --topic /camera/front --output front.mp4

# With quality preset
uv run pymcap_cli video data.mcap --topic /camera/rear --output rear.mp4 --quality high

# Use specific codec and encoder
uv run pymcap_cli video data.mcap --topic /lidar/image --output lidar.mp4 --codec h265 --encoder videotoolbox

# Manual CRF quality control
uv run pymcap_cli video data.mcap --topic /camera/debug --output debug.mp4 --crf 18
```

### `completion` - Shell Autocompletion

Generate shell autocompletion scripts for bash, zsh, or tcsh.

```bash
# Generate bash completion
uv run pymcap_cli completion bash > ~/.local/share/bash-completion/completions/pymcap_cli

# Generate zsh completion
uv run pymcap_cli completion zsh > ~/.zfunc/_pymcap_cli
```

## 🚀 Advanced Usage

### Performance Optimization

```bash
# Fast chunk copying (default) - up to 10x faster for large files
uv run pymcap_cli process data.mcap -o filtered.mcap -y "/camera.*" --chunk-copying

# Disable chunk copying for maximum compatibility
uv run pymcap_cli process data.mcap -o filtered.mcap -y "/camera.*" --no-chunk-copying

# Always decode chunks (slower but handles edge cases)
uv run pymcap_cli process data.mcap -o filtered.mcap -y "/camera.*" --always-decode-chunk
```

### Recovery Modes

```bash
# Graceful error handling (default)
uv run pymcap_cli process corrupt.mcap -o recovered.mcap --recovery-mode

# Strict mode - fail on any errors
uv run pymcap_cli process data.mcap -o output.mcap --no-recovery
```

### Time Filtering Options

```bash
# Using seconds (automatically converted to nanoseconds)
uv run pymcap_cli process data.mcap -o subset.mcap --start-secs 1640995200 --end-secs 1640995260

# Using nanoseconds directly
uv run pymcap_cli process data.mcap -o subset.mcap --start-nsecs 1640995200000000000

# Using RFC3339 timestamps (most readable)
uv run pymcap_cli process data.mcap -o subset.mcap -S "2022-01-01T00:00:00Z" -E "2022-01-01T01:00:00Z"
```

## 💡 Common Use Cases

### Clean Up Debug Data

```bash
# Remove debug topics and metadata to reduce file size
uv run pymcap_cli process raw_data.mcap -o clean_data.mcap \
  -n "/debug.*" -n "/test.*" --exclude-metadata --compression zstd
```

### Extract Camera Data

```bash
# Extract only camera topics with time filtering
uv run pymcap_cli process full_log.mcap -o camera_only.mcap \
  -y "/camera.*" -S "2024-01-01T10:00:00Z" -E "2024-01-01T11:00:00Z"
```

### Recover and Compress

```bash
# Recover corrupt file and compress in one pass
uv run pymcap_cli process corrupt.mcap -o recovered.mcap \
  --recovery-mode --compression lz4
```

### High-Performance Filtering

```bash
# Fast filtering with chunk copying for maximum performance
uv run pymcap_cli process 100gb_file.mcap -o filtered.mcap \
  -y "/lidar.*" --chunk-copying --compression zstd
```

### Optimize for Topic-Specific Playback

```bash
# Rechunk by sensor type for faster topic-specific access
uv run pymcap_cli rechunk robot_log.mcap -o optimized.mcap \
  -p "/camera.*" \
  -p "/lidar.*" \
  -p "/imu.*" \
  -p "/gps.*"
```

## 🔧 Technical Details

- **Smart Chunk Processing**: Automatically chooses between fast chunk copying and individual record processing based on filter criteria
- **MessageIndex Validation**: Validates and rebuilds MessageIndexes when necessary for data integrity
- **Deferred Schema Writing**: Only writes schemas and channels that are actually used by included messages
- **Compression Support**: zstd, lz4, and uncompressed formats with configurable chunk sizes
- **Memory Efficient**: Streams processing with configurable buffer sizes for handling large files
- **Error Recovery**: Multiple fallback strategies for handling corrupt or incomplete MCAP files

## 🛠️ Development

```bash
# Setup development environment
uv sync

# Run with development dependencies
uv run pymcap_cli --help

# Format and lint code
uv run pre-commit run --all-files

# Run tests (if available)
uv run pytest
```
