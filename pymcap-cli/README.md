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

# Show disk usage breakdown
uv run pymcap_cli du large.mcap

# Compress/decompress files
uv run pymcap_cli compress input.mcap -o compressed.mcap --compression lz4
uv run pymcap_cli decompress compressed.mcap -o uncompressed.mcap
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

### `filter` - Topic Filtering

Filter messages by topic patterns (simpler version of `process`).

```bash
# Include specific topics
uv run pymcap_cli filter data.mcap -o filtered.mcap --include-topics "/camera/image" "/lidar/points"

# Exclude topics
uv run pymcap_cli filter data.mcap -o filtered.mcap --exclude-topics "/debug.*" "/test.*"
```

### `compress` / `decompress` - Compression Tools

Change MCAP file compression.

```bash
# Compress with different algorithms
uv run pymcap_cli compress input.mcap -o output.mcap --compression zstd
uv run pymcap_cli compress input.mcap -o output.mcap --compression lz4

# Decompress to uncompressed format
uv run pymcap_cli decompress compressed.mcap -o uncompressed.mcap
```

### `du` - Disk Usage Analysis

Analyze MCAP file size breakdown by chunks, schemas, channels, and message counts.

```bash
uv run pymcap_cli du large.mcap
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
