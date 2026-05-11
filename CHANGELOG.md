# Changelog

User-facing notes for releases.

---

## Unreleased

### pymcap-cli

- New `doctor` command validates MCAP container structure and indexes.
  Findings are grouped by code by default, `--show-all` prints every
  finding, and `--strict-message-order` can promote non-monotonic message
  timestamps to errors.
- New `bridge` command group for live Foxglove WebSocket bridges,
  available through the `bridge` extra. The default command inspects
  server metadata, channels, services, status messages, and connection
  graph data, with table or JSON output, optional compressed JSON,
  sorting, and `--watch`.
- New `bridge record` captures live bridge topics into an MCAP file.
  Topic selection mirrors `ros2 bag record`: exact topic lists, `--all`,
  regex includes, exact / regex excludes, `--duration`,
  `--message-limit`, compression/chunk options, and live progress.
- New `bridge cat` streams decoded live bridge messages to Rich TTY
  panels or JSONL. It shares the `cat` renderer and supports topic
  filters, MessagePath `--query`, `--grep`, case-insensitive grep,
  limits, duration, and byte-rendering modes.
- New `get attachment` and `get metadata` commands extract
  summary-indexed attachment bytes and metadata JSON by name. Attachments
  can be disambiguated with `--offset`; metadata records with the same
  name are merged with later values winning.
- New `tf-export` command reconstructs `urdf`, `sdf`, or `json` robot
  descriptions from `/tf_static` (and optionally `/tf` at a snapshot
  timestamp). Validates tree shape (rejects cycles, multi-parent frames
  without `--allow-multi-parent`, and multi-root trees without
  `--root`).
- `cat` adds `--grep` / `--grep-ignore-case` for decoded scalar-value
  searches. Byte arrays are skipped during grep so image and point-cloud
  payloads do not dominate the search cost.
- `cat` TTY output now renders decoded messages as a tree that annotates
  ROS constants and Foxglove enum wrapper fields with enum names while
  leaving JSON output stable.
- `filter` adds shell-style topic globs (`--topic-glob`,
  `--exclude-topic-glob`), topic inversion (`--invert-topics`),
  time-window inversion (`--invert-time`), and relative start/end anchors
  such as `@5s`, `start+5s`, and `end-30s`.
- `filter` and `split` add `--latch` and `--latch-from-metadata` so
  transient-local topics such as `/tf_static` are preserved through
  trimming and replayed into split outputs.
- `bag2mcap` and `convert` now exit successfully without creating an
  output file when the ROS 1 bag or ROS 2 DB3 input contains no
  connections/topics.
- Invalid `--compression` values are rejected at CLI parse time across
  transform commands before any input paths are opened.
- Progress bars no longer overshoot when a command rereads stream data
  after scanning summaries or indexes.
- Merging inputs that require schema or channel ID remapping now decodes
  every chunk from the affected input stream, preventing stale in-chunk
  Schema/Channel records from leaking into the output.
- Optional commands now report targeted install hints for missing extras;
  `pymcap-cli[all]` includes the new `bridge` extra.

### small-mcap

- zstd compression/decompression can use Python 3.14's
  `compression.zstd` module, with the existing third-party `zstandard`
  backend retained for older interpreters. Missing-backend errors now
  name both accepted zstd providers.
- `Remapper` tracks streams where schema or channel IDs were reassigned,
  including deduplication to a different ID, so processors can avoid
  unsafe raw chunk copies after remapping.

### robo-ws-bridge

- `WebSocketBridgeClient` now tracks advertised services, remove-status
  events, and connection-graph updates. It exposes `services` and
  `connection_graph`, supports connection-graph subscribe/unsubscribe
  callbacks, and restores the graph subscription after reconnects.

### pureini

- Point-cloud zstd encode/decode paths support Python 3.14's stdlib
  `compression.zstd` module and keep the existing `zstandard` fallback
  for older Python versions.

### mcap-codec-support

- The point-cloud package exposes `is_compressed_codec_available()` so
  callers can detect whether a compressed point-cloud backend is
  importable before selecting that path.

---

## pymcap-cli 0.9.0, small-mcap 0.7.0

Headline: a new `duplicates` command for finding redundant MCAP files
across a directory, and `--delete-source` on every transform command for
in-place pipelines. Plus a logging cleanup so commands are quiet by default
and chatty under `-v`.

### pymcap-cli 0.9.0

- New `duplicates` command — content-based detection of duplicate MCAPs
  across a directory of files. Works off message-index metadata (no full
  decode) and shares the comparison engine with `diff`, so the two stay
  consistent.
- New `--delete-source` flag on `compress`, `filter`, `merge`, `process`,
  `rechunk`, `recover`, and `split`. The input MCAP is removed only after
  the output is written successfully — handy for in-place pipelines
  without a wrapper script.
- Global `--verbose / -v` and `--quiet`. All commands now route
  diagnostics through `logging`; results stay on stdout, so piping is
  unchanged.
- Filtered exports (`--topic`, schema-aware exporters) report accurate
  progress totals instead of overshooting — `get_total_message_count`
  now respects the include predicate.

### small-mcap 0.7.0

- `read_info_approximate` is much faster on indexed files: each chunk's
  entire `MessageIndex` group is read in one `stream.read()` and parsed
  from a `BytesIO` slice instead of seek+read per record. On a 52 GB /
  17 045-chunk file the cold-cache cost drops from ~2.3 s to ~250 ms
  (≈9×). `pymcap-cli`'s `info` / `du` summaries inherit the speedup.

---

## pymcap-cli 0.8.0, small-mcap 0.6.0, mcap-codec-support 0.2.0 (new), mcap-ros2-support-fast 0.4.0, digitalis 0.9.0, pointcloud2 0.4.0, robo-ws-bridge 0.3.0, websocket-proxy 0.6.0

Headline: `pymcap-cli` learns a real exporter framework with a stack of
new output formats; bulk MCAP processing matches or beats Go's `mcap` CLI
on every workload tested; codec helpers move into a new
`mcap-codec-support` package so non-CLI consumers can reuse them.

### pymcap-cli 0.8.0

New exporters and commands:

- `export-parquet` — one Parquet file per topic plus a
  `_topics.parquet` index. Arrow types follow the ROS 2 schema (uint8
  stays uint8, float32 stays float32 — no silent BIGINT/DOUBLE
  promotion). PointCloud2 payloads come out as `LIST<STRUCT<x,y,z,…>>`
  with original dtypes preserved; `CompressedPointCloud2` is
  decompressed transparently. `_log_time`, `_publish_time` and any
  `builtin_interfaces/Time` field land as `TIMESTAMP_NS` so SQL date
  arithmetic works without manual ns math. Image, video and audio blobs
  are skipped by default; pass `--include-blobs` to keep them. On a
  6.3 GB MCAP, end-to-end throughput is roughly 700 MB/s cold cache.
- `export-csv`, `export-json` — flat per-topic dumps.
- `export-geo` — geojson, kml, or gpx from NavSatFix-style topics.
- `export-images` — extract image messages (compressed or raw) to disk.
- `export-pcd` — point-cloud topics to per-message PCD files.

`plot`, `video`, `roscompress` and `rosdecompress` are now driven by the
same framework, so threading and chunk handling are consistent across
all of them.

Other features:

- `split --expression / -E <message-path>` opens a new segment every
  time the evaluated value of a `ros-parser` message path changes on
  the target topic. Other topics stick with the current segment. Chunks
  with no target-topic messages fast-copy without decoding.
- `split` gains explicit overwrite policies.
- `roscompress` adds a JPEG path for compressing image topics.

Performance:

- `cat` defaults to `--bytes smart`: payloads ≤64 bytes still print as
  int lists, anything larger collapses to `<N bytes>`. On a 1.1 GB file
  with PointCloud2 messages, `cat` drops from ~31 s to ~0.47 s
  (~65× faster). `ints`, `base64` and `skip` modes are still available.
- Filter / compress / decompress / merge match or beat the Go `mcap` CLI
  on every bulk workload tested — up to ~7× on fast-copy zstd→zstd and
  ~3× on compress. Chunks are stream-copied without per-chunk
  allocation, and decompression of upcoming chunks runs on a worker
  pool while the main thread writes the current one. A new `RECOMPRESS`
  path handles compression-only mismatches by decompressing and
  recompressing the chunk payload without decoding messages.
- `du` and `info` use `read_info_approximate` for much faster summary
  reads on large files where exact message counts aren't needed.

### small-mcap 0.6.0

- New `read_info_approximate(...)` for fast summaries when you don't
  need exact message counts. Used by `pymcap-cli du` and friends.
- Stream-copy chunk path: `_CRCWriter.copy_from` and
  `McapWriterRaw.add_chunk_raw` pipe raw chunk bytes input → output
  through a reusable buffer with one CRC pass — backs the Go-mcap-class
  bulk speeds in `pymcap-cli`.
- `stream_reader` honors `lazy_chunks=True` (previously a no-op), so
  bulk metadata scans skip over chunk bodies.

### mcap-codec-support 0.2.0 (new package)

Extracted from `pymcap-cli`. Contains the video and point-cloud encoder
/ decoder factories, schemas, and compression backends. Lets non-CLI
consumers like `websocket-proxy` reuse them without pulling in CLI
dependencies. Existing `pymcap-cli[video]` / `[pointcloud]` extras
transparently resolve through the new package; the old point-cloud
transformer module path keeps a compat shim.

### mcap-ros2-support-fast 0.4.0

- Encoder accepts numpy `ndarray`s directly for primitive arrays;
  encode paths for `uint8` / `int32` / `float64` arrays are tightened.
- Wrong-sized fixed arrays raise `ValueError` immediately instead of
  silently truncating or producing a `struct.error` deep in
  `struct.pack`.

### digitalis 0.9.0, pointcloud2 0.4.0, robo-ws-bridge 0.3.0, websocket-proxy 0.6.0

Internal-only release alongside the wave. `websocket-proxy` picks up
the codec-support extraction; the others are tag bumps with no
user-visible API changes.

---

## pymcap-cli 0.7.0, small-mcap 0.5.0, websocket-proxy 0.5.0, mcap-ros2-support-fast 0.3.0

### pymcap-cli 0.7.0

- New `diff` command — compares two MCAP files using their message
  indexes. Fast and metadata-driven, no full decode required.
- RIHS01 schema hashing is implemented end-to-end, so schema identity
  matches the ROS 2 standard.
- `info` summaries print precise start and end times instead of rounded
  output.

The other three packages in this wave are internal-only bumps.

---

## pymcap-cli 0.6.0

- The `processor` pipeline preserves the source MCAP's profile in its
  output. Previously the profile could be reset on rewrite, which
  confused downstream tools that key off it.

---

## pymcap-cli 0.5.0, small-mcap 0.4.0, ros-parser 0.4.0, pureini 0.3.0, websocket-proxy 0.4.0

Headline: ROS 1 `.bag` files become first-class inputs via the new
`bag2mcap` command, and `roscompress` video encoding collapses to a
single ffmpeg process.

### pymcap-cli 0.5.0

- New `bag2mcap` command — pure-Python ROS 1 bag v2.0 reader and
  converter to MCAP with the `ros1` profile. Supports `none`, `bz2`,
  and `lz4` compressed bags.
- `roscompress` ImagePipeEncoder: encodes JPEG/PNG straight to
  H.264/H.265 in a single ffmpeg `image2pipe` process, replacing the
  previous per-frame `ffprobe` + `ffmpeg` decode subprocesses. Frame
  counts are now exact: pending message metadata is buffered and paired
  with encoder output rather than estimated.
- Docs additions for `diag`, `plot`, `rosdecompress`, plus updated `cat`
  examples.

The other four packages in this wave carry the supporting pieces for
the above (decoder factories used by `rosdecompress`, etc.); no
independently user-visible features.

---

## pymcap-cli 0.4.0, digitalis 0.8.0, mcap-ros2-support-fast 0.2.0, pureini 0.2.0, robo-ws-bridge 0.2.0, ros-parser 0.3.0, websocket-proxy 0.3.0

Headline: a slate of new inspection commands — `diag` for ROS 2
diagnostics, `plot` for time-series, `rosdecompress` to undo
`roscompress` — and dynamic ROS 2 message resolution via `rosdistro`.

### pymcap-cli 0.4.0

- New `diag` command — reads `diagnostic_msgs/DiagnosticArray` and gives
  a scannable view of system health. Defaults to showing only WARN /
  ERROR / STALE components. Modes: summary table (with Hz and a colored
  sparkline timeline per component), `--tree`, `--inspect[-all]`, and
  `--json` (with `frequency_hz` and `level_durations_s`). Filterable by
  level, name, and hardware ID. Defaults scan both `/diagnostics` and
  `/diagnostics_agg`.
- New `plot` command — extracts time-series via `ros-parser` message
  paths and renders interactive Plotly charts. Multiple overlaid
  series, text-to-number mapping for string fields, named labels
  (`Label=/topic.field`), LTTB downsampling (`-d N`), byte-level
  progress bar, `--xy` mode for trajectory plots, Scattergl for WebGL
  acceleration. Plotly is an optional dependency (`plot` extra).
- New `rosdecompress` command — reverses `roscompress`: converts
  `CompressedVideo` back to `CompressedImage` / `Image` and
  `CompressedPointCloud2` back to `PointCloud2`. Supports both PyAV and
  ffmpeg-CLI backends.
- `cat` gains `-o/--output` (with progress bar), `--bytes
ints/base64/skip` for binary-field serialization, auto-truncation of
  bytes in TTY display, and a message-count summary on file write.
- New `all` extra combining `video`, `pointcloud`, and `plot`.

### ros-parser 0.3.0

- Dynamic message-package resolution via `rosdistro`: replaces the
  previous hardcoded list of GitHub repos with on-demand lookup of
  `distribution.yaml`, so any ROS 2 package can resolve.
- Fix: `MathModifier` validation for `@rpy` and `@quat` now returns
  synthetic `MessageDefinition`s so subsequent field access (`.yaw`,
  `.roll`, …) validates correctly.

The other five packages in this wave are internal-only bumps.

---

## pymcap-cli 0.3.0, digitalis 0.7.0, ros-parser 0.2.0, small-mcap 0.3.0

### digitalis 0.7.0

- New live Plot panel for the inspector — Braille-character line chart
  for live numeric fields, with a numpy-backed circular `TimeSeriesBuffer`,
  multi-series, zoom, and auto-fit.
- ScanLevelIndicator and channels-table updates for cleaner scan-depth
  feedback and richer aggregation.

### small-mcap 0.3.0

- Non-seekable stream handling and parallel decompression in benchmark
  paths.

### pymcap-cli 0.3.0, ros-parser 0.2.0

- Tag bumps to align with the inspector and small-mcap changes; no
  user-visible surface changes.
- Interactive VHS scripts and GIFs added for several `pymcap-cli`
  commands, used in the docs.

---

## pymcap-cli 0.2.0, small-mcap 0.2.0, websocket-proxy 0.2.0

### small-mcap 0.2.0

- `pread`-based chunk processing in the reader. The chunk reader does
  one `pread(...)` per chunk rather than `seek` + `read`, removing a
  class of races and making parallel decompression safe.

### pymcap-cli 0.2.0

- Persistent thread-local codec contexts for image decoding — replaces
  per-frame `av.open(BytesIO(...))` with reusable contexts for JPEG and
  PNG, with a container-based fallback for unrecognised formats. Most
  of the per-frame overhead in image-heavy decode paths goes away.

### websocket-proxy 0.2.0

- `pymcap-cli` is now an explicit dependency. `VideoEncoder`,
  `EncoderConfig`, `detect_encoder`, `calculate_downscale_dimensions`,
  `build_encoder_options`, `PointCloudCompressor`, and the schema
  constants all live in `pymcap_cli.image_utils` and are imported from
  there instead of duplicated. The voxel-based point-cloud transformer
  is replaced by a `pureini`-based one.

### MCAP Web Inspector

(packaged with the wave; not separately versioned)

- JSON schema is the single source of truth for types: TS types are
  generated via `json-schema-to-typescript`, all component fields move
  from camelCase to snake_case, `bigint` is dropped from
  `McapInfoOutput` (converted at the boundary in `stats.ts`), and a
  `--link` flag is added.
- Metadata and attachment support: `McapInfoOutput` gains optional
  `metadata` and `attachments`; new `MetadataInfo` and `AttachmentInfo`
  shapes; `ChannelInfo` and `SchemaInfo` carry additional properties.
- `/view` page UX: drop the in-page `FileDropzone`, switch `FileInfo`
  to a responsive `SimpleGrid` (2/3/4 cols), wrap `CompressionTable` in
  a collapsed `Accordion`, and reduce `ScanStepper` to two steps with a
  subtle "Run exact scan…" link.
