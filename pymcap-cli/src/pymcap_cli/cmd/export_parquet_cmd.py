"""Export an MCAP file to a directory of Parquet files (one per topic).

Output layout::

    out/
      sensor_lidar_front_points.parquet
      sensor_camera_front_camera_info.parquet
      ...
      _topics.parquet        # topic → file mapping + row counts

Nested ROS messages map to Parquet/Arrow ``STRUCT``, arrays to ``LIST``, and
``sensor_msgs/PointCloud2`` payloads are decoded into a ``LIST<STRUCT<...>>``
by :class:`pymcap_cli.encoding.pointcloud.Pointcloud2DecoderFactory` (which
delegates to :func:`pointcloud2.read_points`).

The output is consumable by any tool that reads Parquet — DuckDB, Polars,
pandas, Arrow, Spark, …::

    duckdb -c "SELECT count(*) FROM 'out/sensor_lidar_front_points.parquet'"
    pl.read_parquet("out/sensor_lidar_front_points.parquet")
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Annotated, Any

from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from rich.console import Console
from small_mcap import read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_transform import create_progress, get_total_message_count
from pymcap_cli.encoding.arrow_schema import ArrowSchemaCache
from pymcap_cli.encoding.pointcloud import (
    CompressedPointcloud2DecoderFactory,
    Pointcloud2DecoderFactory,
)
from pymcap_cli.types.types_manual import (  # noqa: TC001 — runtime for cyclopts
    ForceOverwriteOption,
    OutputPathOption,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from small_mcap import Channel, DecodedMessage, Schema

console = Console()

_TABLE_NAME_RE = re.compile(r"[^0-9a-zA-Z_]+")

# Schemas whose payload is a raw media blob — useless for SQL analysis and
# often dominating the export size / time. Skipped by default unless the
# caller passes ``--include-blobs``.
_DEFAULT_BLOB_SCHEMAS: frozenset[str] = frozenset(
    {
        "sensor_msgs/msg/Image",
        "sensor_msgs/Image",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
        "foxglove_msgs/msg/CompressedImage",
        "foxglove_msgs/CompressedImage",
        "foxglove_msgs/msg/CompressedVideo",
        "foxglove_msgs/CompressedVideo",
        "foxglove_msgs/msg/RawImage",
        "foxglove_msgs/RawImage",
        "audio_common_msgs/msg/AudioData",
        "audio_common_msgs/AudioData",
    }
)

_COMPRESSED_POINTCLOUD2_SCHEMA = "point_cloud_interfaces/msg/CompressedPointCloud2"

# Structural fingerprint of ``builtin_interfaces/Time`` and ``/Duration``.
# Any ROS message with exactly these two fields is universally a timestamp
# value, so we collapse it to int nanoseconds so pyarrow can coerce it to
# its matching ``timestamp('ns')`` / ``duration('ns')`` column type.
_TIME_FIELDS: frozenset[str] = frozenset({"sec", "nanosec"})

# Cap on in-flight write futures — each queued arrow Table can be hundreds of
# MB, so keep the backlog shallow.
_MAX_WRITER_BACKLOG = 4


def _topic_to_filename(topic: str) -> str:
    """Map an MCAP topic to a safe Parquet filename stem (``/a/b`` → ``a_b``)."""
    name = _TABLE_NAME_RE.sub("_", topic).strip("_")
    if not name:
        name = "topic"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def _unique_topic_filename(topic: str, used_filenames: set[str]) -> str:
    filename = _topic_to_filename(topic)
    if filename not in used_filenames:
        return filename

    stem = filename
    suffix = 2
    while filename in used_filenames:
        filename = f"{stem}_{suffix}"
        suffix += 1
    return filename


def _to_plain(obj: Any) -> Any:
    """Recursively convert a decoded ROS message into plain dict/list/primitive values.

    The ROS2 decoder returns ``memoryview.cast('d')``/``'f'``/``'i'`` etc. for
    primitive fixed-size arrays — those memoryviews know their element width
    and are valid sequences of typed values. Only byte-format memoryviews
    (``'B'`` / ``'b'`` for ``uint8[]`` / ``int8[]``) should collapse to
    ``bytes``; for typed memoryviews we call ``.tolist()`` so downstream
    consumers see 9 floats, not 72 bytes.

    Values shaped like ``builtin_interfaces/Time`` / ``Duration`` (exactly
    ``sec`` + ``nanosec`` fields) collapse to int nanoseconds so pyarrow can
    coerce them to ``timestamp('ns')`` / ``duration('ns')``.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, memoryview):
        if obj.format in ("B", "b", "c"):
            return obj.tobytes()
        return obj.tolist()
    if isinstance(obj, dict):
        if obj.keys() == _TIME_FIELDS:
            return int(obj["sec"]) * 1_000_000_000 + int(obj["nanosec"])
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        if set(slots) == _TIME_FIELDS:
            return int(obj.sec) * 1_000_000_000 + int(obj.nanosec)
        return {k: _to_plain(getattr(obj, k)) for k in slots}
    dct = getattr(obj, "__dict__", None)
    if dct is not None:
        if dct.keys() == _TIME_FIELDS:
            return int(obj.sec) * 1_000_000_000 + int(obj.nanosec)
        return {k: _to_plain(v) for k, v in dct.items()}
    return str(obj)


def _build_row(msg: DecodedMessage) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    decoded = msg.decoded_message
    if isinstance(decoded, np.ndarray) and decoded.dtype.names:
        # PointCloud2 — keep the structured numpy array as-is; pyarrow preserves
        # the per-field dtypes when we hand it a StructArray at flush time.
        payload: dict[str, Any] = {"points": decoded}
    else:
        payload = _to_plain(decoded)
        if not isinstance(payload, dict):
            payload = {"value": payload}

    payload["_log_time"] = int(msg.message.log_time)
    payload["_publish_time"] = int(msg.message.publish_time)
    return payload


def _build_table(
    rows: list[dict[str, Any]],
    schema_type: pa.StructType | None = None,
) -> pa.Table:
    """Build a pyarrow Table from rows, preserving exact primitive widths.

    * Numpy structured array columns (PointCloud2 ``points``) become
      ``LIST<STRUCT<...>>`` with dtypes from the numpy layout.
    * If *schema_type* is supplied, each matching column is typed against the
      corresponding ROS field (``uint8`` → Arrow ``uint8``, ``float32`` →
      ``float32``, nested message → ``struct``) instead of pyarrow inferring
      ``int64`` / ``float64`` from Python values.
    * ``_log_time`` / ``_publish_time`` become ``timestamp('ns')``.
    """
    import numpy as np  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    field_types: dict[str, pa.DataType] = {}
    if schema_type is not None:
        for i in range(schema_type.num_fields):
            f = schema_type.field(i)
            field_types[f.name] = f.type

    column_names = list(rows[0].keys())
    arrays: list[pa.Array] = []
    for name in column_names:
        values = [row[name] for row in rows]
        if isinstance(values[0], np.ndarray) and values[0].dtype.names:
            # PointCloud2 points — build from the numpy structured dtype.
            field_names = list(values[0].dtype.names)
            concat = {n: np.concatenate([v[n] for v in values]) for n in field_names}
            struct_arr = pa.StructArray.from_arrays(
                [pa.array(concat[n]) for n in field_names],
                names=field_names,
            )
            offsets = np.zeros(len(values) + 1, dtype=np.int32)
            np.cumsum([len(v) for v in values], out=offsets[1:])
            arrays.append(pa.ListArray.from_arrays(pa.array(offsets), struct_arr))
        elif name in ("_log_time", "_publish_time"):
            arrays.append(pa.array(values, type=pa.timestamp("ns")))
        elif name in field_types:
            arrays.append(pa.array(values, type=field_types[name]))
        else:
            arrays.append(pa.array(values))
    return pa.Table.from_arrays(arrays, names=column_names)


class _TopicWriter:
    """Per-topic ParquetWriter serialised by a per-topic lock.

    ``pa.parquet.ParquetWriter`` is not thread-safe, so a pool of worker
    threads must not append to the same writer concurrently. Different
    topics' writers still run in parallel, which is where most of the
    parallelism comes from when several topics are active.
    """

    def __init__(self, path: Path, schema: pa.Schema, compression: str) -> None:
        import pyarrow.parquet as pq  # noqa: PLC0415

        self.path = path
        self._writer = pq.ParquetWriter(path, schema, compression=compression)
        self._lock = Lock()

    def write(self, batch: pa.Table) -> None:
        with self._lock:
            self._writer.write_table(batch)

    def close(self) -> None:
        with self._lock:
            self._writer.close()


def _flush(
    writer_pool: ThreadPoolExecutor,
    pending: deque[Future[None]],
    topic_writers: dict[str, _TopicWriter],
    output_dir: Path,
    filename: str,
    compression: str,
    rows: list[dict[str, Any]],
    schema_type: pa.StructType | None,
) -> None:
    """Build the pyarrow Table on the main thread, hand Parquet write to *writer_pool*."""
    if not rows:
        return
    batch = _build_table(rows, schema_type)
    rows.clear()

    # Open the topic's ParquetWriter lazily on the first batch — we need the
    # arrow schema (which comes from the batch) to create it.
    if filename not in topic_writers:
        topic_writers[filename] = _TopicWriter(
            output_dir / f"{filename}.parquet", batch.schema, compression
        )
    topic_writer = topic_writers[filename]

    # Backpressure: cap in-flight write futures to keep memory bounded.
    while pending and len(pending) >= _MAX_WRITER_BACKLOG:
        pending.popleft().result()
    pending.append(writer_pool.submit(topic_writer.write, batch))


def export_parquet(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    batch_size: Annotated[int, Parameter(name=["--batch-size"])] = 20000,
    num_workers: Annotated[int, Parameter(name=["--num-workers"])] = 8,
    writer_threads: Annotated[int, Parameter(name=["--writer-threads"])] = 4,
    compression: Annotated[str, Parameter(name=["--compression"])] = "zstd",
    topic: Annotated[list[str] | None, Parameter(name=["--topic", "-t"])] = None,
    include_blobs: Annotated[bool, Parameter(name=["--include-blobs"])] = False,
    skip_schema: Annotated[list[str] | None, Parameter(name=["--skip-schema"])] = None,
) -> int:
    """Export an MCAP file to a directory of Parquet files (one per topic).

    Parameters
    ----------
    file
        Input MCAP file (local path or HTTP/HTTPS URL).
    output
        Output directory. Will be created if it doesn't exist.
    force
        Overwrite existing ``*.parquet`` files in the output directory.
    batch_size
        Number of rows to buffer per topic before writing a row group.
    num_workers
        Chunk-decompression worker threads for the MCAP reader.
    writer_threads
        Parallel Parquet writers — different topics write concurrently.
    compression
        Parquet compression codec: ``zstd`` (default), ``snappy``, ``none``,
        or any other codec pyarrow accepts.
    topic
        Topic names to include. If omitted, all topics are exported.
    include_blobs
        Include schemas that carry raw media payloads (``sensor_msgs/Image``,
        ``CompressedImage``, ``CompressedVideo`` …). Off by default.
    skip_schema
        Extra schema names to exclude (in addition to the built-in blob list).
    """
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError:
        console.print(
            "[red]Error:[/red] The 'pyarrow' package is required. "
            "Install with: uv add 'pymcap-cli[parquet]'"
        )
        return 1

    out_dir = Path(output)
    if out_dir.exists() and not out_dir.is_dir():
        console.print(f"[red]Error:[/red] {out_dir} exists and is not a directory.")
        return 1
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        console.print(f"[red]Error:[/red] {out_dir} is not empty. Use --force to overwrite.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for p in out_dir.glob("*.parquet"):
            p.unlink()

    topic_filter = set(topic) if topic else None
    skipped_schemas: set[str] = set() if include_blobs else set(_DEFAULT_BLOB_SCHEMAS)
    if skip_schema:
        skipped_schemas.update(skip_schema)

    # Prefer decoding CompressedPointCloud2 if pureini is available; otherwise
    # fall back to skipping it so the export doesn't explode on an unknown schema.
    factories: list[Any] = [Pointcloud2DecoderFactory()]
    try:
        factories.append(CompressedPointcloud2DecoderFactory())
    except ImportError:
        skipped_schemas.add(_COMPRESSED_POINTCLOUD2_SCHEMA)
        console.print(
            "[dim]pureini not installed — skipping CompressedPointCloud2 "
            "(install [yellow]pymcap-cli[pointcloud][/yellow] to include).[/dim]"
        )
    factories.append(Ros2DecoderFactory())

    def _should_include(channel: Channel, schema: Schema | None) -> bool:
        if schema is not None and schema.name in skipped_schemas:
            return False
        return not (topic_filter is not None and channel.topic not in topic_filter)

    total = get_total_message_count(file)

    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    topic_files: dict[str, str] = {}
    used_filenames: set[str] = set()
    topic_counts: dict[str, int] = defaultdict(int)
    topic_schema_names: dict[str, str] = {}
    schema_cache = ArrowSchemaCache()
    topic_schema_types: dict[str, pa.StructType | None] = {}
    topic_writers: dict[str, _TopicWriter] = {}

    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {out_dir}/")
    if skipped_schemas:
        console.print(
            f"[dim]Skipping {len(skipped_schemas)} blob schema(s) — pass "
            f"[yellow]--include-blobs[/yellow] to include them.[/dim]"
        )

    pending_writes: deque[Future[None]] = deque()
    # 4 MB read buffer — the default 8 KB causes hundreds of thousands of tiny
    # syscalls on a multi-GB MCAP and caps sequential read at ~2 GB/s; 4 MB
    # buffer routinely doubles throughput on warm SSDs.
    read_buffer_bytes = 4 * 1024 * 1024
    try:
        with (
            open_input(file, buffering=read_buffer_bytes) as (stream, _size),
            create_progress(console, title="Exporting to Parquet") as progress,
            ThreadPoolExecutor(
                max_workers=writer_threads, thread_name_prefix="parquet-writer"
            ) as writer_pool,
        ):
            task_id = progress.add_task("Processing messages", total=total)
            for msg in read_message_decoded(
                stream,
                should_include=_should_include,
                decoder_factories=factories,
                num_workers=num_workers,
            ):
                progress.advance(task_id)
                topic_name = msg.channel.topic

                if topic_name not in topic_files:
                    import numpy as np  # noqa: PLC0415

                    filename = _unique_topic_filename(topic_name, used_filenames)
                    used_filenames.add(filename)
                    topic_files[topic_name] = filename
                    topic_schema_names[topic_name] = msg.schema.name if msg.schema else ""
                    if isinstance(msg.decoded_message, np.ndarray):
                        topic_schema_types[topic_name] = None
                    else:
                        topic_schema_types[topic_name] = schema_cache.get(msg.schema)

                try:
                    row = _build_row(msg)
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"[yellow]Warning:[/yellow] skipping message on {topic_name}: {exc}"
                    )
                    continue

                buffers[topic_name].append(row)
                topic_counts[topic_name] += 1
                if len(buffers[topic_name]) >= batch_size:
                    _flush(
                        writer_pool,
                        pending_writes,
                        topic_writers,
                        out_dir,
                        topic_files[topic_name],
                        compression,
                        buffers[topic_name],
                        topic_schema_types.get(topic_name),
                    )

            for topic_name, rows in buffers.items():
                _flush(
                    writer_pool,
                    pending_writes,
                    topic_writers,
                    out_dir,
                    topic_files[topic_name],
                    compression,
                    rows,
                    topic_schema_types.get(topic_name),
                )
            while pending_writes:
                pending_writes.popleft().result()
    finally:
        # Close all parquet writers so footers/metadata flush to disk.
        for w in topic_writers.values():
            w.close()

    # Write a topic index so callers can discover the per-topic files and row
    # counts without having to enumerate the directory.
    index_rows = [
        {
            "topic": t,
            "file": f"{topic_files[t]}.parquet",
            "schema": topic_schema_names.get(t, ""),
            "message_count": topic_counts[t],
        }
        for t in topic_files
    ]
    if index_rows:
        pq.write_table(pa.Table.from_pylist(index_rows), out_dir / "_topics.parquet")

    console.print(
        f"\n[green bold]✓ Exported {len(topic_files)} topic(s) to {out_dir}/[/green bold]"
    )
    for t_name in sorted(topic_files):
        fname = topic_files[t_name]
        console.print(
            f"  [cyan]{t_name}[/cyan] → [yellow]{fname}.parquet[/yellow] "
            f"({topic_counts[t_name]:,} rows)"
        )
    return 0
