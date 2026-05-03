"""Parquet exporter — one ``<topic>.parquet`` per topic plus a ``_topics.parquet`` index.

Buffers rows per topic up to ``batch_size`` then dispatches each Arrow Table
to a shared :class:`concurrent.futures.ThreadPoolExecutor` so that distinct
topics' Parquet writers run in parallel.  ``pyarrow.parquet.ParquetWriter``
itself isn't thread-safe, so a per-writer lock serialises writes to the same
file.

Nested ROS messages map to Parquet/Arrow ``STRUCT``, arrays to ``LIST``, and
``sensor_msgs/PointCloud2`` payloads decoded by
:class:`mcap_codec_support.pointcloud.Pointcloud2DecoderFactory` become
``LIST<STRUCT<...>>``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar

from mcap_codec_support.pointcloud import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    CompressedPointCloudDecoderFactory,
    Pointcloud2DecoderFactory,
)
from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory

from pymcap_cli.encoding.arrow_schema import ArrowSchemaCache
from pymcap_cli.exporters._common import (
    SkipSchemaMixin,
    message_timestamps_ns,
    normalize_schema_name,
    prepare_output_file,
)
from pymcap_cli.exporters.base import Exporter, TopicWriter
from pymcap_cli.types.to_plain import to_plain

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pyarrow as pa
    from rich.console import Console
    from small_mcap import DecodedMessage

    from pymcap_cli.exporters.base import TopicContext


# Cap on in-flight write futures — each queued Arrow Table can be hundreds of
# MB, so keep the backlog shallow.
_MAX_WRITER_BACKLOG = 4


def _compressed_pointcloud_codec_available() -> bool:
    try:
        import pureini  # noqa: F401, PLC0415
    except ImportError:
        pass
    else:
        return True
    try:
        import DracoPy  # noqa: F401, PLC0415
    except ImportError:
        return False
    else:
        return True


def _build_row(msg: DecodedMessage) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    decoded = msg.decoded_message
    if isinstance(decoded, np.ndarray) and decoded.dtype.names:
        # PointCloud2 — keep the structured numpy array as-is; pyarrow preserves
        # the per-field dtypes when we hand it a StructArray at flush time.
        payload: dict[str, Any] = {"points": decoded}
    else:
        payload = to_plain(decoded)
        if not isinstance(payload, dict):
            payload = {"value": payload}

    log_time_ns, publish_time_ns = message_timestamps_ns(msg)
    payload["_log_time_ns"] = log_time_ns
    payload["_publish_time_ns"] = publish_time_ns
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
    * ``_log_time_ns`` / ``_publish_time_ns`` become ``timestamp('ns')``.
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
            field_names = list(values[0].dtype.names)
            concat = {n: np.concatenate([v[n] for v in values]) for n in field_names}
            struct_arr = pa.StructArray.from_arrays(
                [pa.array(concat[n]) for n in field_names],
                names=field_names,
            )
            offsets = np.zeros(len(values) + 1, dtype=np.int32)
            np.cumsum([len(v) for v in values], out=offsets[1:])
            arrays.append(pa.ListArray.from_arrays(pa.array(offsets), struct_arr))
        elif name in ("_log_time_ns", "_publish_time_ns"):
            arrays.append(pa.array(values, type=pa.timestamp("ns")))
        elif name in field_types:
            arrays.append(pa.array(values, type=field_types[name]))
        else:
            arrays.append(pa.array(values))
    return pa.Table.from_arrays(arrays, names=column_names)


class _LockedParquetWriter:
    """``pa.parquet.ParquetWriter`` with a lock — single file, multi-thread safe."""

    def __init__(self, path: Path, schema: pa.Schema, compression: str) -> None:
        import pyarrow.parquet as pq  # noqa: PLC0415

        self._writer = pq.ParquetWriter(path, schema, compression=compression)
        self._lock = Lock()

    def write(self, batch: pa.Table) -> None:
        with self._lock:
            self._writer.write_table(batch)

    def close(self) -> None:
        with self._lock:
            self._writer.close()


class _ParquetTopicWriter(TopicWriter):
    """Per-topic row buffer + lazy ParquetWriter, dispatched via shared pool."""

    def __init__(
        self,
        ctx: TopicContext,
        *,
        batch_size: int,
        compression: str,
        writer_pool: ThreadPoolExecutor,
        pending: deque[Future[None]],
        schema_cache: ArrowSchemaCache,
    ) -> None:
        self.topic = ctx.topic
        self.writer_key = ctx.writer_key
        self._path = prepare_output_file(
            ctx.output_path / f"{ctx.safe_filename}.parquet",
            force=ctx.force,
        )
        self.safe_filename = ctx.safe_filename
        self.schema_name = ctx.schema.name if ctx.schema is not None else ""
        self._batch_size = batch_size
        self._compression = compression
        self._writer_pool = writer_pool
        self._pending = pending
        self._schema_cache = schema_cache
        self._ctx_schema = ctx.schema
        self._buffer: list[dict[str, Any]] = []
        self._parquet_writer: _LockedParquetWriter | None = None
        self._schema_type: pa.StructType | None = None
        self._schema_type_resolved = False

    def _resolve_schema_type(self, msg: DecodedMessage) -> None:
        import numpy as np  # noqa: PLC0415

        if isinstance(msg.decoded_message, np.ndarray):
            self._schema_type = None
        else:
            self._schema_type = self._schema_cache.get(self._ctx_schema)
        self._schema_type_resolved = True

    def write(self, msg: DecodedMessage) -> None:
        if not self._schema_type_resolved:
            self._resolve_schema_type(msg)
        self._buffer.append(_build_row(msg))
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        batch = _build_table(self._buffer, self._schema_type)
        self._buffer.clear()
        if self._parquet_writer is None:
            self._parquet_writer = _LockedParquetWriter(self._path, batch.schema, self._compression)
        while len(self._pending) >= _MAX_WRITER_BACKLOG:
            self._pending.popleft().result()
        self._pending.append(self._writer_pool.submit(self._parquet_writer.write, batch))

    def close(self) -> None:
        self._flush()
        # Drain in-flight writes for this topic's writer before closing.
        while self._pending:
            self._pending.popleft().result()
        if self._parquet_writer is not None:
            self._parquet_writer.close()


class ParquetExporter(SkipSchemaMixin, Exporter):
    """Pluggable Parquet exporter.

    One ``<topic>.parquet`` per topic plus a ``_topics.parquet`` index file
    listing topic → file mappings + row counts. ``include_blobs=False``
    (default) drops raw media schemas; ``skip_schema`` adds extra exclusions.
    """

    name: ClassVar[str] = "parquet"

    def __init__(
        self,
        *,
        batch_size: int = 20000,
        writer_threads: int = 4,
        compression: str = "zstd",
        include_blobs: bool = False,
        skip_schema: list[str] | None = None,
    ) -> None:
        # Surface the missing-pyarrow case at construction time so the CLI
        # can return early before opening any inputs.
        import pyarrow as pa  # noqa: F401, PLC0415

        self._batch_size = batch_size
        self._writer_threads = writer_threads
        self._compression = compression
        self._set_skipped_schemas(
            include_blobs=include_blobs,
            skip_schema=skip_schema or (),
        )

        self._factories: list[Any] = [Pointcloud2DecoderFactory()]
        if _compressed_pointcloud_codec_available():
            self._factories.append(CompressedPointCloudDecoderFactory())
            self._compressed_pointcloud_warning: str | None = None
        else:
            self._skipped_schemas.add(normalize_schema_name(COMPRESSED_POINTCLOUD2_SCHEMA))
            self._skipped_schemas.add(normalize_schema_name(FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA))
            self._compressed_pointcloud_warning = (
                "[dim]point cloud codec support not installed — skipping compressed point clouds "
                "(install [yellow]pymcap-cli[pointcloud][/yellow] or "
                "[yellow]pymcap-cli[draco][/yellow] to include).[/dim]"
            )
        self._factories.append(Ros2DecoderFactory())

        self._schema_cache = ArrowSchemaCache()
        self._writer_pool: ThreadPoolExecutor | None = None
        self._pending: deque[Future[None]] = deque()
        self._writers: dict[int, _ParquetTopicWriter] = {}

    def decoder_factories(self) -> list[Any]:
        return list(self._factories)

    def setup(self, console: Console, output_path: Path) -> None:
        if self._compressed_pointcloud_warning:
            console.print(self._compressed_pointcloud_warning)
        if self._skipped_schemas:
            console.print(
                f"[dim]Skipping {len(self._skipped_schemas)} blob schema(s) — pass "
                f"[yellow]--include-blobs[/yellow] to include them.[/dim]"
            )
        # Clear any leftover .parquet files so partial reruns don't mix old +
        # new outputs (validate_output_dir leaves the directory intact).
        for p in output_path.glob("*.parquet"):
            p.unlink()
        self._writer_pool = ThreadPoolExecutor(
            max_workers=self._writer_threads, thread_name_prefix="parquet-writer"
        )

    def open_topic(self, ctx: TopicContext) -> _ParquetTopicWriter:
        if self._writer_pool is None:
            raise RuntimeError("ParquetExporter.setup() must be called before open_topic()")
        writer = _ParquetTopicWriter(
            ctx,
            batch_size=self._batch_size,
            compression=self._compression,
            writer_pool=self._writer_pool,
            pending=self._pending,
            schema_cache=self._schema_cache,
        )
        self._writers[ctx.writer_key] = writer
        return writer

    def finish(self, console: Console, output_path: Path, counts: Mapping[int, int]) -> None:
        # Drain anything still in-flight, then shut the writer pool down.
        while self._pending:
            self._pending.popleft().result()
        if self._writer_pool is not None:
            self._writer_pool.shutdown(wait=True)
            self._writer_pool = None

        if not self._writers:
            return

        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        index_rows = [
            {
                "topic": w.topic,
                "file": f"{w.safe_filename}.parquet",
                "schema": w.schema_name,
                "message_count": counts.get(writer_key, 0),
            }
            for writer_key, w in self._writers.items()
        ]
        pq.write_table(pa.Table.from_pylist(index_rows), output_path / "_topics.parquet")

        for w in sorted(self._writers.values(), key=lambda writer: writer.topic):
            console.print(
                f"  [cyan]{w.topic}[/cyan] → [yellow]{w.safe_filename}.parquet[/yellow] "
                f"({counts.get(w.writer_key, 0):,} rows)"
            )
