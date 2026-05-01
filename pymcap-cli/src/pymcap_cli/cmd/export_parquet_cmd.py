"""Export an MCAP file to a directory of Parquet files (one per topic).

Output layout::

    out/
      sensor_lidar_front_points.parquet
      sensor_camera_front_camera_info.parquet
      ...
      _topics.parquet        # topic → file mapping + row counts

Nested ROS messages map to Parquet/Arrow ``STRUCT``, arrays to ``LIST``, and
``sensor_msgs/PointCloud2`` payloads are decoded into a ``LIST<STRUCT<...>>``
by :class:`pymcap_cli.encoding.pointcloud.Pointcloud2DecoderFactory`.

The output is consumable by any tool that reads Parquet — DuckDB, Polars,
pandas, Arrow, Spark, …::

    duckdb -c "SELECT count(*) FROM 'out/sensor_lidar_front_points.parquet'"
    pl.read_parquet("out/sensor_lidar_front_points.parquet")
"""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.parquet_exporter import ParquetExporter
from pymcap_cli.types.types_manual import (  # noqa: TC001 — runtime for cyclopts
    ForceOverwriteOption,
    OutputPathOption,
)


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
        exporter = ParquetExporter(
            batch_size=batch_size,
            writer_threads=writer_threads,
            compression=compression,
            include_blobs=include_blobs,
            skip_schema=skip_schema,
        )
    except ImportError:
        Console().print(
            "[red]Error:[/red] The 'pyarrow' package is required. "
            "Install with: uv add 'pymcap-cli[parquet]'"
        )
        return 1

    return run_export(
        file=file,
        output=output,
        exporter=exporter,
        topics=topic,
        force=force,
        num_workers=num_workers,
    )
