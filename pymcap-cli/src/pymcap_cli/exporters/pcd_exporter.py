"""PCD exporter — one ``.pcd`` file per ``sensor_msgs/PointCloud2`` message.

Reuses :class:`mcap_codec_support.pointcloud.Pointcloud2DecoderFactory` to
decode the binary blob into a structured numpy array; writes ASCII PCD v0.7
(universally readable by ``pcl_viewer``, Open3D, CloudCompare).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from mcap_codec_support.pointcloud import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    CompressedPointCloudDecoderFactory,
    Pointcloud2DecoderFactory,
    is_compressed_codec_available,
)
from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory

from pymcap_cli.exporters._common import (
    message_timestamps_ns,
    normalize_schema_name,
    prepare_topic_dir,
    schema_name_in,
    unique_message_path,
)
from pymcap_cli.exporters.base import Exporter

if TYPE_CHECKING:
    from pathlib import Path

    from small_mcap import Channel, DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext


_PCD_TYPE_FROM_NUMPY: dict[str, str] = {
    "f": "F",
    "i": "I",
    "u": "U",
}


def _pcd_field_metadata(arr: np.ndarray) -> tuple[list[str], list[int], list[str], list[int]]:
    """Map numpy structured-dtype info → PCD FIELDS / SIZE / TYPE / COUNT lists."""
    if arr.dtype.names is None:
        raise ValueError("PointCloud2 numpy array has no field names")
    names: list[str] = list(arr.dtype.names)
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    fields = arr.dtype.fields
    if fields is None:
        raise ValueError("PointCloud2 dtype has no fields")
    for name in names:
        sub_dtype = fields[name][0]
        kind = sub_dtype.kind
        if kind not in _PCD_TYPE_FROM_NUMPY:
            raise ValueError(f"Unsupported numpy kind {kind!r} for PCD field {name!r}")
        types.append(_PCD_TYPE_FROM_NUMPY[kind])
        sizes.append(sub_dtype.itemsize)
        counts.append(1)
    return names, sizes, types, counts


def write_pcd_ascii(path: Path, points: np.ndarray) -> None:
    """Write a structured numpy array to an ASCII PCD v0.7 file."""
    if points.ndim != 1:
        points = points.reshape(-1)
    names, sizes, types, counts = _pcd_field_metadata(points)
    n = points.shape[0]

    with path.open("w", encoding="ascii") as fh:
        fh.write("# .PCD v0.7 - Point Cloud Data file format\n")
        fh.write("VERSION 0.7\n")
        fh.write(f"FIELDS {' '.join(names)}\n")
        fh.write(f"SIZE {' '.join(str(s) for s in sizes)}\n")
        fh.write(f"TYPE {' '.join(types)}\n")
        fh.write(f"COUNT {' '.join(str(c) for c in counts)}\n")
        fh.write(f"WIDTH {n}\n")
        fh.write("HEIGHT 1\n")
        fh.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        fh.write(f"POINTS {n}\n")
        fh.write("DATA ascii\n")
        # np.savetxt is the fastest path for tabular output of structured arrays;
        # build a 2-D float view for the write.
        cols = [points[name] for name in names]
        stacked = np.column_stack(cols)
        np.savetxt(fh, stacked, fmt="%.6g")


# Canonical (short) schema names — compare via :func:`normalize_schema_name`.
_POINTCLOUD_SCHEMAS: frozenset[str] = frozenset(
    {
        "sensor_msgs/PointCloud2",
        "point_cloud_interfaces/CompressedPointCloud2",
        "foxglove_msgs/CompressedPointCloud",
    }
)


class _PcdTopicWriter:
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = dir_path
        self._used_counts: dict[int, int] = {}

    def write(self, msg: DecodedMessage) -> None:
        log_time_ns, _ = message_timestamps_ns(msg)
        path = unique_message_path(self.dir_path, log_time_ns, "pcd", self._used_counts)
        write_pcd_ascii(path, msg.decoded_message)

    def close(self) -> None:
        pass


class PcdExporter(Exporter):
    """Per-message ASCII PCD files under ``<output>/<topic>/<log_time_ns>.pcd``."""

    name: ClassVar[str] = "pcd"

    def __init__(self) -> None:
        self._factories: list[Any] = [Pointcloud2DecoderFactory()]
        self._compressed_supported = is_compressed_codec_available()
        if self._compressed_supported:
            self._factories.append(CompressedPointCloudDecoderFactory())
        self._factories.append(Ros2DecoderFactory())

    def decoder_factories(self) -> list[Any]:
        return list(self._factories)

    def accepts(self, channel: Channel, schema: Schema | None) -> bool:  # noqa: ARG002
        if not schema_name_in(schema, _POINTCLOUD_SCHEMAS):
            return False
        assert schema is not None
        canonical = normalize_schema_name(schema.name)
        if canonical in {
            normalize_schema_name(COMPRESSED_POINTCLOUD2_SCHEMA),
            normalize_schema_name(FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA),
        }:
            return self._compressed_supported
        return True

    def open_topic(self, ctx: TopicContext) -> _PcdTopicWriter:
        dir_path = prepare_topic_dir(ctx.output_path / ctx.safe_filename, force=ctx.force)
        return _PcdTopicWriter(dir_path)
