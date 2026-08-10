"""PCD exporter — one ``.pcd`` file per ``sensor_msgs/PointCloud2`` message."""

from __future__ import annotations

import math
import shutil
import struct
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from mcap_codec_support.pointcloud import (
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    CompressedPointCloudDecompressFactory,
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
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from _typeshed import SupportsWrite
    from small_mcap import Channel, DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext


class _PointFieldMessage(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def offset(self) -> int: ...

    @property
    def datatype(self) -> int: ...

    @property
    def count(self) -> int: ...


class _PointCloudMessage(Protocol):
    height: int
    width: int
    fields: Sequence[_PointFieldMessage]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool


@dataclass(frozen=True, slots=True)
class _Cloud:
    height: int
    width: int
    fields: tuple[_PointFieldMessage, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: memoryview
    is_dense: bool


@dataclass(frozen=True, slots=True)
class _Field:
    name: str
    offset: int
    datatype: int
    size: int
    pcd_type: str
    format_code: str
    unpacker: struct.Struct


@dataclass(frozen=True, slots=True)
class _RawField:
    name: str
    offset: int
    datatype: int
    count: int


_FIELD_FORMATS: dict[int, tuple[str, int, str]] = {
    1: ("b", 1, "I"),
    2: ("B", 1, "U"),
    3: ("h", 2, "I"),
    4: ("H", 2, "U"),
    5: ("i", 4, "I"),
    6: ("I", 4, "U"),
    7: ("f", 4, "F"),
    8: ("d", 8, "F"),
}


def _effective_row_step(width: int, point_step: int, row_step: int) -> int:
    return width * point_step if row_step == 0 else row_step


def _cloud_from_decoded(message: _PointCloudMessage) -> _Cloud:
    return _Cloud(
        height=int(message.height),
        width=int(message.width),
        fields=tuple(
            _RawField(
                name=field.name,
                offset=int(field.offset),
                datatype=int(field.datatype),
                count=int(field.count),
            )
            for field in message.fields
        ),
        is_bigendian=bool(message.is_bigendian),
        point_step=int(message.point_step),
        row_step=_effective_row_step(
            int(message.width),
            int(message.point_step),
            int(message.row_step),
        ),
        data=memoryview(message.data),
        is_dense=bool(message.is_dense),
    )


def _fields_from_cloud(cloud: _Cloud) -> tuple[_Field, ...]:
    byte_order = ">" if cloud.is_bigendian else "<"
    fields: list[_Field] = []
    for field_index, source in enumerate(cloud.fields):
        name = source.name
        offset = source.offset
        datatype = source.datatype
        count = source.count
        if count <= 0:
            raise ValueError(f"PointField {name!r} count must be positive")
        try:
            format_code, size, pcd_type = _FIELD_FORMATS[datatype]
        except KeyError as exc:
            raise ValueError(f"Unsupported PointField datatype {datatype}") from exc
        base_name = name or f"unnamed_field_{field_index}"
        for component in range(count):
            component_name = f"{base_name}_{component}" if count > 1 else base_name
            component_offset = offset + component * size
            if component_offset < 0 or component_offset + size > cloud.point_step:
                raise ValueError(
                    f"PointField {component_name!r} exceeds point_step {cloud.point_step}"
                )
            fields.append(
                _Field(
                    name=component_name,
                    offset=component_offset,
                    datatype=datatype,
                    size=size,
                    pcd_type=pcd_type,
                    format_code=format_code,
                    unpacker=struct.Struct(f"{byte_order}{format_code}"),
                )
            )
    if not fields:
        raise ValueError("PointCloud2 contains no fields")
    return tuple(fields)


def _combined_unpacker(cloud: _Cloud, fields: tuple[_Field, ...]) -> struct.Struct | None:
    cursor = 0
    format_parts = [">" if cloud.is_bigendian else "<"]
    for field in fields:
        if field.offset < cursor:
            return None
        gap = field.offset - cursor
        if gap:
            format_parts.append(f"{gap}x")
        format_parts.append(field.format_code)
        cursor = field.offset + field.size
    return struct.Struct("".join(format_parts))


def _validate_cloud(cloud: _Cloud) -> None:
    if cloud.width < 0 or cloud.height < 0:
        raise ValueError("PointCloud2 dimensions must be non-negative")
    if cloud.point_step <= 0:
        raise ValueError("PointCloud2 point_step must be positive")
    packed_row_step = cloud.width * cloud.point_step
    if cloud.row_step < packed_row_step:
        raise ValueError(
            f"PointCloud2 row_step {cloud.row_step} is smaller than width * point_step "
            f"({packed_row_step})"
        )
    expected_size = cloud.row_step * cloud.height
    if len(cloud.data) < expected_size:
        raise ValueError(
            f"PointCloud2 data has {len(cloud.data)} bytes, expected at least row_step * height "
            f"({expected_size})"
        )


def _iter_rows(cloud: _Cloud, fields: tuple[_Field, ...]) -> Iterator[str]:
    combined_unpacker = _combined_unpacker(cloud, fields)
    for row in range(cloud.height):
        row_offset = row * cloud.row_step
        for column in range(cloud.width):
            point_offset = row_offset + column * cloud.point_step
            values = (
                list(combined_unpacker.unpack_from(cloud.data, point_offset))
                if combined_unpacker is not None
                else [
                    field.unpacker.unpack_from(cloud.data, point_offset + field.offset)[0]
                    for field in fields
                ]
            )
            if not cloud.is_dense and any(
                field.pcd_type == "F" and math.isnan(float(value))
                for field, value in zip(fields, values, strict=True)
            ):
                continue
            yield " ".join(
                format(value, ".6g") if field.pcd_type == "F" else str(value)
                for field, value in zip(fields, values, strict=True)
            )


def _write_header(fh: SupportsWrite[str], fields: tuple[_Field, ...], point_count: int) -> None:
    fh.write("# .PCD v0.7 - Point Cloud Data file format\n")
    fh.write("VERSION 0.7\n")
    fh.write(f"FIELDS {' '.join(field.name for field in fields)}\n")
    fh.write(f"SIZE {' '.join(str(field.size) for field in fields)}\n")
    fh.write(f"TYPE {' '.join(field.pcd_type for field in fields)}\n")
    fh.write(f"COUNT {' '.join('1' for _field in fields)}\n")
    fh.write(f"WIDTH {point_count}\n")
    fh.write("HEIGHT 1\n")
    fh.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    fh.write(f"POINTS {point_count}\n")
    fh.write("DATA ascii\n")


def _write_rows(fh: SupportsWrite[str], rows: Iterator[str]) -> int:
    point_count = 0
    batch: list[str] = []
    for row in rows:
        batch.append(row)
        point_count += 1
        if len(batch) == 8192:
            fh.write("\n".join(batch))
            fh.write("\n")
            batch.clear()
    if batch:
        fh.write("\n".join(batch))
        fh.write("\n")
    return point_count


def write_pcd_ascii(path: Path, decoded: _PointCloudMessage) -> None:
    """Write PointCloud2 records as an ASCII PCD v0.7 file."""
    cloud = _cloud_from_decoded(decoded)
    _validate_cloud(cloud)
    fields = _fields_from_cloud(cloud)
    rows = _iter_rows(cloud, fields)

    if cloud.is_dense:
        point_count = cloud.width * cloud.height
        with path.open("w", encoding="ascii") as fh:
            _write_header(fh, fields, point_count)
            _write_rows(fh, rows)
        return

    with tempfile.SpooledTemporaryFile(
        mode="w+",
        encoding="ascii",
        max_size=8 * 1024 * 1024,
    ) as body:
        point_count = _write_rows(body, rows)
        body.seek(0)
        with path.open("w", encoding="ascii") as fh:
            _write_header(fh, fields, point_count)
            shutil.copyfileobj(body, fh)


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
        self._factories: list[Any] = []
        self._compressed_supported = is_compressed_codec_available()
        if self._compressed_supported:
            self._factories.append(CompressedPointCloudDecompressFactory())
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
