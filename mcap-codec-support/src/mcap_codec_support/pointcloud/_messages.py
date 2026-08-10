"""ROS-like message classes produced by point-cloud decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mcap_codec_support._messages import Header, _RosMessage
from mcap_codec_support.pointcloud.schemas import POINTCLOUD2 as POINTCLOUD2_SCHEMA_TEXT


@dataclass(slots=True)
class PointField(_RosMessage):
    name: str
    offset: int
    datatype: int
    count: int

    INT8: ClassVar[int] = 1
    UINT8: ClassVar[int] = 2
    INT16: ClassVar[int] = 3
    UINT16: ClassVar[int] = 4
    INT32: ClassVar[int] = 5
    UINT32: ClassVar[int] = 6
    FLOAT32: ClassVar[int] = 7
    FLOAT64: ClassVar[int] = 8

    _type: ClassVar[str] = "sensor_msgs/msg/PointField"
    _full_text: ClassVar[str] = """\
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8
string name
uint32 offset
uint8 datatype
uint32 count"""
    _fields_and_field_types: ClassVar[dict[str, str]] = {
        "name": "string",
        "offset": "uint32",
        "datatype": "uint8",
        "count": "uint32",
    }


@dataclass(slots=True)
class PointCloud2(_RosMessage):
    header: Header
    height: int
    width: int
    fields: list[PointField]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool

    _type: ClassVar[str] = "sensor_msgs/msg/PointCloud2"
    _full_text: ClassVar[str] = POINTCLOUD2_SCHEMA_TEXT
    _fields_and_field_types: ClassVar[dict[str, str]] = {
        "header": "std_msgs/Header",
        "height": "uint32",
        "width": "uint32",
        "fields": "sequence<sensor_msgs/PointField>",
        "is_bigendian": "bool",
        "point_step": "uint32",
        "row_step": "uint32",
        "data": "sequence<uint8>",
        "is_dense": "bool",
    }
