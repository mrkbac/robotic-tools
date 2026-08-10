"""ROS-like message classes shared by decoded codec outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


class _RosMessage:
    __slots__ = ()

    _type: ClassVar[str]
    _full_text: ClassVar[str]
    _fields_and_field_types: ClassVar[dict[str, str]]

    @classmethod
    def get_fields_and_field_types(cls) -> dict[str, str]:
        return cls._fields_and_field_types.copy()


@dataclass(slots=True)
class Time(_RosMessage):
    sec: int
    nanosec: int

    _type: ClassVar[str] = "builtin_interfaces/msg/Time"
    _full_text: ClassVar[str] = "int32 sec\nuint32 nanosec"
    _fields_and_field_types: ClassVar[dict[str, str]] = {
        "sec": "int32",
        "nanosec": "uint32",
    }


@dataclass(slots=True)
class Header(_RosMessage):
    stamp: Time
    frame_id: str

    _type: ClassVar[str] = "std_msgs/msg/Header"
    _full_text: ClassVar[str] = """\
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""
    _fields_and_field_types: ClassVar[dict[str, str]] = {
        "stamp": "builtin_interfaces/Time",
        "frame_id": "string",
    }
