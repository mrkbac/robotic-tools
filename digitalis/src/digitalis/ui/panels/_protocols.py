"""Shared Protocol types for ROS2 message fields used across panels."""

from typing import Protocol


class Stamp(Protocol):
    """ROS2 builtin_interfaces/Time."""

    sec: int
    nanosec: int


class Header(Protocol):
    """ROS2 std_msgs/Header."""

    frame_id: str
    stamp: Stamp
