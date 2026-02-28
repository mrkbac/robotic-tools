"""Python typing protocol definitions for the PointCloud2 message."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class PointFieldDict(TypedDict, total=False):
    """A dictionary representation of a PointField.

    The `name` and `datatype` fields are required, while `count` and `offset` are optional.
    """

    name: str
    offset: int
    datatype: int
    count: int


class PointFieldMsg(Protocol):
    """PointField holds the description of one point entry in the PointCloud2 message format.

    Based on https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/PointField.msg
    """

    @property
    def name(self) -> str:
        """Name of field."""
        ...

    @property
    def offset(self) -> int:
        """Offset from start of point struct."""
        ...

    @property
    def datatype(self) -> int:
        """Datatype enumeration, see above."""
        ...

    @property
    def count(self) -> int:
        """How many elements in the field."""
        ...


class Pointcloud2Msg(Protocol):
    """Python Protocol for the ROS PointCloud2 message.

    Based on: https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/PointCloud2.msg

    This message holds a collection of N-dimensional points, which may
    contain additional information such as normals, intensity, etc. The
    point data is stored as a binary blob, its layout described by the
    contents of the "fields" array.
    The point cloud data may be organized 2d (image-like) or 1d (unordered).
    Point clouds organized as 2d images may be produced by camera depth sensors
    such as stereo or time-of-flight.
    """

    @property
    def header(self) -> Any:
        """Time of sensor data acquisition, and the coordinate frame ID (for 3d points)."""
        ...

    @property
    def height(self) -> int:
        """If the cloud is unordered, height is 1 and width is the length of the point cloud."""
        ...

    @property
    def width(self) -> int:
        """If the cloud is unordered, height is 1 and width is the length of the point cloud."""
        ...

    @property
    def fields(self) -> list[PointFieldMsg]:
        """Describes the channels and their layout in the binary data blob."""
        ...

    @property
    def is_bigendian(self) -> bool:
        """Is this data bigendian?."""
        ...

    @property
    def point_step(self) -> int:
        """Length of a point in bytes."""
        ...

    @property
    def row_step(self) -> int:
        """Length of a row in bytes."""
        ...

    @property
    def data(self) -> bytes:
        """Actual point data, size is (row_step*height)."""
        ...

    @property
    def is_dense(self) -> bool:
        """True if there are no invalid points."""
        ...
