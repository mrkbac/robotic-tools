# Copyright 2008 Willow Garage, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Based on
# https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs_py/sensor_msgs_py/point_cloud2.py


"""PointCloud2 lib for non ROS environment.

.. include:: ../../README.md

"""

from __future__ import annotations

import array
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

if TYPE_CHECKING:
    from pointcloud2.messages import Pointcloud2Msg, PointFieldDict, PointFieldMsg


__docformat__ = 'google'


@dataclass
class PointField:
    """PointField holds the description of one point entry in the PointCloud2 message format.

    Based on https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/PointField.msg
    """

    name: str
    offset: int
    datatype: int
    count: int = 1

    INT8: ClassVar[int] = 1
    UINT8: ClassVar[int] = 2
    INT16: ClassVar[int] = 3
    UINT16: ClassVar[int] = 4
    INT32: ClassVar[int] = 5
    UINT32: ClassVar[int] = 6
    FLOAT32: ClassVar[int] = 7
    FLOAT64: ClassVar[int] = 8


FIELD_TYPE_TO_NP: dict[int, np.dtype] = {}
FIELD_TYPE_TO_NP[PointField.INT8] = np.dtype(np.int8)
FIELD_TYPE_TO_NP[PointField.UINT8] = np.dtype(np.uint8)
FIELD_TYPE_TO_NP[PointField.INT16] = np.dtype(np.int16)
FIELD_TYPE_TO_NP[PointField.UINT16] = np.dtype(np.uint16)
FIELD_TYPE_TO_NP[PointField.INT32] = np.dtype(np.int32)
FIELD_TYPE_TO_NP[PointField.UINT32] = np.dtype(np.uint32)
FIELD_TYPE_TO_NP[PointField.FLOAT32] = np.dtype(np.float32)
FIELD_TYPE_TO_NP[PointField.FLOAT64] = np.dtype(np.float64)

NP_TO_FIELD_TYPE: dict[Any, int] = {v: k for k, v in FIELD_TYPE_TO_NP.items()}


@dataclass
class PointCloud2:
    """PointCloud2 message interface.

    @private

    Based on: https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/PointCloud2.msg
    """

    header: Any
    height: int
    width: int
    fields: Sequence[PointFieldMsg]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool


DUMMY_FIELD_PREFIX = 'unnamed_field'


def _normalize_fields(fields: Iterable[PointFieldMsg | PointFieldDict]) -> list[PointField]:
    """Convert fields to a list of PointField objects with calculated offsets.

    Args:
        fields: The fields to normalize.

    Returns:
        A list of PointField objects.
    """
    normalized_fields = []
    current_offset = 0

    for field in fields:
        if isinstance(field, dict):
            # PointFieldDict case
            field_datatype = field['datatype']
            field_name = field.get('name', '')
            field_count = field.get('count', 1)

            # Calculate offset if not provided
            if 'offset' in field:
                field_offset = field['offset']
                current_offset = field_offset
            else:
                field_offset = current_offset

            # Create PointField object
            normalized_fields.append(
                PointField(
                    name=field_name, offset=field_offset, datatype=field_datatype, count=field_count
                )
            )

            # Update current_offset for automatic calculation
            if 'offset' not in field:
                datatype_size = FIELD_TYPE_TO_NP[field_datatype].itemsize
                current_offset += field_count * datatype_size
        else:
            # Already a PointField object
            normalized_fields.append(field)

    return normalized_fields


def dtype_from_fields(
    fields: Iterable[PointFieldMsg | PointFieldDict], point_step: int | None = None
) -> np.dtype:
    """Convert a Iterable of sensor_msgs.msg.PointField messages to a np.dtype.

    Example:
    >>> dtype_from_fields([PointField('x', 0, PointField.FLOAT32)])
    dtype([('x', '<f4')])


    Args:
        fields: The fields to convert.
        point_step: The point step of the point cloud. If None, the point step is
            calculated from the fields.

    Returns:
        The NumPy dtype.
    """
    # Normalize fields to PointField objects
    normalized_fields = _normalize_fields(fields)

    # Create a lists containing the names, offsets and datatypes of all fields
    field_names: list[str] = []
    field_offsets: list[int] = []
    field_datatypes: list[str] = []

    for i, field in enumerate(normalized_fields):
        # Datatype as numpy datatype
        datatype = FIELD_TYPE_TO_NP[field.datatype]
        # Name field
        name = f'{DUMMY_FIELD_PREFIX}_{i}' if not field.name else field.name
        # Handle fields with count > 1 by creating subfields with a suffix consisting
        # of '_' followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            subfield_name = f'{name}_{a}' if field.count > 1 else name
            assert subfield_name not in field_names, 'Duplicate field names are not allowed!'
            field_names.append(subfield_name)
            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)

    # Create a tuple for each field containing name and data type
    dtype_dict = {
        'names': field_names,
        'formats': field_datatypes,
        'offsets': field_offsets,
    }
    if point_step is not None:
        dtype_dict['itemsize'] = point_step
    return np.dtype(dtype_dict)


def fields_from_dtype(dtype: np.dtype) -> list[PointFieldMsg]:
    """Convert a NumPy dtype to a list of PointField messages.

    Example:
    >>> fields_from_dtype(np.dtype([('x', '<f4')]))
    [PointField(name='x', offset=0, datatype=7, count=1)]

    Args:
        dtype: The NumPy dtype to convert.

    Returns:
        A list of PointField messages.
    """
    fields = []
    for name, (dt, offset) in dtype.fields.items():
        fields.append(PointField(name, offset, NP_TO_FIELD_TYPE[np.dtype(dt.type)]))

    return fields


def read_points(
    cloud: Pointcloud2Msg,
    field_names: list[str] | None = None,
    *,
    skip_nans: bool = False,
    uvs: Iterable[int] | np.ndarray | None = None,
    reshape_organized_cloud: bool = False,
) -> np.ndarray:
    """Read points from a sensor_msgs.PointCloud2 compatible type.

    See `pointcloud2.messages.Pointcloud2Msg` for more information.

    Args:
        cloud: The point cloud to read from `pointcloud2.messages.Pointcloud2Msg`.
        field_names: The names of fields to read. If None, read all fields.
        skip_nans: If True, then don't return any point with a NaN value.
        uvs: If specified, then only return the points at the given coordinates.
        reshape_organized_cloud: Returns the array as an 2D organized point cloud if set.

    Returns:
        Structured NumPy array containing points.
    """
    points = np.ndarray(
        shape=(cloud.width * cloud.height,),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data,
    )

    # Keep only the requested fields
    if field_names is not None:
        assert all(field_name in points.dtype.names for field_name in field_names), (
            'Requests field is not in the fields of the PointCloud!'
        )
        # Mask fields
        points = points[list(field_names)]

    # Swap array if byte order does not match
    if bool(sys.byteorder != 'little') != bool(cloud.is_bigendian):
        points = points.byteswap()

    # Check if we want to drop points with nan values
    if skip_nans and not cloud.is_dense:
        # Init mask which selects all points
        not_nan_mask = np.ones(len(points), dtype=bool)
        for field_name in points.dtype.names:
            # Only keep points without any non values in the mask
            not_nan_mask = np.logical_and(not_nan_mask, ~np.isnan(points[field_name]))
        # Select these points
        points = points[not_nan_mask]

    # Select points indexed by the uvs field
    if uvs is not None:
        # Don't convert to numpy array if it is already one
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        # Index requested points
        points = points[uvs]

    # Cast into 2d array if cloud is 'organized'
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)

    return points


def create_cloud(
    header: Any,
    fields: Iterable[PointFieldMsg | PointFieldDict],
    points: np.ndarray,
    step: int | None = None,
) -> PointCloud2:
    """Create a PointCloud2 message.

    Args:
        header: The point cloud header, see `pointcloud2.messages.Pointcloud2Msg.header`
        fields: The point cloud fields. Can be a list of `PointField` objects, or a
            list of dictionaries with keys `name`, `offset`, `datatype`, and `count`.
        points: The point cloud points. List of iterables, i.e. one iterable
                   for each point, with the elements of each iterable being the
                   values of the fields for that point (in the same order as
                   the fields parameter)
        step: The point step of the point cloud. If None, the point step is
            calculated from the fields.

    Returns:
        The point cloud as PointCloud2 message.
    """
    # Check if input is numpy array
    if isinstance(points, np.ndarray):
        # Check if this is an unstructured array
        if points.dtype.names is None:
            # Convert unstructured to structured array
            points = unstructured_to_structured(
                points,
                dtype=dtype_from_fields(fields, point_step=step),
            )
        else:
            assert points.dtype == dtype_from_fields(fields, point_step=step), (
                'PointFields and structured NumPy array dtype do not match for all fields! \
                    Check their field order, names and types.'
            )
    else:
        # Cast python objects to structured NumPy array (slow)
        points = np.array(
            # Points need to be tuples in the structured array
            list(map(tuple, points)),
            dtype=dtype_from_fields(fields, point_step=step),
        )

    # Handle organized clouds
    assert len(points.shape) <= 2, (
        'Too many dimensions for organized cloud! \
            Points can only be organized in max. two dimensional space'
    )
    height = 1
    width = points.shape[0]
    # Check if input points are an organized cloud (2D array of points)
    if len(points.shape) == 2:
        height = points.shape[1]

    # Convert numpy points to array.array
    memory_view = memoryview(points)
    casted = memory_view.cast('B')
    array_array = array.array('B')
    array_array.frombytes(casted)

    # Convert fields to PointField objects if they are dictionaries
    converted_fields = _normalize_fields(fields)

    # Put everything together
    return PointCloud2(
        header=header,
        height=height,
        width=width,
        fields=converted_fields,
        is_bigendian=sys.byteorder != 'little',
        point_step=points.dtype.itemsize,
        row_step=points.dtype.itemsize * width,
        data=array_array.tobytes(),
        is_dense=False,
    )
