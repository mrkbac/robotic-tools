"""Point cloud preprocessing that shrinks clouds before codec compression.

Two order-preserving-per-point transforms improve Cloudini/Draco output:

* Drop invalid points — Some lidar sensors emit fixed-width clouds padded with
  ``(0, 0, 0)`` (and, on non-dense clouds, NaN) placeholder returns. Removing
  them cuts the point count 20-40% with no information loss.
* Group points by a discrete field (``line`` — the laser ring index). Cloudini
  delta-encodes each field along point order, so clustering points from the
  same ring makes consecutive xyz and per-point timestamps nearly monotonic,
  which shrinks the deltas. Spatial sorts (Morton, sort-by-x) instead *hurt*,
  because they scramble the monotonic per-point timestamp field.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from pointcloud2 import Pointcloud2Msg

_FLOAT32_DATATYPE = 7

# ROS PointField datatype -> little-endian numpy dtype string.
_DATATYPE_TO_NP = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}
# Counting-sort bucket cap: a group-by key (e.g. lidar ring "line") is a small
# discrete index. Above this the raw counting-sort bails to the structured argsort.
_MAX_SORT_BUCKETS = 4096


def _field_offset(msg: Pointcloud2Msg, name: str) -> int | None:
    for field in msg.fields:
        if field.name == name and field.datatype == _FLOAT32_DATATYPE and field.count == 1:
            return field.offset
    return None


def _sort_field_offset_dtype(msg: Pointcloud2Msg, name: str) -> tuple[int, str] | None:
    for field in msg.fields:
        if field.name == name and field.count == 1:
            np_dtype = _DATATYPE_TO_NP.get(field.datatype)
            if np_dtype is None:
                return None
            return field.offset, np_dtype
    return None


def _invalid_xyz_mask_loop(
    x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray
) -> np.ndarray:
    mask = x_values == x_values
    for idx in range(x_values.shape[0]):
        x_value = x_values[idx]
        y_value = y_values[idx]
        z_value = z_values[idx]
        mask[idx] = (
            x_value != x_value
            or y_value != y_value
            or z_value != z_value
            or (x_value == 0.0 and y_value == 0.0 and z_value == 0.0)
        )
    return mask


@cache
def _invalid_xyz_mask_numba() -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    import numba as nb  # noqa: PLC0415

    return nb.njit(cache=True, fastmath=False, nogil=True)(_invalid_xyz_mask_loop)


def _copy_valid_points_loop(
    data: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    point_step: int,
    output: np.ndarray,
) -> int:
    kept = 0
    for idx in range(x_values.shape[0]):
        x_value = x_values[idx]
        y_value = y_values[idx]
        z_value = z_values[idx]
        if not (
            x_value != x_value
            or y_value != y_value
            or z_value != z_value
            or (x_value == 0.0 and y_value == 0.0 and z_value == 0.0)
        ):
            src_offset = idx * point_step
            dst_offset = kept * point_step
            for byte_offset in range(point_step):
                output[dst_offset + byte_offset] = data[src_offset + byte_offset]
            kept += 1
    return kept


def _copy_valid_word_points_loop(
    data: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    words_per_point: int,
    output: np.ndarray,
) -> int:
    kept = 0
    for idx in range(x_values.shape[0]):
        x_value = x_values[idx]
        y_value = y_values[idx]
        z_value = z_values[idx]
        if not (
            x_value != x_value
            or y_value != y_value
            or z_value != z_value
            or (x_value == 0.0 and y_value == 0.0 and z_value == 0.0)
        ):
            src_offset = idx * words_per_point
            dst_offset = kept * words_per_point
            for word_offset in range(words_per_point):
                output[dst_offset + word_offset] = data[src_offset + word_offset]
            kept += 1
    return kept


def _compact_group_words_loop(
    data: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    keys: np.ndarray,
    words_per_point: int,
    n_buckets: int,
    output: np.ndarray,
    bucket: np.ndarray,
    counts: np.ndarray,
) -> int:
    """Drop invalid points and stable-sort survivors by ``keys`` in one pass.

    A counting sort (O(n + n_buckets)): pass 1 tags each point with its bucket
    (-1 = invalid) and counts survivors per bucket; pass 2 scatters survivors —
    in original order within a bucket, so it is stable — into contiguous bucket
    runs. Operates on whole points as machine words (cache-friendly, no
    per-field structured copy). ``bucket`` (len n) and ``counts`` (len
    n_buckets+1, zeroed) are caller-provided scratch so the JIT kernel does no
    allocation.
    """
    n = x_values.shape[0]
    for idx in range(n):
        x_value = x_values[idx]
        y_value = y_values[idx]
        z_value = z_values[idx]
        if (
            x_value != x_value
            or y_value != y_value
            or z_value != z_value
            or (x_value == 0.0 and y_value == 0.0 and z_value == 0.0)
        ):
            bucket[idx] = -1
        else:
            b = keys[idx]
            bucket[idx] = b
            counts[b + 1] += 1
    for b in range(1, n_buckets + 1):
        counts[b] += counts[b - 1]
    total = counts[n_buckets]
    for idx in range(n):
        b = bucket[idx]
        if b < 0:
            continue
        dst = counts[b]
        counts[b] += 1
        src_offset = idx * words_per_point
        dst_offset = dst * words_per_point
        output[dst_offset : dst_offset + words_per_point] = data[
            src_offset : src_offset + words_per_point
        ]
    return total


@cache
def _compact_group_words_numba() -> Callable[..., int]:
    import numba as nb  # noqa: PLC0415

    return nb.njit(cache=True, fastmath=False, nogil=True)(_compact_group_words_loop)


@cache
def _copy_valid_points_numba() -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], int
]:
    import numba as nb  # noqa: PLC0415

    return nb.njit(cache=True, fastmath=False, nogil=True)(_copy_valid_points_loop)


@cache
def _copy_valid_word_points_numba() -> Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], int
]:
    import numba as nb  # noqa: PLC0415

    return nb.njit(cache=True, fastmath=False, nogil=True)(_copy_valid_word_points_loop)


def _invalid_xyz_mask(
    x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray
) -> np.ndarray:
    import numpy as np  # noqa: PLC0415

    try:
        return _invalid_xyz_mask_numba()(x_values, y_values, z_values)
    except ImportError:
        return (
            np.isnan(x_values)
            | np.isnan(y_values)
            | np.isnan(z_values)
            | ((x_values == 0.0) & (y_values == 0.0) & (z_values == 0.0))
        )


def _drop_invalid_preserve_order_raw(msg: Pointcloud2Msg) -> Pointcloud2Msg:
    import numpy as np  # noqa: PLC0415
    from pointcloud2 import PointCloud2  # noqa: PLC0415

    point_count = msg.width * msg.height
    expected_size = point_count * msg.point_step
    if expected_size > len(msg.data):
        return msg

    x_offset = _field_offset(msg, "x")
    y_offset = _field_offset(msg, "y")
    z_offset = _field_offset(msg, "z")
    if x_offset is None or y_offset is None or z_offset is None:
        return msg
    if max(x_offset, y_offset, z_offset) + 4 > msg.point_step:
        return msg

    data = np.frombuffer(msg.data, dtype=np.uint8, count=expected_size)
    x_values = np.ndarray(
        shape=(point_count,),
        dtype="<f4",
        buffer=msg.data,
        offset=x_offset,
        strides=(msg.point_step,),
    )
    y_values = np.ndarray(
        shape=(point_count,),
        dtype="<f4",
        buffer=msg.data,
        offset=y_offset,
        strides=(msg.point_step,),
    )
    z_values = np.ndarray(
        shape=(point_count,),
        dtype="<f4",
        buffer=msg.data,
        offset=z_offset,
        strides=(msg.point_step,),
    )

    output = np.empty(expected_size, dtype=np.uint8)
    if msg.point_step % 8 == 0:
        word_dtype = np.uint64
        word_size = 8
    elif msg.point_step % 4 == 0:
        word_dtype = np.uint32
        word_size = 4
    elif msg.point_step % 2 == 0:
        word_dtype = np.uint16
        word_size = 2
    else:
        word_dtype = None
        word_size = 1

    if word_dtype is None:
        copied_count = _copy_valid_points_numba()(
            data, x_values, y_values, z_values, msg.point_step, output
        )
    else:
        copied_count = _copy_valid_word_points_numba()(
            data.view(word_dtype),
            x_values,
            y_values,
            z_values,
            msg.point_step // word_size,
            output.view(word_dtype),
        )
    if copied_count == point_count:
        return msg

    return PointCloud2(
        header=msg.header,
        height=1,
        width=copied_count,
        fields=msg.fields,
        is_bigendian=False,
        point_step=msg.point_step,
        row_step=copied_count * msg.point_step,
        data=output[: copied_count * msg.point_step].tobytes(),
        is_dense=True,
    )


def _drop_invalid_and_group_raw(msg: Pointcloud2Msg, sort_field: str) -> Pointcloud2Msg | None:
    """Drop invalid points and group by ``sort_field`` in the raw word domain.

    Returns ``None`` (so the caller falls back to the structured argsort path)
    when the layout or key isn't suited to the fast counting sort: odd
    ``point_step``, missing/oversized xyz, unsupported key type, or a key whose
    value range isn't a small non-negative bucket index.
    """
    import numpy as np  # noqa: PLC0415
    from pointcloud2 import PointCloud2  # noqa: PLC0415

    point_count = msg.width * msg.height
    expected_size = point_count * msg.point_step
    if point_count <= 1 or expected_size > len(msg.data) or msg.point_step % 2 != 0:
        return None

    x_offset = _field_offset(msg, "x")
    y_offset = _field_offset(msg, "y")
    z_offset = _field_offset(msg, "z")
    if x_offset is None or y_offset is None or z_offset is None:
        return None
    if max(x_offset, y_offset, z_offset) + 4 > msg.point_step:
        return None

    key_info = _sort_field_offset_dtype(msg, sort_field)
    if key_info is None:
        return None
    key_offset, key_np_dtype = key_info
    if key_offset + np.dtype(key_np_dtype).itemsize > msg.point_step:
        return None

    x_values = np.ndarray((point_count,), "<f4", msg.data, x_offset, (msg.point_step,))
    y_values = np.ndarray((point_count,), "<f4", msg.data, y_offset, (msg.point_step,))
    z_values = np.ndarray((point_count,), "<f4", msg.data, z_offset, (msg.point_step,))
    key_raw = np.ndarray((point_count,), key_np_dtype, msg.data, key_offset, (msg.point_step,))

    keys = np.ascontiguousarray(key_raw)
    if keys.dtype.kind == "f":
        keys_int = keys.astype(np.int32)
        if not np.array_equal(keys_int, keys):  # non-integral key — not a bucket index
            return None
        keys = keys_int
    # Integer keys keep their native dtype (numba indexes any int type); only
    # signed types can be negative, so range-check just those.
    if keys.dtype.kind == "i" and keys.size and int(keys.min()) < 0:
        return None
    if keys.size and int(keys.max()) >= _MAX_SORT_BUCKETS:
        return None
    n_buckets = int(keys.max()) + 1 if keys.size else 1

    if msg.point_step % 8 == 0:
        word_dtype, word_size = np.uint64, 8
    elif msg.point_step % 4 == 0:
        word_dtype, word_size = np.uint32, 4
    else:
        word_dtype, word_size = np.uint16, 2

    data = np.frombuffer(msg.data, dtype=np.uint8, count=expected_size)
    output = np.empty(expected_size, dtype=np.uint8)
    # int16 bucket (values in [-1, _MAX_SORT_BUCKETS)) — less scratch traffic than int64.
    bucket = np.empty(point_count, dtype=np.int16)
    counts = np.zeros(n_buckets + 1, dtype=np.int64)
    kept = _compact_group_words_numba()(
        data.view(word_dtype),
        x_values,
        y_values,
        z_values,
        keys,
        msg.point_step // word_size,
        n_buckets,
        output.view(word_dtype),
        bucket,
        counts,
    )
    return PointCloud2(
        header=msg.header,
        height=1,
        width=kept,
        fields=msg.fields,
        is_bigendian=False,
        point_step=msg.point_step,
        row_step=kept * msg.point_step,
        data=output[: kept * msg.point_step].tobytes(),
        is_dense=True,
    )


def drop_invalid_and_reorder(
    msg: Pointcloud2Msg,
    *,
    drop_invalid: bool = True,
    sort_field: str | None = "line",
) -> Pointcloud2Msg:
    """Drop ``(0,0,0)``/NaN points and group remaining points by ``sort_field``.

    Returns the original ``msg`` unchanged when nothing applies (no xyz fields,
    big-endian data, empty cloud, or nothing to drop/sort) so callers pay no
    rebuild cost on clouds this cannot help.
    """
    import numpy as np  # noqa: PLC0415
    from pointcloud2 import create_cloud, read_points  # noqa: PLC0415

    if msg.width * msg.height == 0 or msg.point_step == 0 or msg.is_bigendian:
        return msg

    field_names = {field.name for field in msg.fields}
    if not {"x", "y", "z"}.issubset(field_names):
        return msg

    if drop_invalid and sort_field is None:
        try:
            return _drop_invalid_preserve_order_raw(msg)
        except ImportError:
            pass

    if drop_invalid and sort_field is not None and sort_field in field_names:
        # Fast raw counting-sort path; None means the layout/key isn't suited,
        # so fall through to the structured argsort path below.
        try:
            grouped = _drop_invalid_and_group_raw(msg, sort_field)
        except ImportError:
            grouped = None
        if grouped is not None:
            return grouped

    points = read_points(msg)

    mask: np.ndarray | None = None
    if drop_invalid:
        x, y, z = points["x"], points["y"], points["z"]
        invalid = _invalid_xyz_mask(x, y, z)
        if invalid.any():
            mask = ~invalid

    kept = points if mask is None else points[mask]

    order: np.ndarray | None = None
    if sort_field is not None and sort_field in field_names and kept.shape[0] > 1:
        order = np.argsort(kept[sort_field], kind="stable")

    if mask is None and order is None:
        return msg

    if order is not None:
        kept = kept[order]

    cloud = create_cloud(msg.header, msg.fields, kept, step=msg.point_step)
    # Every remaining point is finite and non-origin, so the cloud is now dense.
    cloud.is_dense = True
    return cloud
