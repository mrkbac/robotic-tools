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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pointcloud2 import Pointcloud2Msg


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

    points = read_points(msg)

    mask: np.ndarray | None = None
    if drop_invalid:
        x, y, z = points["x"], points["y"], points["z"]
        invalid = np.isnan(x) | np.isnan(y) | np.isnan(z) | ((x == 0.0) & (y == 0.0) & (z == 0.0))
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
