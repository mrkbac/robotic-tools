from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PointcloudCleanupConfig:
    enabled: bool
    drop_invalid: bool
    sort_field: str | None


def pointcloud_worker_count(*, max_workers: int = 4) -> int:
    """Return the measured throughput knee, bounded for the caller's workload."""
    return min(max_workers, max(2, (os.cpu_count() or 4) - 2))


def resolve_pointcloud_cleanup(
    *,
    pointcloud_compression_enabled: bool,
    pointcloud_drop_invalid: bool | None,
    pointcloud_sort_field: str | None,
) -> PointcloudCleanupConfig:
    has_cleanup_flag = pointcloud_drop_invalid is not None or pointcloud_sort_field is not None

    drop_invalid = (
        pointcloud_drop_invalid
        if pointcloud_drop_invalid is not None
        else pointcloud_compression_enabled or has_cleanup_flag
    )
    sort_field = _normalize_sort_field(pointcloud_sort_field)

    return PointcloudCleanupConfig(
        enabled=drop_invalid or sort_field is not None,
        drop_invalid=drop_invalid,
        sort_field=sort_field,
    )


def _normalize_sort_field(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("--pointcloud-sort-field must not be empty")
    if normalized.lower() == "none":
        return None
    return normalized
