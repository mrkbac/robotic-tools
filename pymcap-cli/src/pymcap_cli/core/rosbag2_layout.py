"""rosbag2 split-directory discovery and aggregation.

rosbag2 writes a recording as a directory of split files named
``<bagname>/<bagname>_<N>.mcap`` (plus a ``metadata.yaml`` we intentionally
ignore). This module turns such a directory into an ordered list of split
files, and aggregates the per-split summaries into one logical view for ``info``.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from small_mcap import RebuildInfo, Statistics, Summary

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.utils import read_or_rebuild_info

if TYPE_CHECKING:
    from small_mcap import (
        AttachmentIndex,
        Channel,
        ChunkIndex,
        MessageIndex,
        MetadataIndex,
        Schema,
    )

_MCAP_SUFFIX = ".mcap"


def find_bag_splits(directory: Path) -> list[Path]:
    """Ordered split files for a rosbag2 directory.

    Globs ``<name>_<N>.mcap`` and sorts by the integer ``N`` (so ``_10`` follows
    ``_9``, not ``_1``). When no indexed splits exist, falls back to a single
    ``<name>.mcap`` if present. Returns ``[]`` when the directory holds no
    resolvable MCAP. Pure; never raises.
    """
    name = directory.name
    prefix = f"{name}_"
    indexed: list[tuple[int, Path]] = []
    for candidate in directory.glob(f"{prefix}*{_MCAP_SUFFIX}"):
        stem = candidate.name[len(prefix) : -len(_MCAP_SUFFIX)]
        if stem.isdigit():
            indexed.append((int(stem), candidate))
    if indexed:
        indexed.sort(key=lambda item: item[0])
        return [path for _, path in indexed]

    single = directory / f"{name}{_MCAP_SUFFIX}"
    if single.is_file():
        return [single]
    return []


def _is_url(path: str) -> bool:
    return urlparse(path).scheme in ("http", "https")


def expand_bag_paths(paths: list[str]) -> list[str]:
    """Flat-map input args, splicing rosbag2 directories into their split files.

    URLs and plain files pass through unchanged. A directory expands to its
    ordered split files (order preserved relative to surrounding args). A
    directory with no resolvable MCAP raises ``ValueError`` naming it.
    """
    expanded: list[str] = []
    for path in paths:
        if _is_url(path):
            expanded.append(path)
            continue
        candidate = Path(path)
        if not candidate.is_dir():
            expanded.append(path)
            continue
        splits = find_bag_splits(candidate)
        if not splits:
            raise ValueError(f"{path!r} is not an MCAP file or a rosbag2 bag directory")
        expanded.extend(str(split) for split in splits)
    return expanded


def read_aggregated_bag_info(
    splits: list[Path], *, rebuild: bool = False, exact_sizes: bool = False
) -> tuple[RebuildInfo, int]:
    """Read each split and fold them into one merged ``RebuildInfo``.

    Returns ``(merged_info, total_bytes)``. The merged info feeds the existing
    ``info_to_dict`` pipeline unchanged.
    """
    per_split: list[tuple[RebuildInfo, int]] = []
    for split in splits:
        with open_input(str(split), buffering=0) as (stream, size):
            info = read_or_rebuild_info(stream, size, rebuild=rebuild, exact_sizes=exact_sizes)
        per_split.append((info, size))
    merged = _merge_rebuild_infos(per_split)
    total_bytes = sum(size for _, size in per_split)
    return merged, total_bytes


def _merge_rebuild_infos(per_split: list[tuple[RebuildInfo, int]]) -> RebuildInfo:
    """Fold per-split ``RebuildInfo`` objects into one.

    Splits share a recording, so channel/schema ids are reused across files and
    are unioned. ``chunk_information`` is keyed by per-file byte offset, which
    collides across files; we shift each split's keys (and the matching
    ``ChunkIndex.chunk_start_offset``) by the cumulative byte size of preceding
    splits so the offset is globally unique and the ``info_to_dict`` join holds.
    """
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    chunk_indexes: list[ChunkIndex] = []
    attachment_indexes: list[AttachmentIndex] = []
    metadata_indexes: list[MetadataIndex] = []

    channel_message_counts: dict[int, int] = defaultdict(int)
    channel_sizes: dict[int, int] = defaultdict(int)
    chunk_information: dict[int, list[MessageIndex]] = {}

    message_count = attachment_count = metadata_count = chunk_count = 0
    start_times: list[int] = []
    end_times: list[int] = []
    estimated_sizes = False
    has_chunk_information = False
    has_channel_sizes = False
    base_offset = 0

    for info, size in per_split:
        summary = info.summary
        schemas.update(summary.schemas)
        channels.update(summary.channels)
        attachment_indexes.extend(summary.attachment_indexes)
        metadata_indexes.extend(summary.metadata_indexes)

        chunk_indexes.extend(
            dataclasses.replace(
                chunk_index,
                chunk_start_offset=chunk_index.chunk_start_offset + base_offset,
            )
            for chunk_index in summary.chunk_indexes
        )

        stats = summary.statistics
        if stats is not None:
            message_count += stats.message_count
            attachment_count += stats.attachment_count
            metadata_count += stats.metadata_count
            chunk_count += stats.chunk_count
            for channel_id, count in stats.channel_message_counts.items():
                channel_message_counts[channel_id] += count
            if stats.message_count > 0:
                start_times.append(stats.message_start_time)
                end_times.append(stats.message_end_time)

        if info.chunk_information is not None:
            has_chunk_information = True
            for offset, indexes in info.chunk_information.items():
                chunk_information[offset + base_offset] = indexes

        if info.channel_sizes is not None:
            has_channel_sizes = True
            for channel_id, channel_size in info.channel_sizes.items():
                channel_sizes[channel_id] += channel_size

        estimated_sizes = estimated_sizes or info.estimated_channel_sizes
        base_offset += size

    statistics = Statistics(
        message_count=message_count,
        schema_count=len(schemas),
        channel_count=len(channels),
        attachment_count=attachment_count,
        metadata_count=metadata_count,
        chunk_count=chunk_count,
        message_start_time=min(start_times) if start_times else 0,
        message_end_time=max(end_times) if end_times else 0,
        channel_message_counts=dict(channel_message_counts),
    )
    summary = Summary(
        statistics=statistics,
        schemas=schemas,
        channels=channels,
        chunk_indexes=chunk_indexes,
        attachment_indexes=attachment_indexes,
        metadata_indexes=metadata_indexes,
    )
    return RebuildInfo(
        header=per_split[0][0].header,
        summary=summary,
        channel_sizes=dict(channel_sizes) if has_channel_sizes else None,
        estimated_channel_sizes=estimated_sizes,
        chunk_information=chunk_information if has_chunk_information else None,
    )
