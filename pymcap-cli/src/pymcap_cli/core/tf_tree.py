"""Read and analyze TF transforms from an MCAP file.

Shared by `tftree` (live terminal rendering) and `tf-export` (URDF / SDF / JSON).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap import include_topics, read_message_decoded

from pymcap_cli.core.input_handler import open_input


@dataclass
class TransformData:
    """A single transform between two frames at one log_time."""

    frame_id: str
    child_frame_id: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    is_static: bool
    timestamp_ns: int


def read_transforms(
    file: str,
    *,
    include_dynamic: bool = False,
    snapshot_ns: int | None = None,
) -> dict[tuple[str, str], TransformData]:
    """Collapse all transforms to one per (parent, child).

    For `/tf_static` the last write wins. For `/tf` (when `include_dynamic`),
    the transform whose `timestamp_ns` is closest to `snapshot_ns` wins; if
    `snapshot_ns` is None, the most recent dynamic write wins.
    """
    topics = ["/tf_static"]
    if include_dynamic:
        topics.append("/tf")

    best: dict[tuple[str, str], TransformData] = {}

    with open_input(file) as (stream, _file_size):
        for msg in read_message_decoded(
            stream,
            should_include=include_topics(topics),
            decoder_factories=[DecoderFactory()],
        ):
            is_static = msg.channel.topic == "/tf_static"
            log_time = msg.message.log_time
            for transform_stamped in msg.decoded_message.transforms:
                trans = transform_stamped.transform.translation
                rot = transform_stamped.transform.rotation
                key = (transform_stamped.header.frame_id, transform_stamped.child_frame_id)
                prev = best.get(key)

                if prev is not None:
                    if is_static and not prev.is_static:
                        pass
                    elif prev.is_static and not is_static:
                        continue
                    elif snapshot_ns is None:
                        if log_time < prev.timestamp_ns:
                            continue
                    elif abs(log_time - snapshot_ns) >= abs(prev.timestamp_ns - snapshot_ns):
                        continue

                best[key] = TransformData(
                    frame_id=transform_stamped.header.frame_id,
                    child_frame_id=transform_stamped.child_frame_id,
                    translation=(trans.x, trans.y, trans.z),
                    rotation=(rot.x, rot.y, rot.z, rot.w),
                    is_static=is_static,
                    timestamp_ns=log_time,
                )

    return best


def detect_multiple_parents(
    transforms: dict[tuple[str, str], TransformData],
) -> dict[str, set[str]]:
    """Children that appear under more than one parent — violates tree shape."""
    child_to_parents: dict[str, set[str]] = defaultdict(set)
    for parent, child in transforms:
        child_to_parents[child].add(parent)
    return {child: parents for child, parents in child_to_parents.items() if len(parents) > 1}


def build_tree_and_find_roots(
    transforms: dict[tuple[str, str], TransformData],
) -> tuple[dict[str, list[str]], list[str]]:
    """Return (parent → [children], roots). Roots are parents that aren't children."""
    tree_dict: dict[str, list[str]] = defaultdict(list)
    parents: set[str] = set()
    children: set[str] = set()
    for transform in transforms.values():
        tree_dict[transform.frame_id].append(transform.child_frame_id)
        parents.add(transform.frame_id)
        children.add(transform.child_frame_id)
    root_frames = sorted(parents - children) or sorted(parents)
    return dict(tree_dict), root_frames


def detect_cycles(
    transforms: dict[tuple[str, str], TransformData],
) -> list[list[str]]:
    """Return frame cycles in the directed parent→child graph (DFS)."""
    tree_dict, _roots = build_tree_and_find_roots(transforms)
    cycles: list[list[str]] = []
    color: dict[str, int] = {}
    parent: dict[str, str] = {}

    def visit(node: str) -> None:
        color[node] = 1
        for child in tree_dict.get(node, []):
            if color.get(child) == 1:
                cycle = [child]
                cursor = node
                while cursor != child and cursor in parent:
                    cycle.append(cursor)
                    cursor = parent[cursor]
                cycle.append(child)
                cycles.append(list(reversed(cycle)))
            elif color.get(child) != 2:
                parent[child] = node
                visit(child)
        color[node] = 2

    for node in tree_dict:
        if color.get(node) is None:
            visit(node)
    return cycles


def quaternion_to_euler_rad(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Quaternion → (roll, pitch, yaw) in radians."""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
