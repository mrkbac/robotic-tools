"""Read and traverse TF transforms from an MCAP file.

Shared by `tftree`, `tf-get`, and `tf-export`.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Protocol

from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap import include_topics, read_message_decoded

from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.input_handler import open_input

Edge = tuple[str, str]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class RosStamp(Protocol):
    sec: int
    nanosec: int


@dataclass(slots=True)
class TransformData:
    """A single transform between two frames at one ROS header stamp.

    The transform is stored as ``frame_id_T_child_frame_id``: it maps coordinates
    expressed in ``child_frame_id`` into coordinates expressed in ``frame_id``.
    Quaternion components use ROS geometry_msgs order: ``x, y, z, w``.
    """

    frame_id: str
    child_frame_id: str
    translation: Vec3
    rotation: Quat
    is_static: bool
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class TransformValue:
    translation: Vec3
    rotation: Quat


@dataclass(frozen=True, slots=True)
class TfPathStep:
    from_frame: str
    to_frame: str
    edge: Edge
    transform: TransformData
    is_inverted: bool

    def transform_to_next(self) -> TransformValue:
        value = _transform_value_from_data(self.transform)
        if self.is_inverted:
            return _invert_transform(value)
        return value


@dataclass(frozen=True, slots=True)
class TfLookupResult:
    target: str
    source: str
    transform: TransformValue
    path: tuple[TfPathStep, ...]


class TfLookupError(ValueError):
    """A TF graph cannot answer a requested lookup."""


@dataclass(slots=True)
class TfGraph:
    transforms: dict[Edge, TransformData]
    counts: dict[Edge, int] = field(default_factory=dict)
    topics_by_edge: dict[Edge, set[str]] = field(default_factory=dict)

    @property
    def frames(self) -> set[str]:
        frames: set[str] = set()
        for parent, child in self.transforms:
            frames.add(parent)
            frames.add(child)
        return frames

    def path(self, *, source: str, target: str) -> tuple[TfPathStep, ...] | None:
        if source == target:
            return ()

        previous: dict[str, TfPathStep] = {}
        seen = {source}
        queue: deque[str] = deque([source])
        while queue:
            frame = queue.popleft()
            if frame == target:
                break
            for step in self._neighbor_steps(frame):
                if step.to_frame in seen:
                    continue
                seen.add(step.to_frame)
                previous[step.to_frame] = step
                queue.append(step.to_frame)

        if target not in seen:
            return None

        steps: list[TfPathStep] = []
        cursor = target
        while cursor != source:
            step = previous[cursor]
            steps.append(step)
            cursor = step.from_frame
        return tuple(reversed(steps))

    def component_for(self, frame: str) -> set[str]:
        if frame not in self.frames:
            return set()

        component = {frame}
        queue: deque[str] = deque([frame])
        while queue:
            current = queue.popleft()
            for step in self._neighbor_steps(current):
                if step.to_frame in component:
                    continue
                component.add(step.to_frame)
                queue.append(step.to_frame)
        return component

    def _neighbor_steps(self, frame: str) -> list[TfPathStep]:
        steps: list[TfPathStep] = []
        for edge, transform in sorted(self.transforms.items()):
            parent, child = edge
            if child == frame:
                steps.append(
                    TfPathStep(
                        from_frame=child,
                        to_frame=parent,
                        edge=edge,
                        transform=transform,
                        is_inverted=False,
                    )
                )
            if parent == frame:
                steps.append(
                    TfPathStep(
                        from_frame=parent,
                        to_frame=child,
                        edge=edge,
                        transform=transform,
                        is_inverted=True,
                    )
                )
        return sorted(steps, key=lambda item: (item.to_frame, item.edge))


class TfGraphBuilder:
    def __init__(self, *, snapshot_ns: int | None = None) -> None:
        self.snapshot_ns = snapshot_ns
        self.transforms: dict[Edge, TransformData] = {}
        self.counts: dict[Edge, int] = defaultdict(int)
        self.topics_by_edge: dict[Edge, set[str]] = defaultdict(set)

    def add(
        self,
        *,
        static: bool,
        stamp_ns: int,
        parent: str,
        child: str,
        translation: Vec3,
        rotation: Quat,
    ) -> bool:
        edge = (parent, child)
        self.counts[edge] += 1
        self.topics_by_edge[edge].add("/tf_static" if static else "/tf")

        transform = TransformData(
            frame_id=parent,
            child_frame_id=child,
            translation=translation,
            rotation=rotation,
            is_static=static,
            timestamp_ns=stamp_ns,
        )
        previous = self.transforms.get(edge)
        if previous is not None and not self._should_replace(previous, transform):
            return False

        self.transforms[edge] = transform
        return True

    def graph(self) -> TfGraph:
        return TfGraph(
            transforms=dict(self.transforms),
            counts=dict(self.counts),
            topics_by_edge={edge: set(topics) for edge, topics in self.topics_by_edge.items()},
        )

    def _should_replace(self, previous: TransformData, current: TransformData) -> bool:
        if current.is_static and not previous.is_static:
            return True
        if previous.is_static and not current.is_static:
            return False
        if self.snapshot_ns is None:
            return current.timestamp_ns >= previous.timestamp_ns
        current_delta = abs(current.timestamp_ns - self.snapshot_ns)
        previous_delta = abs(previous.timestamp_ns - self.snapshot_ns)
        return current_delta < previous_delta


def read_tf_graph(
    file: str,
    *,
    include_dynamic: bool = False,
    snapshot_ns: int | None = None,
) -> TfGraph:
    topics = ["/tf_static"]
    if include_dynamic:
        topics.append("/tf")

    builder = TfGraphBuilder(snapshot_ns=snapshot_ns)
    with open_input(file) as (stream, _file_size):
        for msg in read_message_decoded(
            stream,
            should_include=include_topics(topics),
            decoder_factories=[DecoderFactory()],
        ):
            for transform_stamped in msg.decoded_message.transforms:
                trans = transform_stamped.transform.translation
                rot = transform_stamped.transform.rotation
                builder.add(
                    static=msg.channel.topic == "/tf_static",
                    stamp_ns=stamp_to_ns(transform_stamped.header.stamp),
                    parent=transform_stamped.header.frame_id,
                    child=transform_stamped.child_frame_id,
                    translation=(trans.x, trans.y, trans.z),
                    rotation=(rot.x, rot.y, rot.z, rot.w),
                )
    return builder.graph()


def read_transforms(
    file: str,
    *,
    include_dynamic: bool = False,
    snapshot_ns: int | None = None,
) -> dict[Edge, TransformData]:
    """Collapse all transforms to one per (parent, child)."""
    return read_tf_graph(
        file,
        include_dynamic=include_dynamic,
        snapshot_ns=snapshot_ns,
    ).transforms


def build_tree_and_find_roots(
    transforms: dict[Edge, TransformData],
) -> tuple[dict[str, list[str]], list[str]]:
    """Return (parent -> [children], roots). Roots are parents that are not children."""
    tree_dict: dict[str, list[str]] = defaultdict(list)
    parents: set[str] = set()
    children: set[str] = set()
    for transform in transforms.values():
        tree_dict[transform.frame_id].append(transform.child_frame_id)
        parents.add(transform.frame_id)
        children.add(transform.child_frame_id)
    root_frames = sorted(parents - children) or sorted(parents)
    return {parent: sorted(children) for parent, children in tree_dict.items()}, root_frames


def lookup_transform(
    graph: TfGraph,
    *,
    target: str,
    source: str,
) -> TfLookupResult:
    frames = graph.frames
    if not graph.transforms:
        raise TfLookupError("No transforms found on /tf_static or /tf")
    if target not in frames:
        raise TfLookupError(f"Target frame '{target}' is not present in the TF graph")
    if source not in frames:
        raise TfLookupError(f"Source frame '{source}' is not present in the TF graph")

    path = graph.path(source=source, target=target)
    if path is None:
        raise TfLookupError(f"No TF path connects source '{source}' to target '{target}'")

    result = TransformValue(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))
    for step in path:
        result = _compose_transforms(step.transform_to_next(), result)

    return TfLookupResult(target=target, source=source, transform=result, path=path)


def quaternion_to_euler_rad(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert ROS XYZW quaternion to roll, pitch, yaw radians."""
    x, y, z, w = _normalize_quaternion((x, y, z, w))
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def stamp_to_ns(stamp: RosStamp) -> int:
    return int(stamp.sec) * NS_TO_SEC + int(stamp.nanosec)


def _transform_value_from_data(transform: TransformData) -> TransformValue:
    return TransformValue(
        translation=transform.translation,
        rotation=_normalize_quaternion(transform.rotation),
    )


def _compose_transforms(left: TransformValue, right: TransformValue) -> TransformValue:
    """Return ``left * right`` for transforms represented as target_T_source."""
    left_rotation = _normalize_quaternion(left.rotation)
    right_rotation = _normalize_quaternion(right.rotation)
    rotated_translation = _rotate_vector(left_rotation, right.translation)
    return TransformValue(
        translation=(
            rotated_translation[0] + left.translation[0],
            rotated_translation[1] + left.translation[1],
            rotated_translation[2] + left.translation[2],
        ),
        rotation=_normalize_quaternion(_quaternion_multiply(left_rotation, right_rotation)),
    )


def _invert_transform(transform: TransformValue) -> TransformValue:
    rotation = _normalize_quaternion(transform.rotation)
    inverse_rotation = _quaternion_conjugate(rotation)
    tx, ty, tz = transform.translation
    inverse_translation = _rotate_vector(inverse_rotation, (-tx, -ty, -tz))
    return TransformValue(translation=inverse_translation, rotation=inverse_rotation)


def _quaternion_norm(quaternion: Quat) -> float:
    qx, qy, qz, qw = quaternion
    return math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)


def _normalize_quaternion(quaternion: Quat) -> Quat:
    norm = _quaternion_norm(quaternion)
    if not math.isfinite(norm) or norm == 0:
        raise TfLookupError("Cannot use a non-finite or zero-length quaternion")
    qx, qy, qz, qw = quaternion
    return qx / norm, qy / norm, qz / norm, qw / norm


def _quaternion_conjugate(quaternion: Quat) -> Quat:
    qx, qy, qz, qw = quaternion
    return -qx, -qy, -qz, qw


def _quaternion_multiply(left: Quat, right: Quat) -> Quat:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate_vector(quaternion: Quat, vector: Vec3) -> Vec3:
    qx, qy, qz, qw = _normalize_quaternion(quaternion)
    vx, vy, vz = vector
    uv = (
        qy * vz - qz * vy,
        qz * vx - qx * vz,
        qx * vy - qy * vx,
    )
    uuv = (
        qy * uv[2] - qz * uv[1],
        qz * uv[0] - qx * uv[2],
        qx * uv[1] - qy * uv[0],
    )
    return (
        vx + 2.0 * (qw * uv[0] + uuv[0]),
        vy + 2.0 * (qw * uv[1] + uuv[1]),
        vz + 2.0 * (qw * uv[2] + uuv[2]),
    )
