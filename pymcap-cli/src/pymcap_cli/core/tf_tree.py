"""Read and traverse TF transforms from an MCAP file.

Shared by `tftree`, `tf-get`, and `tf-export`.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntFlag
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

from mcap_ros2_support_fast.decoder import DecoderFactory
from small_mcap import include_topics, read_message_decoded

from pymcap_cli.constants import NS_TO_SEC
from pymcap_cli.core.input_handler import open_input

TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"

Edge = tuple[str, str]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class RosStamp(Protocol):
    sec: int
    nanosec: int


class _RosVec3(Protocol):
    x: float
    y: float
    z: float


class _RosQuat(Protocol):
    x: float
    y: float
    z: float
    w: float


class _RosTransform(Protocol):
    translation: _RosVec3
    rotation: _RosQuat


class _RosTfHeader(Protocol):
    stamp: RosStamp
    frame_id: str


class _RosTransformStamped(Protocol):
    header: _RosTfHeader
    child_frame_id: str
    transform: _RosTransform


class TfMessageLike(Protocol):
    """A decoded ``tf2_msgs/msg/TFMessage`` (list of stamped transforms)."""

    transforms: Sequence[_RosTransformStamped]


class TfTopicFlag(IntFlag):
    DYNAMIC = 1
    STATIC = 2


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
    transforms: dict[Edge, TransformData] = field(default_factory=dict)
    snapshot_ns: int | None = None
    counts: dict[Edge, int] = field(default_factory=dict)
    topic_flags_by_edge: dict[Edge, TfTopicFlag] = field(default_factory=dict)
    keep_series: bool = False
    series: dict[Edge, list[TransformData]] = field(default_factory=dict)
    _parent_to_edges: dict[str, list[Edge]] = field(default_factory=dict, init=False, repr=False)
    _child_to_edges: dict[str, list[Edge]] = field(default_factory=dict, init=False, repr=False)
    _frames: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        for edge in self.transforms:
            self._index_edge(edge)
        for edge, samples in self.series.items():
            self._index_edge(edge)
            samples.sort(key=lambda sample: sample.timestamp_ns)

    @property
    def frames(self) -> set[str]:
        return self._frames

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
        if frame not in self._frames:
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
        self.counts[edge] = self.counts.get(edge, 0) + 1
        self.topic_flags_by_edge[edge] = self.topic_flags_by_edge.get(edge, TfTopicFlag(0)) | (
            TfTopicFlag.STATIC if static else TfTopicFlag.DYNAMIC
        )

        transform = TransformData(
            frame_id=parent,
            child_frame_id=child,
            translation=translation,
            rotation=rotation,
            is_static=static,
            timestamp_ns=stamp_ns,
        )
        if self.keep_series:
            insort(
                self.series.setdefault(edge, []),
                transform,
                key=lambda sample: sample.timestamp_ns,
            )
        previous = self.transforms.get(edge)
        if previous is None:
            self.transforms[edge] = transform
            self._index_edge(edge)
            return True
        if not self._should_replace(previous, transform):
            return False
        self.transforms[edge] = transform
        return True

    def lookup(self, *, target: str, source: str) -> TfLookupResult:
        path = self._prepare_path(target=target, source=source)
        result = TransformValue(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))
        for step in path:
            result = _compose_transforms(step.transform_to_next(), result)
        return _finalize_result(target=target, source=source, transform=result, path=path)

    def lookup_at(self, *, target: str, source: str, time_ns: int) -> TfLookupResult:
        """Lookup ``target_T_source`` interpolated at ``time_ns``.

        Mirrors ``tf2_ros.Buffer.lookup_transform(..., time=Time(time_ns))``: for
        each edge on the BFS path, samples bracketing ``time_ns`` are blended
        (lerp + slerp). Raises :class:`TfLookupError` when ``time_ns`` is
        outside any edge's recorded range (matches tf2's ExtrapolationException).
        Static edges are time-invariant and bypass interpolation.
        """
        if not self.keep_series:
            raise TfLookupError(
                "TF graph was built without per-edge time series; rebuild with keep_series=True"
            )
        path = self._prepare_path(target=target, source=source)
        result = TransformValue(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0))
        for step in path:
            edge_value = self._sample_at(step.edge, time_ns)
            if step.is_inverted:
                edge_value = _invert_transform(edge_value)
            result = _compose_transforms(edge_value, result)
        return _finalize_result(target=target, source=source, transform=result, path=path)

    def _prepare_path(self, *, target: str, source: str) -> tuple[TfPathStep, ...]:
        if not self.transforms:
            raise TfLookupError("No transforms found on /tf_static or /tf")
        if target not in self._frames:
            raise TfLookupError(f"Target frame '{target}' is not present in the TF graph")
        if source not in self._frames:
            raise TfLookupError(f"Source frame '{source}' is not present in the TF graph")
        path = self.path(source=source, target=target)
        if path is None:
            raise TfLookupError(f"No TF path connects source '{source}' to target '{target}'")
        return path

    def _sample_at(self, edge: Edge, time_ns: int) -> TransformValue:
        samples = self.series.get(edge)
        if not samples:
            raise TfLookupError(f"No samples recorded for edge {edge[0]} -> {edge[1]}")
        # Static transforms are time-invariant. If any sample on this edge is
        # static, treat the edge as static (matches ROS 2 /tf_static semantics).
        for sample in samples:
            if sample.is_static:
                return _transform_value_from_data(sample)
        earliest = samples[0].timestamp_ns
        latest = samples[-1].timestamp_ns
        if time_ns < earliest or time_ns > latest:
            raise TfLookupError(
                f"Extrapolation: requested time {time_ns} ns is outside "
                f"[{earliest}, {latest}] for edge {edge[0]} -> {edge[1]}"
            )
        idx = bisect_left(samples, time_ns, key=lambda sample: sample.timestamp_ns)
        if idx < len(samples) and samples[idx].timestamp_ns == time_ns:
            exact_idx = (
                bisect_right(
                    samples,
                    time_ns,
                    key=lambda sample: sample.timestamp_ns,
                )
                - 1
            )
            return _transform_value_from_data(samples[exact_idx])
        lo = samples[idx - 1]
        hi = samples[idx]
        span = hi.timestamp_ns - lo.timestamp_ns
        alpha = (time_ns - lo.timestamp_ns) / span if span else 0.0
        return _interpolate_transforms(lo, hi, alpha)

    def has_static_dynamic_overlap(self, edge: Edge) -> bool:
        flags = self.topic_flags_by_edge.get(edge, TfTopicFlag(0))
        return flags == (TfTopicFlag.STATIC | TfTopicFlag.DYNAMIC)

    def _should_replace(self, previous: TransformData, current: TransformData) -> bool:
        """Snapshot-view selection rule for which sample represents an edge.

        Tie-break is deterministic by read order:
        - Latest-mode (``snapshot_ns is None``): equal stamps replace (later
          message wins), so identical-stamped frames behave like a publisher's
          last write.
        - Snapshot-mode: equal distances keep the first sample (current does
          not replace), since strict ``<`` is used.
        """
        if current.is_static and not previous.is_static:
            return True
        if previous.is_static and not current.is_static:
            return False
        if self.snapshot_ns is None:
            return current.timestamp_ns >= previous.timestamp_ns
        current_delta = abs(current.timestamp_ns - self.snapshot_ns)
        previous_delta = abs(previous.timestamp_ns - self.snapshot_ns)
        return current_delta < previous_delta

    def _index_edge(self, edge: Edge) -> None:
        parent, child = edge
        insort(self._parent_to_edges.setdefault(parent, []), edge)
        insort(self._child_to_edges.setdefault(child, []), edge)
        self._frames.add(parent)
        self._frames.add(child)

    def _neighbor_steps(self, frame: str) -> Iterator[TfPathStep]:
        for edge in self._child_to_edges.get(frame, ()):
            yield TfPathStep(
                from_frame=frame,
                to_frame=edge[0],
                edge=edge,
                transform=self.transforms[edge],
                is_inverted=False,
            )
        for edge in self._parent_to_edges.get(frame, ()):
            yield TfPathStep(
                from_frame=frame,
                to_frame=edge[1],
                edge=edge,
                transform=self.transforms[edge],
                is_inverted=True,
            )


def read_tf_graph(
    file: str,
    *,
    include_dynamic: bool = False,
    snapshot_ns: int | None = None,
    keep_series: bool = False,
) -> TfGraph:
    topics = [TF_STATIC_TOPIC]
    if include_dynamic:
        topics.append(TF_TOPIC)

    graph = TfGraph(snapshot_ns=snapshot_ns, keep_series=keep_series)
    with open_input(file) as (stream, _file_size):
        for msg in read_message_decoded(
            stream,
            should_include=include_topics(topics),
            decoder_factories=[DecoderFactory()],
        ):
            add_tf_message(graph, msg.channel.topic, msg.decoded_message)
    return graph


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
    root_frames = sorted(parents - children)
    return {parent: sorted(children) for parent, children in tree_dict.items()}, root_frames


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


def add_tf_message(graph: TfGraph, topic: str, message: TfMessageLike) -> bool:
    """Feed one decoded ``tf2_msgs/TFMessage`` into ``graph``.

    Returns True if this message introduced a new (parent, child) edge — callers
    use that to refresh a live display only when the tree structure changes.
    """
    is_static = topic == TF_STATIC_TOPIC
    added_new_edge = False
    for transform_stamped in message.transforms:
        edge = (transform_stamped.header.frame_id, transform_stamped.child_frame_id)
        if edge not in graph.transforms:
            added_new_edge = True
        trans = transform_stamped.transform.translation
        rot = transform_stamped.transform.rotation
        graph.add(
            static=is_static,
            stamp_ns=stamp_to_ns(transform_stamped.header.stamp),
            parent=transform_stamped.header.frame_id,
            child=transform_stamped.child_frame_id,
            translation=(trans.x, trans.y, trans.z),
            rotation=(rot.x, rot.y, rot.z, rot.w),
        )
    return added_new_edge


def _transform_value_from_data(transform: TransformData) -> TransformValue:
    return TransformValue(
        translation=transform.translation,
        rotation=_normalize_quaternion(transform.rotation),
    )


def _finalize_result(
    *,
    target: str,
    source: str,
    transform: TransformValue,
    path: tuple[TfPathStep, ...],
) -> TfLookupResult:
    """Normalize once at the end to absorb floating-point drift from chained multiplies."""
    return TfLookupResult(
        target=target,
        source=source,
        transform=TransformValue(
            translation=transform.translation,
            rotation=_normalize_quaternion(transform.rotation),
        ),
        path=path,
    )


def _interpolate_transforms(lo: TransformData, hi: TransformData, alpha: float) -> TransformValue:
    lo_tx, lo_ty, lo_tz = lo.translation
    hi_tx, hi_ty, hi_tz = hi.translation
    translation = (
        lo_tx + alpha * (hi_tx - lo_tx),
        lo_ty + alpha * (hi_ty - lo_ty),
        lo_tz + alpha * (hi_tz - lo_tz),
    )
    rotation = _slerp(
        _normalize_quaternion(lo.rotation),
        _normalize_quaternion(hi.rotation),
        alpha,
    )
    return TransformValue(translation=translation, rotation=rotation)


def _slerp(q0: Quat, q1: Quat, alpha: float) -> Quat:
    """Shortest-path quaternion SLERP. Inputs must be unit quaternions."""
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot
    # Near-collinear: fall back to nlerp to avoid divide-by-near-zero in sin(theta).
    if dot > 0.9995:
        result = (
            q0[0] + alpha * (q1[0] - q0[0]),
            q0[1] + alpha * (q1[1] - q0[1]),
            q0[2] + alpha * (q1[2] - q0[2]),
            q0[3] + alpha * (q1[3] - q0[3]),
        )
        return _normalize_quaternion(result)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (
        s0 * q0[0] + s1 * q1[0],
        s0 * q0[1] + s1 * q1[1],
        s0 * q0[2] + s1 * q1[2],
        s0 * q0[3] + s1 * q1[3],
    )


def _compose_transforms(left: TransformValue, right: TransformValue) -> TransformValue:
    """Return ``left * right`` for transforms represented as target_T_source.

    Both inputs must already be normalized; callers obtain ``TransformValue`` via
    :func:`_transform_value_from_data` which performs the single normalization.
    """
    rotated_translation = _rotate_vector(left.rotation, right.translation)
    return TransformValue(
        translation=(
            rotated_translation[0] + left.translation[0],
            rotated_translation[1] + left.translation[1],
            rotated_translation[2] + left.translation[2],
        ),
        rotation=_quaternion_multiply(left.rotation, right.rotation),
    )


def _invert_transform(transform: TransformValue) -> TransformValue:
    inverse_rotation = _quaternion_conjugate(transform.rotation)
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
    """Rotate ``vector`` by a unit ``quaternion``.

    Callers must pass an already-normalized quaternion. The hot path composes
    many transforms in sequence, so re-normalizing per rotation would dominate.
    """
    qx, qy, qz, qw = quaternion
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
