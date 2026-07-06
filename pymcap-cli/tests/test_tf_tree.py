"""Tests for the shared TF-graph accumulation helper."""

from __future__ import annotations

import dataclasses
from typing import Any

from pymcap_cli.core.tf_tree import TF_STATIC_TOPIC, TF_TOPIC, TfGraph, add_tf_message


def _mk(name: str, slots: list[str]) -> type:
    return dataclasses.make_dataclass(name, [(s, Any) for s in slots], slots=True)


def _tf_message(
    frame: str,
    child: str,
    *,
    sec: int = 1,
    nanosec: int = 0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> Any:
    time_cls = _mk("Time", ["sec", "nanosec"])
    header_cls = _mk("Header", ["stamp", "frame_id"])
    vec_cls = _mk("Vector3", ["x", "y", "z"])
    quat_cls = _mk("Quaternion", ["x", "y", "z", "w"])
    transform_cls = _mk("Transform", ["translation", "rotation"])
    stamped_cls = _mk("TransformStamped", ["header", "child_frame_id", "transform"])
    message_cls = _mk("TFMessage", ["transforms"])
    tx, ty, tz = translation
    qx, qy, qz, qw = rotation
    return message_cls(
        transforms=[
            stamped_cls(
                header=header_cls(stamp=time_cls(sec=sec, nanosec=nanosec), frame_id=frame),
                child_frame_id=child,
                transform=transform_cls(
                    translation=vec_cls(x=tx, y=ty, z=tz),
                    rotation=quat_cls(x=qx, y=qy, z=qz, w=qw),
                ),
            )
        ]
    )


def test_add_tf_message_populates_edge() -> None:
    graph = TfGraph()
    changed = add_tf_message(
        graph, TF_TOPIC, _tf_message("base_link", "wheel", translation=(1.0, 2.0, 3.0))
    )
    assert changed is True
    transform = graph.transforms[("base_link", "wheel")]
    assert transform.translation == (1.0, 2.0, 3.0)
    assert transform.is_static is False
    assert transform.timestamp_ns == 1_000_000_000


def test_add_tf_message_marks_static_by_topic() -> None:
    graph = TfGraph()
    add_tf_message(graph, TF_STATIC_TOPIC, _tf_message("base_link", "sensor"))
    assert graph.transforms[("base_link", "sensor")].is_static is True


def test_add_tf_message_returns_false_when_no_new_edge() -> None:
    graph = TfGraph()
    add_tf_message(graph, TF_TOPIC, _tf_message("a", "b"))
    # Re-adding the same edge (newer stamp) is not a structural change.
    changed = add_tf_message(graph, TF_TOPIC, _tf_message("a", "b", sec=2))
    assert changed is False
    assert graph.counts[("a", "b")] == 2
