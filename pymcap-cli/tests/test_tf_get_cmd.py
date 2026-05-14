from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

from pymcap_cli.cmd import tf_get_cmd
from pymcap_cli.core.tf_tree import TfGraph, TransformData, quaternion_to_euler_rad
from rich.console import Console

from tests.fixtures.mcap_generator import create_tf_mcap

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _td(
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
) -> TransformData:
    return TransformData(
        frame_id=parent,
        child_frame_id=child,
        translation=xyz,
        rotation=(0.0, 0.0, 0.0, 1.0),
        is_static=True,
        timestamp_ns=0,
    )


def _run_tf_get(
    path: Path,
    target: str,
    source: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    at: str | None = None,
) -> tuple[int, str]:
    output = io.StringIO()
    monkeypatch.setattr(
        tf_get_cmd,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=180),
    )
    rc = tf_get_cmd.tf_get(str(path), target, source, at=at)
    return rc, output.getvalue()


def _graph(transforms: dict[tuple[str, str], TransformData]) -> TfGraph:
    return TfGraph(transforms=transforms)


def test_tf_graph_lookup_direct_parent_from_child() -> None:
    result = _graph({("base", "camera"): _td("base", "camera", (1.0, 2.0, 3.0))}).lookup(
        target="base",
        source="camera",
    )

    assert result.transform.translation == (1.0, 2.0, 3.0)
    assert [step.from_frame for step in result.path] == ["camera"]
    assert [step.to_frame for step in result.path] == ["base"]
    assert result.path[0].is_inverted is False


def test_tf_graph_lookup_travels_up_and_down_tree() -> None:
    transforms = {
        ("root", "base"): _td("root", "base", (1.0, 0.0, 0.0)),
        ("base", "camera"): _td("base", "camera", (2.0, 0.0, 0.0)),
    }

    graph = _graph(transforms)
    up = graph.lookup(target="root", source="camera")
    down = graph.lookup(target="camera", source="root")

    assert up.transform.translation == (3.0, 0.0, 0.0)
    assert down.transform.translation == (-3.0, 0.0, 0.0)
    assert [step.is_inverted for step in up.path] == [False, False]
    assert [step.is_inverted for step in down.path] == [True, True]


def test_tf_graph_lookup_composes_rotated_translation() -> None:
    yaw_90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    transforms = {
        ("root", "mid"): TransformData(
            frame_id="root",
            child_frame_id="mid",
            translation=(1.0, 0.0, 0.0),
            rotation=yaw_90,
            is_static=True,
            timestamp_ns=0,
        ),
        ("mid", "leaf"): _td("mid", "leaf", (1.0, 0.0, 0.0)),
    }

    result = _graph(transforms).lookup(target="root", source="leaf")

    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            result.transform.translation,
            (1.0, 1.0, 0.0),
            strict=True,
        )
    )
    assert result.transform.rotation == yaw_90


def test_tf_graph_supports_add_lookup_add_lookup() -> None:
    graph = TfGraph()
    graph.add(
        static=True,
        stamp_ns=0,
        parent="root",
        child="base",
        translation=(1.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    first = graph.lookup(target="root", source="base")
    assert first.transform.translation == (1.0, 0.0, 0.0)

    graph.add(
        static=True,
        stamp_ns=0,
        parent="base",
        child="camera",
        translation=(0.0, 2.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    second = graph.lookup(target="root", source="camera")
    assert second.transform.translation == (1.0, 2.0, 0.0)


def test_tf_graph_lookup_uses_ros_xyzw_quaternion_order() -> None:
    yaw_90_xyzw = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    transforms = {
        ("root", "rotated"): TransformData(
            frame_id="root",
            child_frame_id="rotated",
            translation=(0.0, 0.0, 0.0),
            rotation=yaw_90_xyzw,
            is_static=True,
            timestamp_ns=0,
        ),
        ("rotated", "leaf"): _td("rotated", "leaf", (1.0, 0.0, 0.0)),
    }

    result = _graph(transforms).lookup(target="root", source="leaf")

    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            result.transform.translation,
            (0.0, 1.0, 0.0),
            strict=True,
        )
    )


def test_quaternion_to_euler_uses_ros_rpy_xyz_convention() -> None:
    roll, pitch, yaw = quaternion_to_euler_rad(
        0.03813457647485015,
        0.18930785741199999,
        0.2392983377447303,
        0.9515485246437885,
    )

    assert math.isclose(roll, math.radians(10.0), abs_tol=1e-12)
    assert math.isclose(pitch, math.radians(20.0), abs_tol=1e-12)
    assert math.isclose(yaw, math.radians(30.0), abs_tol=1e-12)


def test_tf_graph_lookup_between_siblings() -> None:
    transforms = {
        ("root", "left"): _td("root", "left", (1.0, 0.0, 0.0)),
        ("root", "right"): _td("root", "right", (0.0, 2.0, 0.0)),
    }

    result = _graph(transforms).lookup(target="left", source="right")

    assert result.transform.translation == (-1.0, 2.0, 0.0)
    assert [step.from_frame for step in result.path] == ["right", "root"]
    assert [step.to_frame for step in result.path] == ["root", "left"]


def test_tf_get_command_prints_result_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bag = tmp_path / "tf.mcap"
    bag.write_bytes(
        create_tf_mcap(
            static_edges=[
                ("base", "camera", (1.0, 2.0, 3.0)),
            ],
        )
    )

    rc, output = _run_tf_get(bag, "base", "camera", monkeypatch)

    assert rc == 0
    assert "base <- camera" in output
    assert "Traversal Path" in output
    assert "base -> camera" in output
    assert "/tf_static" in output
    assert "qx" not in output
    assert "qy" not in output
    assert "qz" not in output
    assert "qw" not in output


def test_tf_get_command_at_uses_dynamic_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bag = tmp_path / "tf.mcap"
    bag.write_bytes(
        create_tf_mcap(
            dynamic_edges=[
                ("odom", "base", (10.0, 0.0, 0.0)),
            ],
            dynamic_samples=3,
        )
    )

    rc, output = _run_tf_get(bag, "odom", "base", monkeypatch, at="240000000")

    assert rc == 0
    assert "odom <- base" in output
    assert "/tf" in output
    assert "200000000" in output


def test_tf_get_command_at_uses_ros_header_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bag = tmp_path / "tf.mcap"
    bag.write_bytes(
        create_tf_mcap(
            dynamic_edges=[
                ("odom", "base", (10.0, 0.0, 0.0)),
            ],
            dynamic_samples=3,
            dynamic_stamp_offset_ns=1_000_000_000,
        )
    )

    rc, output = _run_tf_get(bag, "odom", "base", monkeypatch, at="1240000000")

    assert rc == 0
    assert "1200000000" in output
    assert "300000000" not in output


def test_tf_get_command_rejects_disconnected_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bag = tmp_path / "tf.mcap"
    bag.write_bytes(
        create_tf_mcap(
            static_edges=[
                ("a", "b", (1.0, 0.0, 0.0)),
                ("x", "y", (1.0, 0.0, 0.0)),
            ],
        )
    )

    rc, _output = _run_tf_get(bag, "a", "y", monkeypatch)
    captured = capsys.readouterr()

    assert rc == 1
    assert "No TF path" in captured.err
    assert "target component" in captured.err
    assert "source component" in captured.err


def test_tf_get_command_rejects_multiple_parent_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bag = tmp_path / "tf.mcap"
    bag.write_bytes(
        create_tf_mcap(
            static_edges=[
                ("a", "c", (1.0, 0.0, 0.0)),
                ("b", "c", (1.0, 0.0, 0.0)),
            ],
        )
    )

    rc, _output = _run_tf_get(bag, "a", "c", monkeypatch)
    captured = capsys.readouterr()

    assert rc == 1
    assert "multiple_parents" in captured.err
