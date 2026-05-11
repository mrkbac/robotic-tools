"""Unit tests for `pymcap-cli tf-export` and its renderers."""

from __future__ import annotations

import io
import json
import math
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd.tf_export_cmd import tf_export
from pymcap_cli.core.tf_tree import (
    TransformData,
    build_tree_and_find_roots,
    detect_cycles,
    detect_multiple_parents,
    quaternion_to_euler_rad,
    read_transforms,
)
from pymcap_cli.exporters.sdf_exporter import render_sdf
from pymcap_cli.exporters.urdf_exporter import render_urdf
from small_mcap import CompressionType, McapWriter

from tests.fixtures.mcap_generator import create_tf_mcap

if TYPE_CHECKING:
    from pathlib import Path


def _td(
    parent: str,
    child: str,
    *,
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    static: bool = True,
    ts: int = 0,
) -> TransformData:
    return TransformData(
        frame_id=parent,
        child_frame_id=child,
        translation=xyz,
        rotation=quat,
        is_static=static,
        timestamp_ns=ts,
    )


def _write_tf_fixture(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "tf.mcap"
    path.write_bytes(create_tf_mcap(**kwargs))
    return path


def test_quaternion_to_euler_identity() -> None:
    roll, pitch, yaw = quaternion_to_euler_rad(0.0, 0.0, 0.0, 1.0)
    assert (roll, pitch, yaw) == (0.0, 0.0, 0.0)


def test_quaternion_to_euler_90deg_yaw() -> None:
    # 90 degree rotation about z-axis: quaternion (0, 0, sin(45), cos(45))
    s = math.sin(math.radians(45))
    c = math.cos(math.radians(45))
    _r, _p, yaw = quaternion_to_euler_rad(0.0, 0.0, s, c)
    assert math.isclose(yaw, math.pi / 2, abs_tol=1e-9)


def test_build_tree_and_find_roots() -> None:
    transforms = {
        ("base_link", "wheel"): _td("base_link", "wheel"),
        ("base_link", "camera"): _td("base_link", "camera"),
        ("camera", "lens"): _td("camera", "lens"),
    }
    tree, roots = build_tree_and_find_roots(transforms)
    assert roots == ["base_link"]
    assert sorted(tree["base_link"]) == ["camera", "wheel"]
    assert tree["camera"] == ["lens"]


def test_detect_multiple_parents() -> None:
    transforms = {
        ("a", "c"): _td("a", "c"),
        ("b", "c"): _td("b", "c"),
        ("a", "d"): _td("a", "d"),
    }
    multi = detect_multiple_parents(transforms)
    assert multi == {"c": {"a", "b"}}


def test_detect_cycles() -> None:
    transforms = {
        ("a", "b"): _td("a", "b"),
        ("b", "c"): _td("b", "c"),
        ("c", "a"): _td("c", "a"),
    }
    cycles = detect_cycles(transforms)
    assert cycles, "expected a cycle"
    nodes = set(cycles[0])
    assert {"a", "b", "c"} <= nodes


def test_detect_cycles_none_when_tree() -> None:
    transforms = {
        ("a", "b"): _td("a", "b"),
        ("b", "c"): _td("b", "c"),
    }
    assert detect_cycles(transforms) == []


def test_render_urdf_emits_links_and_joints() -> None:
    transforms = {
        ("base_link", "wheel"): _td("base_link", "wheel", xyz=(1.0, 2.0, 3.0)),
        ("base_link", "camera"): _td("base_link", "camera"),
    }
    text = render_urdf(transforms, robot_name="bot")
    root = ET.fromstring(text)  # noqa: S314 — string we just rendered

    assert root.tag == "robot"
    assert root.attrib["name"] == "bot"

    link_names = {link.attrib["name"] for link in root.findall("link")}
    assert link_names == {"base_link", "wheel", "camera"}

    joints = root.findall("joint")
    assert len(joints) == 2
    by_name = {j.attrib["name"]: j for j in joints}

    wheel_joint = by_name["base_link__to__wheel"]
    assert wheel_joint.attrib["type"] == "fixed"
    assert wheel_joint.find("parent").attrib["link"] == "base_link"
    assert wheel_joint.find("child").attrib["link"] == "wheel"
    origin = wheel_joint.find("origin")
    assert origin.attrib["xyz"] == "1 2 3"
    assert origin.attrib["rpy"] == "0 0 0"


def test_render_urdf_rejects_quaternion_rotation() -> None:
    with pytest.raises(ValueError, match="rpy"):
        render_urdf({}, robot_name="r", rotation="quat")


def test_render_sdf_emits_model_with_pose_relative_to_parent() -> None:
    transforms = {
        ("base_link", "wheel"): _td("base_link", "wheel", xyz=(1.0, 0.0, 0.0)),
    }
    text = render_sdf(transforms, robot_name="bot")
    sdf = ET.fromstring(text)  # noqa: S314 — string we just rendered
    assert sdf.tag == "sdf"
    model = sdf.find("model")
    assert model is not None
    assert model.attrib["name"] == "bot"

    links = {link.attrib["name"]: link for link in model.findall("link")}
    assert set(links) == {"base_link", "wheel"}

    # Root link has no <pose>; child link does.
    assert links["base_link"].find("pose") is None
    pose = links["wheel"].find("pose")
    assert pose is not None
    assert pose.attrib["relative_to"] == "base_link"
    assert pose.text.split()[0] == "1"

    joint = model.find("joint")
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").text == "base_link"
    assert joint.find("child").text == "wheel"


def test_read_transforms_static_only(tmp_path: Path) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[("base_link", "wheel", (1.0, 0.0, 0.0))],
    )
    transforms = read_transforms(str(mcap))
    assert set(transforms) == {("base_link", "wheel")}
    assert transforms[("base_link", "wheel")].is_static is True
    assert transforms[("base_link", "wheel")].translation == (1.0, 0.0, 0.0)


def test_read_transforms_includes_dynamic_when_requested(tmp_path: Path) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[("base_link", "wheel", (1.0, 0.0, 0.0))],
        dynamic_edges=[("odom", "base_link", (10.0, 0.0, 0.0))],
        dynamic_samples=3,
    )
    static_only = read_transforms(str(mcap))
    assert ("odom", "base_link") not in static_only

    with_dynamic = read_transforms(str(mcap), include_dynamic=True)
    assert ("odom", "base_link") in with_dynamic
    assert with_dynamic[("odom", "base_link")].is_static is False


def test_tf_export_writes_urdf_file(tmp_path: Path) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[
            ("base_link", "wheel", (1.0, 0.0, 0.0)),
            ("base_link", "camera", (0.0, 1.0, 0.0)),
        ],
    )
    out = tmp_path / "robot.urdf"
    rc = tf_export(str(mcap), output=out, format_="urdf", robot_name="bot")
    assert rc == 0
    parsed = ET.parse(out).getroot()  # noqa: S314 — file we just wrote
    assert parsed.tag == "robot"
    link_names = {link.attrib["name"] for link in parsed.findall("link")}
    assert link_names == {"base_link", "wheel", "camera"}


def test_tf_export_writes_json_file(tmp_path: Path) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[("base_link", "wheel", (1.0, 2.0, 3.0))],
    )
    out = tmp_path / "tree.json"
    rc = tf_export(str(mcap), output=out, format_="json")
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload == [
        {
            "parent": "base_link",
            "child": "wheel",
            "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "is_static": True,
            "timestamp_ns": 0,
        }
    ]


def test_tf_export_rejects_multi_parent_without_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[
            ("a", "c", (0.0, 0.0, 0.0)),
            ("b", "c", (0.0, 0.0, 0.0)),
        ],
    )
    rc = tf_export(str(mcap), output=tmp_path / "out.urdf")
    assert rc == 1
    captured = capsys.readouterr()
    assert "multiple parents" in captured.err.lower()


def test_tf_export_allows_multi_parent_with_flag(tmp_path: Path) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[
            ("a", "c", (0.0, 0.0, 0.0)),
            ("b", "c", (0.0, 0.0, 0.0)),
            ("root", "a", (0.0, 0.0, 0.0)),
            ("root", "b", (0.0, 0.0, 0.0)),
        ],
    )
    out = tmp_path / "out.urdf"
    rc = tf_export(str(mcap), output=out, allow_multi_parent=True)
    assert rc == 0
    parsed = ET.parse(out).getroot()  # noqa: S314 — file we just wrote
    joint_names = {j.attrib["name"] for j in parsed.findall("joint")}
    # Only one of (a, c) / (b, c) should survive.
    assert ("a__to__c" in joint_names) ^ ("b__to__c" in joint_names)


def test_tf_export_requires_root_for_multi_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[
            ("root_a", "child_a", (0.0, 0.0, 0.0)),
            ("root_b", "child_b", (0.0, 0.0, 0.0)),
        ],
    )
    rc = tf_export(str(mcap), output=tmp_path / "out.urdf")
    assert rc == 1
    captured = capsys.readouterr()
    assert "root" in captured.err.lower()


def test_tf_export_root_restricts_subtree(tmp_path: Path) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[
            ("root_a", "child_a", (0.0, 0.0, 0.0)),
            ("root_b", "child_b", (0.0, 0.0, 0.0)),
        ],
    )
    out = tmp_path / "out.urdf"
    rc = tf_export(str(mcap), output=out, root="root_a")
    assert rc == 0
    parsed = ET.parse(out).getroot()  # noqa: S314 — file we just wrote
    link_names = {link.attrib["name"] for link in parsed.findall("link")}
    assert link_names == {"root_a", "child_a"}


def test_tf_export_writes_to_stdout_when_no_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mcap = _write_tf_fixture(
        tmp_path,
        static_edges=[("base_link", "wheel", (1.0, 0.0, 0.0))],
    )
    rc = tf_export(str(mcap), format_="json")
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload[0]["child"] == "wheel"


def test_tf_export_no_transforms_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty.mcap"
    buf = io.BytesIO()
    w = McapWriter(buf, chunk_size=512, compression=CompressionType.NONE)
    w.start()
    w.add_schema(schema_id=1, name="x", encoding="json", data=b"{}")
    w.add_channel(channel_id=1, topic="/other", message_encoding="json", schema_id=1)
    w.add_message(channel_id=1, log_time=0, publish_time=0, data=b"{}")
    w.finish()
    empty.write_bytes(buf.getvalue())

    rc = tf_export(str(empty), output=tmp_path / "out.urdf")
    assert rc == 1
    assert "no transforms" in capsys.readouterr().err.lower()
