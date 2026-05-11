"""URDF emission from a flat (parent, child) → TransformData map.

URDF is a tree of `<link>` and `<joint>` elements rooted at one link. Static
transforms become `type="fixed"` joints with an `<origin xyz="…" rpy="…"/>`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from pymcap_cli.core.tf_tree import quaternion_to_euler_rad

if TYPE_CHECKING:
    from pymcap_cli.core.tf_tree import TransformData


def _joint_name(parent: str, child: str) -> str:
    return f"{parent}__to__{child}"


def render_urdf(
    transforms: dict[tuple[str, str], TransformData],
    *,
    robot_name: str = "robot",
    rotation: str = "rpy",
) -> str:
    """Render an URDF XML document. `rotation` must be `"rpy"` (URDF has no quaternion form)."""
    if rotation != "rpy":
        msg = f"URDF only supports rpy rotation, got {rotation!r}"
        raise ValueError(msg)

    robot = Element("robot", attrib={"name": robot_name})

    frames: set[str] = set()
    for parent, child in transforms:
        frames.add(parent)
        frames.add(child)

    for frame in sorted(frames):
        SubElement(robot, "link", attrib={"name": frame})

    for (parent, child), transform in sorted(transforms.items()):
        joint = SubElement(
            robot,
            "joint",
            attrib={"name": _joint_name(parent, child), "type": "fixed"},
        )
        tx, ty, tz = transform.translation
        qx, qy, qz, qw = transform.rotation
        roll, pitch, yaw = quaternion_to_euler_rad(qx, qy, qz, qw)
        SubElement(
            joint,
            "origin",
            attrib={
                "xyz": f"{tx:g} {ty:g} {tz:g}",
                "rpy": f"{roll:g} {pitch:g} {yaw:g}",
            },
        )
        SubElement(joint, "parent", attrib={"link": parent})
        SubElement(joint, "child", attrib={"link": child})

    indent(robot, space="  ")
    body = tostring(robot, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}\n'
