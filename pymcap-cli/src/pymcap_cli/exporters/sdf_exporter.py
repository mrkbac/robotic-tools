"""SDF emission from a flat (parent, child) → TransformData map.

SDF 1.7 model with `<link>`s placed in their parent frame via `<pose
relative_to="…">`. Fixed joints connect parent → child to preserve the
constraint relationship visible in `tftree`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from pymcap_cli.core.tf_tree import build_tree_and_find_roots, quaternion_to_euler_rad

if TYPE_CHECKING:
    from pymcap_cli.core.tf_tree import TransformData


def _joint_name(parent: str, child: str) -> str:
    return f"{parent}__to__{child}"


def _pose_text(transform: TransformData) -> str:
    tx, ty, tz = transform.translation
    qx, qy, qz, qw = transform.rotation
    roll, pitch, yaw = quaternion_to_euler_rad(qx, qy, qz, qw)
    return f"{tx:g} {ty:g} {tz:g} {roll:g} {pitch:g} {yaw:g}"


def render_sdf(
    transforms: dict[tuple[str, str], TransformData],
    *,
    robot_name: str = "robot",
) -> str:
    """Render an SDF 1.7 model document."""
    sdf = Element("sdf", attrib={"version": "1.7"})
    model = SubElement(sdf, "model", attrib={"name": robot_name})

    _tree, roots = build_tree_and_find_roots(transforms)
    parent_of: dict[str, str] = {}
    frames: set[str] = set()
    for parent, child in transforms:
        parent_of[child] = parent
        frames.add(parent)
        frames.add(child)

    # Root links: no pose (model origin).
    for root in sorted(set(roots) & frames):
        SubElement(model, "link", attrib={"name": root})

    # Child links: pose relative to their parent.
    for child in sorted(frames - set(roots)):
        parent = parent_of[child]
        transform = transforms[(parent, child)]
        link = SubElement(model, "link", attrib={"name": child})
        pose = SubElement(link, "pose", attrib={"relative_to": parent})
        pose.text = _pose_text(transform)

    for parent, child in sorted(transforms.keys()):
        joint = SubElement(
            model,
            "joint",
            attrib={"name": _joint_name(parent, child), "type": "fixed"},
        )
        SubElement(joint, "parent").text = parent
        SubElement(joint, "child").text = child

    indent(sdf, space="  ")
    body = tostring(sdf, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}\n'
