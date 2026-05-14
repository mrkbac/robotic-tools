"""`pymcap-cli tf-export` — TF tree → URDF / SDF / JSON."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from cyclopts import Group, Parameter

from pymcap_cli.core.tf_findings import detect_cycles, detect_multiple_parents
from pymcap_cli.core.tf_tree import (
    TransformData,
    build_tree_and_find_roots,
    read_transforms,
)
from pymcap_cli.exporters.sdf_exporter import render_sdf
from pymcap_cli.exporters.urdf_exporter import render_urdf
from pymcap_cli.log_setup import ERR
from pymcap_cli.utils import parse_time_arg

logger = logging.getLogger(__name__)

OUTPUT_GROUP = Group("Output")
SELECTION_GROUP = Group("Selection")

Format = Literal["urdf", "sdf", "json"]


class _Vec3Dict(TypedDict):
    x: float
    y: float
    z: float


class _QuatDict(TypedDict):
    x: float
    y: float
    z: float
    w: float


class _TransformJson(TypedDict):
    parent: str
    child: str
    translation: _Vec3Dict
    rotation: _QuatDict
    is_static: bool
    timestamp_ns: int


def _transform_to_dict(transform: TransformData) -> _TransformJson:
    tx, ty, tz = transform.translation
    qx, qy, qz, qw = transform.rotation
    return {
        "parent": transform.frame_id,
        "child": transform.child_frame_id,
        "translation": {"x": tx, "y": ty, "z": tz},
        "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
        "is_static": transform.is_static,
        "timestamp_ns": transform.timestamp_ns,
    }


def _render_json(transforms: dict[tuple[str, str], TransformData]) -> str:
    payload = [_transform_to_dict(t) for _key, t in sorted(transforms.items())]
    return json.dumps(payload, indent=2) + "\n"


def _restrict_to_root(
    transforms: dict[tuple[str, str], TransformData], root: str
) -> dict[tuple[str, str], TransformData]:
    """Keep only transforms reachable from `root` via parent → child edges."""
    tree, _roots = build_tree_and_find_roots(transforms)
    reachable: set[str] = set()
    stack = [root]
    while stack:
        frame = stack.pop()
        if frame in reachable:
            continue
        reachable.add(frame)
        stack.extend(tree.get(frame, []))
    return {key: t for key, t in transforms.items() if key[0] in reachable and key[1] in reachable}


def _resolve_multi_parent(
    transforms: dict[tuple[str, str], TransformData],
    multi: dict[str, set[str]],
) -> dict[tuple[str, str], TransformData]:
    """When `--allow-multi-parent` is set, keep the most recently timestamped parent per child."""
    keep_parent: dict[str, str] = {}
    for child, parents in multi.items():
        best_parent = max(parents, key=lambda p: transforms[(p, child)].timestamp_ns)
        keep_parent[child] = best_parent

    result: dict[tuple[str, str], TransformData] = {}
    for (parent, child), transform in transforms.items():
        if child in keep_parent and keep_parent[child] != parent:
            continue
        result[(parent, child)] = transform
    return result


def tf_export(
    file: str,
    *,
    output: Annotated[
        Path | None,
        Parameter(
            name=["--output", "-o"],
            group=OUTPUT_GROUP,
            help="Output file. Defaults to stdout.",
        ),
    ] = None,
    format_: Annotated[
        Format,
        Parameter(
            name=["--format", "-f"],
            group=OUTPUT_GROUP,
            help="Output format.",
        ),
    ] = "urdf",
    robot_name: Annotated[
        str | None,
        Parameter(
            name=["--robot-name"],
            group=OUTPUT_GROUP,
            help="Name for the <robot>/<model> element. Defaults to the input filename stem.",
        ),
    ] = None,
    root: Annotated[
        str | None,
        Parameter(
            name=["--root"],
            group=SELECTION_GROUP,
            help="Root frame. Required when the tree has multiple roots.",
        ),
    ] = None,
    include_dynamic_at: Annotated[
        str | None,
        Parameter(
            name=["--include-dynamic-at"],
            group=SELECTION_GROUP,
            help="Include /tf at a snapshot time (RFC3339 or nanoseconds).",
        ),
    ] = None,
    allow_multi_parent: Annotated[
        bool,
        Parameter(
            name=["--allow-multi-parent"],
            group=SELECTION_GROUP,
            help="Resolve multi-parent frames by keeping the most recent parent.",
        ),
    ] = False,
) -> int:
    """Export the TF tree from an MCAP file as URDF, SDF, or JSON.

    Parameters
    ----------
    file
        Path to MCAP file (local file or HTTP/HTTPS URL).
    """
    snapshot_ns: int | None = None
    include_dynamic = False
    if include_dynamic_at is not None:
        try:
            snapshot_ns = parse_time_arg(include_dynamic_at)
        except ValueError as exc:
            ERR.print(f"[red]Invalid --include-dynamic-at value:[/red] {exc}")
            return 1
        include_dynamic = True

    try:
        transforms = read_transforms(
            file,
            include_dynamic=include_dynamic,
            snapshot_ns=snapshot_ns,
        )
    except (OSError, ValueError, RuntimeError):
        logger.exception("Error reading MCAP file")
        return 1

    if not transforms:
        ERR.print("[red]No transforms found on /tf_static or /tf.[/red]")
        return 1

    multi = detect_multiple_parents(transforms)
    if multi and not allow_multi_parent:
        ERR.print(
            "[red]Tree violation:[/red] some frames have multiple parents. "
            "Pass --allow-multi-parent to keep the most recent parent per child."
        )
        for child, parents in sorted(multi.items()):
            parents_str = ", ".join(sorted(parents))
            ERR.print(f"  {child} <- {{{parents_str}}}")
        return 1
    if multi:
        transforms = _resolve_multi_parent(transforms, multi)

    cycles = detect_cycles(transforms)
    if cycles:
        ERR.print("[red]Cycle detected in TF tree — cannot export:[/red]")
        for cycle in cycles:
            ERR.print("  " + " -> ".join(cycle))
        return 1

    _tree, roots = build_tree_and_find_roots(transforms)
    if root is not None:
        if root not in {parent for parent, _child in transforms} | {
            child for _parent, child in transforms
        }:
            ERR.print(f"[red]Root frame '{root}' not present in the TF tree.[/red]")
            return 1
        transforms = _restrict_to_root(transforms, root)
        if not transforms:
            ERR.print(f"[red]Root frame '{root}' has no descendants.[/red]")
            return 1
    elif len(roots) > 1:
        ERR.print(
            "[red]Multiple root frames found — pass --root to pick one:[/red] " + ", ".join(roots)
        )
        return 1

    name = robot_name if robot_name is not None else Path(file).stem or "robot"

    if format_ == "urdf":
        text = render_urdf(transforms, robot_name=name)
    elif format_ == "sdf":
        text = render_sdf(transforms, robot_name=name)
    else:
        text = _render_json(transforms)

    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        logger.info("Wrote %s", output)

    return 0
