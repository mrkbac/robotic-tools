"""Validation and diagnostic findings for TF graphs."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from pymcap_cli.core.tf_tree import Edge, TfGraph, TransformData, build_tree_and_find_roots

_QUATERNION_NORM_EPSILON = 1e-3


class TfSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class TfFindingCode(str, Enum):
    MULTIPLE_PARENTS = "multiple_parents"
    CYCLE = "cycle"
    NON_FINITE_VALUE = "non_finite_value"
    EMPTY_FRAME_ID = "empty_frame_id"
    NON_UNIT_QUATERNION = "non_unit_quaternion"
    TF_STATIC_TF_OVERLAP = "tf_static_tf_overlap"
    SLASH_INCONSISTENCY = "slash_inconsistency"
    COMPONENTS = "components"


@dataclass(frozen=True, slots=True)
class TfFinding:
    severity: TfSeverity
    code: TfFindingCode
    message: str
    frames: tuple[str, ...] = field(default_factory=tuple)


def detect_multiple_parents(
    transforms: dict[Edge, TransformData],
) -> dict[str, set[str]]:
    """Children that appear under more than one parent."""
    child_to_parents: dict[str, set[str]] = defaultdict(set)
    for parent, child in transforms:
        child_to_parents[child].add(parent)
    return {child: parents for child, parents in child_to_parents.items() if len(parents) > 1}


def detect_cycles(
    transforms: dict[Edge, TransformData],
) -> list[list[str]]:
    """Return frame cycles in the directed parent -> child graph."""
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

    for node in sorted(tree_dict):
        if color.get(node) is None:
            visit(node)
    return cycles


def detect_disconnected_components(
    transforms: dict[Edge, TransformData],
) -> list[set[str]]:
    """Connected components in the undirected edge graph."""
    parent_of: dict[str, str] = {}

    def find(node: str) -> str:
        while parent_of.get(node, node) != node:
            parent_of[node] = parent_of.get(parent_of[node], parent_of[node])
            node = parent_of[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent_of[right_root] = left_root

    for parent, child in transforms:
        parent_of.setdefault(parent, parent)
        parent_of.setdefault(child, child)
        union(parent, child)

    groups: dict[str, set[str]] = defaultdict(set)
    for node in parent_of:
        groups[find(node)].add(node)
    return sorted(groups.values(), key=lambda group: (-len(group), sorted(group)[0]))


def detect_invalid_quaternions(
    transforms: dict[Edge, TransformData],
    *,
    epsilon: float = _QUATERNION_NORM_EPSILON,
) -> list[tuple[Edge, float]]:
    """Edges whose quaternion is not unit-length."""
    bad: list[tuple[Edge, float]] = []
    for edge, transform in transforms.items():
        norm = _quaternion_norm(transform.rotation)
        if math.isfinite(norm) and abs(norm - 1.0) > epsilon:
            bad.append((edge, norm))
    return bad


def detect_non_finite_values(
    transforms: dict[Edge, TransformData],
) -> list[Edge]:
    """Edges with NaN or Inf anywhere in translation or rotation."""
    return [
        edge
        for edge, transform in transforms.items()
        if not all(math.isfinite(value) for value in (*transform.translation, *transform.rotation))
    ]


def detect_empty_frame_ids(
    transforms: dict[Edge, TransformData],
) -> list[Edge]:
    """Edges where either frame_id or child_frame_id is empty."""
    return [edge for edge in transforms if edge[0] == "" or edge[1] == ""]


def detect_slash_inconsistency(
    transforms: dict[Edge, TransformData],
) -> dict[str, set[str]]:
    """Frames whose name appears in more than one form, for example 'foo' and '/foo'."""
    variants: dict[str, set[str]] = defaultdict(set)
    for parent, child in transforms:
        for name in (parent, child):
            variants[name.lstrip("/")].add(name)
    return {key: names for key, names in variants.items() if len(names) > 1}


def collect_tf_findings(graph: TfGraph) -> list[TfFinding]:
    """Run TF graph detectors and return findings sorted by severity."""
    transforms = graph.transforms
    findings: list[TfFinding] = []

    findings.extend(
        TfFinding(
            severity=TfSeverity.ERROR,
            code=TfFindingCode.MULTIPLE_PARENTS,
            message=f"Frame '{child}' has multiple parents: {', '.join(sorted(parents))}",
            frames=(child, *sorted(parents)),
        )
        for child, parents in sorted(detect_multiple_parents(transforms).items())
    )

    findings.extend(
        TfFinding(
            severity=TfSeverity.ERROR,
            code=TfFindingCode.CYCLE,
            message="Cycle: " + " -> ".join(cycle),
            frames=tuple(cycle),
        )
        for cycle in detect_cycles(transforms)
    )

    findings.extend(
        TfFinding(
            severity=TfSeverity.ERROR,
            code=TfFindingCode.NON_FINITE_VALUE,
            message=f"Edge {edge[0]} -> {edge[1]} has NaN or Inf in translation/rotation",
            frames=edge,
        )
        for edge in detect_non_finite_values(transforms)
    )

    findings.extend(
        TfFinding(
            severity=TfSeverity.ERROR,
            code=TfFindingCode.EMPTY_FRAME_ID,
            message=f"Edge with empty frame id: parent='{edge[0]}' child='{edge[1]}'",
            frames=edge,
        )
        for edge in detect_empty_frame_ids(transforms)
    )

    findings.extend(
        TfFinding(
            severity=TfSeverity.WARNING,
            code=TfFindingCode.NON_UNIT_QUATERNION,
            message=f"Edge {edge[0]} -> {edge[1]} has non-unit quaternion (norm={norm:.6f})",
            frames=edge,
        )
        for edge, norm in detect_invalid_quaternions(transforms)
    )

    findings.extend(
        TfFinding(
            severity=TfSeverity.WARNING,
            code=TfFindingCode.TF_STATIC_TF_OVERLAP,
            message=f"Edge {edge[0]} -> {edge[1]} was published on both /tf and /tf_static",
            frames=edge,
        )
        for edge, topics in sorted(graph.topics_by_edge.items())
        if "/tf_static" in topics and "/tf" in topics
    )

    findings.extend(
        TfFinding(
            severity=TfSeverity.WARNING,
            code=TfFindingCode.SLASH_INCONSISTENCY,
            message=(
                f"Frame '{normalized}' appears as "
                + ", ".join(f"'{variant}'" for variant in sorted(raw_variants))
            ),
            frames=tuple(sorted(raw_variants)),
        )
        for normalized, raw_variants in sorted(detect_slash_inconsistency(transforms).items())
    )

    components = detect_disconnected_components(transforms)
    if len(components) > 1:
        sizes = ", ".join(str(len(component)) for component in components)
        findings.append(
            TfFinding(
                severity=TfSeverity.INFO,
                code=TfFindingCode.COMPONENTS,
                message=f"TF graph has {len(components)} disconnected components (sizes: {sizes})",
            )
        )

    severity_rank = {TfSeverity.ERROR: 0, TfSeverity.WARNING: 1, TfSeverity.INFO: 2}
    findings.sort(key=lambda finding: (severity_rank[finding.severity], finding.code.value))
    return findings


def has_error_findings(findings: list[TfFinding]) -> bool:
    return any(finding.severity is TfSeverity.ERROR for finding in findings)


def _quaternion_norm(quaternion: tuple[float, float, float, float]) -> float:
    qx, qy, qz, qw = quaternion
    return math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
