"""Unit tests for TF graph findings."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pymcap_cli.cmd.tftree_cmd import tftree
from pymcap_cli.core.tf_findings import (
    TfFinding,
    TfFindingCode,
    TfSeverity,
    collect_tf_findings,
    detect_disconnected_components,
    detect_empty_frame_ids,
    detect_invalid_quaternions,
    detect_non_finite_values,
    detect_slash_inconsistency,
)
from pymcap_cli.core.tf_tree import TfGraph, TransformData

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


def _codes(findings: list[TfFinding], code: TfFindingCode) -> list[TfFinding]:
    return [finding for finding in findings if finding.code is code]


def _graph(transforms: dict[tuple[str, str], TransformData]) -> TfGraph:
    return TfGraph(transforms=transforms)


def test_detect_disconnected_components_single() -> None:
    transforms = {("a", "b"): _td("a", "b"), ("b", "c"): _td("b", "c")}
    components = detect_disconnected_components(transforms)
    assert len(components) == 1
    assert components[0] == {"a", "b", "c"}


def test_detect_disconnected_components_two() -> None:
    transforms = {
        ("a", "b"): _td("a", "b"),
        ("x", "y"): _td("x", "y"),
    }
    components = detect_disconnected_components(transforms)
    assert len(components) == 2
    assert {frozenset(component) for component in components} == {
        frozenset({"a", "b"}),
        frozenset({"x", "y"}),
    }


def test_detect_invalid_quaternions_unit_passes() -> None:
    transforms = {("a", "b"): _td("a", "b", quat=(0.0, 0.0, 0.0, 1.0))}
    assert detect_invalid_quaternions(transforms) == []


def test_detect_invalid_quaternions_nonunit_flagged() -> None:
    transforms = {("a", "b"): _td("a", "b", quat=(0.5, 0.0, 0.0, 1.0))}
    bad = detect_invalid_quaternions(transforms)
    assert len(bad) == 1
    edge, norm = bad[0]
    assert edge == ("a", "b")
    assert norm > 1.0


def test_detect_non_finite_values_nan_flagged() -> None:
    transforms = {("a", "b"): _td("a", "b", xyz=(math.nan, 0.0, 0.0))}
    assert detect_non_finite_values(transforms) == [("a", "b")]


def test_detect_non_finite_values_inf_flagged() -> None:
    transforms = {("a", "b"): _td("a", "b", quat=(math.inf, 0.0, 0.0, 1.0))}
    assert detect_non_finite_values(transforms) == [("a", "b")]


def test_detect_non_finite_values_clean() -> None:
    transforms = {("a", "b"): _td("a", "b")}
    assert detect_non_finite_values(transforms) == []


def test_detect_empty_frame_ids() -> None:
    transforms = {
        ("a", "b"): _td("a", "b"),
        ("", "c"): _td("", "c"),
        ("d", ""): _td("d", ""),
    }
    assert set(detect_empty_frame_ids(transforms)) == {("", "c"), ("d", "")}


def test_detect_slash_inconsistency() -> None:
    transforms = {
        ("base", "foo"): _td("base", "foo"),
        ("/base", "bar"): _td("/base", "bar"),
    }
    variants = detect_slash_inconsistency(transforms)
    assert variants == {"base": {"base", "/base"}}


def test_detect_slash_inconsistency_consistent() -> None:
    transforms = {("base", "foo"): _td("base", "foo")}
    assert detect_slash_inconsistency(transforms) == {}


def test_collect_tf_findings_clean() -> None:
    transforms = {
        ("base", "child"): _td("base", "child", xyz=(1.0, 0.0, 0.0)),
    }
    findings = collect_tf_findings(_graph(transforms))
    assert findings == []


def test_collect_tf_findings_multiple_parents_is_error() -> None:
    transforms = {
        ("a", "c"): _td("a", "c"),
        ("b", "c"): _td("b", "c"),
    }
    findings = collect_tf_findings(_graph(transforms))
    multi = _codes(findings, TfFindingCode.MULTIPLE_PARENTS)
    assert len(multi) == 1
    assert multi[0].severity is TfSeverity.ERROR
    assert "c" in multi[0].message


def test_collect_tf_findings_cycle_is_error() -> None:
    transforms = {
        ("a", "b"): _td("a", "b"),
        ("b", "a"): _td("b", "a"),
    }
    findings = collect_tf_findings(_graph(transforms))
    cycles = _codes(findings, TfFindingCode.CYCLE)
    assert len(cycles) >= 1
    assert all(finding.severity is TfSeverity.ERROR for finding in cycles)


def test_collect_tf_findings_nonunit_quat_is_warning() -> None:
    transforms = {
        ("a", "b"): _td("a", "b", xyz=(1.0, 0.0, 0.0), quat=(2.0, 0.0, 0.0, 0.0)),
    }
    findings = collect_tf_findings(_graph(transforms))
    bad = _codes(findings, TfFindingCode.NON_UNIT_QUATERNION)
    assert len(bad) == 1
    assert bad[0].severity is TfSeverity.WARNING


def test_collect_tf_findings_tf_static_overlap_is_warning() -> None:
    graph = TfGraph(
        transforms={("a", "b"): _td("a", "b", xyz=(1.0, 0.0, 0.0), static=True)},
        topics_by_edge={("a", "b"): {"/tf_static", "/tf"}},
    )
    findings = collect_tf_findings(graph)
    overlap = _codes(findings, TfFindingCode.TF_STATIC_TF_OVERLAP)
    assert len(overlap) == 1
    assert overlap[0].severity is TfSeverity.WARNING


def test_collect_tf_findings_components_is_info() -> None:
    transforms = {
        ("a", "b"): _td("a", "b", xyz=(1.0, 0.0, 0.0)),
        ("x", "y"): _td("x", "y", xyz=(1.0, 0.0, 0.0)),
    }
    findings = collect_tf_findings(_graph(transforms))
    assert any(
        finding.code is TfFindingCode.COMPONENTS and finding.severity is TfSeverity.INFO
        for finding in findings
    )


def test_collect_tf_findings_sorted_errors_first() -> None:
    transforms = {
        ("a", "c"): _td("a", "c", xyz=(1.0, 0.0, 0.0)),
        ("b", "c"): _td("b", "c", xyz=(1.0, 0.0, 0.0)),
        ("c", "d"): _td("c", "d", xyz=(1.0, 0.0, 0.0), quat=(2.0, 0.0, 0.0, 0.0)),
    }
    findings = collect_tf_findings(_graph(transforms))
    severities = [finding.severity for finding in findings]
    rank = {TfSeverity.ERROR: 0, TfSeverity.WARNING: 1, TfSeverity.INFO: 2}
    assert severities == sorted(severities, key=lambda severity: rank[severity])


def test_tftree_command_cycle_returns_nonzero(tmp_path: Path, capsys) -> None:
    bag = tmp_path / "cycle.mcap"
    bag.write_bytes(
        create_tf_mcap(
            static_edges=[
                ("a", "b", (1.0, 0.0, 0.0)),
                ("b", "a", (1.0, 0.0, 0.0)),
            ],
        )
    )
    rc = tftree(str(bag))
    out = capsys.readouterr().out
    assert rc == 1
    assert "cycle" in out.lower()


def test_tftree_command_clean_returns_zero(tmp_path: Path, capsys) -> None:
    bag = tmp_path / "clean.mcap"
    bag.write_bytes(
        create_tf_mcap(
            static_edges=[
                ("base", "child", (1.0, 0.0, 0.0)),
            ],
        )
    )
    rc = tftree(str(bag))
    capsys.readouterr()
    assert rc == 0
