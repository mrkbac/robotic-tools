"""`pymcap-cli tf-get` - lookup a transform between two TF frames."""

from __future__ import annotations

import logging
import math
from typing import Annotated

from cyclopts import Group, Parameter
from rich.console import Console
from rich.table import Table

from pymcap_cli.core.tf_findings import (
    TfFinding,
    TfSeverity,
    collect_tf_findings,
    has_error_findings,
)
from pymcap_cli.core.tf_tree import (
    TfGraph,
    TfLookupError,
    TfLookupResult,
    quaternion_to_euler_rad,
    read_tf_graph,
)
from pymcap_cli.log_setup import ERR
from pymcap_cli.utils import parse_time_arg

logger = logging.getLogger(__name__)
console = Console()

SELECTION_GROUP = Group("Selection")


def tf_get(
    file: str,
    target: str,
    source: str,
    *,
    at: Annotated[
        str | None,
        Parameter(
            name=["--at"],
            group=SELECTION_GROUP,
            help=(
                "Resolve dynamic /tf samples at this time (RFC3339 or nanoseconds). "
                "Defaults to per-edge latest, like ROS 2 lookup_transform with Time(0)."
            ),
        ),
    ] = None,
) -> int:
    """Lookup TARGET_T_SOURCE in a TF tree from an MCAP file.

    The returned transform maps coordinates expressed in SOURCE into
    coordinates expressed in TARGET.
    """
    time_ns: int | None = None
    if at is not None:
        try:
            time_ns = parse_time_arg(at)
        except ValueError as exc:
            ERR.print(f"[red]Invalid --at value:[/red] {exc}")
            return 1

    try:
        graph = read_tf_graph(file, include_dynamic=True, keep_series=time_ns is not None)
    except (OSError, ValueError, RuntimeError):
        logger.exception("Error reading MCAP file")
        return 1

    if not graph.transforms:
        ERR.print("[red]No transforms found on /tf_static or /tf.[/red]")
        return 1

    findings = collect_tf_findings(graph)
    errors = [finding for finding in findings if finding.severity is TfSeverity.ERROR]
    if errors:
        ERR.print("[red]TF graph errors prevent lookup:[/red]")
        for finding in errors:
            ERR.print(f"  {finding.code.value}: {finding.message}")
        return 1

    try:
        if time_ns is None:
            result = graph.lookup(target=target, source=source)
        else:
            result = graph.lookup_at(target=target, source=source, time_ns=time_ns)
    except TfLookupError as exc:
        ERR.print(f"[red]{exc}[/red]")
        _print_lookup_context(graph, target=target, source=source, findings=findings)
        return 1

    console.print(_result_table(result))
    console.print()
    console.print(_path_table(result, time_ns=time_ns))
    return 0


def _result_table(result: TfLookupResult) -> Table:
    tx, ty, tz = result.transform.translation
    qx, qy, qz, qw = result.transform.rotation
    roll, pitch, yaw = quaternion_to_euler_rad(qx, qy, qz, qw)

    table = Table(title=f"{result.target} <- {result.source}", show_header=False)
    table.add_column("field", style="bold cyan", no_wrap=True)
    table.add_column("value")
    table.add_row(
        "translation",
        f"x={_fmt(tx)}  y={_fmt(ty)}  z={_fmt(tz)}",
    )
    table.add_row(
        "rotation rpy",
        (
            f"roll={_fmt(math.degrees(roll))} deg  "
            f"pitch={_fmt(math.degrees(pitch))} deg  "
            f"yaw={_fmt(math.degrees(yaw))} deg"
        ),
    )
    return table


def _path_table(result: TfLookupResult, *, time_ns: int | None = None) -> Table:
    stamp_header = "time_ns (interp)" if time_ns is not None else "timestamp_ns"
    table = Table(title="Traversal Path")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("from", no_wrap=True)
    table.add_column("to", no_wrap=True)
    table.add_column("stored edge", no_wrap=True)
    table.add_column("use", no_wrap=True)
    table.add_column("topic", no_wrap=True)
    table.add_column(stamp_header, justify="right", no_wrap=True)

    if not result.path:
        table.add_row("0", result.source, result.target, "identity", "identity", "-", "0")
        return table

    for index, step in enumerate(result.path, start=1):
        parent, child = step.edge
        # In --at mode the displayed time is the user's requested time (the
        # value the bracket samples were interpolated *to*), not any single
        # stored sample.
        stamp = (
            time_ns
            if time_ns is not None and not step.transform.is_static
            else step.transform.timestamp_ns
        )
        table.add_row(
            str(index),
            step.from_frame,
            step.to_frame,
            f"{parent} -> {child}",
            "inverse" if step.is_inverted else "direct",
            "/tf_static" if step.transform.is_static else "/tf",
            str(stamp),
        )
    return table


def _print_lookup_context(
    graph: TfGraph,
    *,
    target: str,
    source: str,
    findings: list[TfFinding],
) -> None:
    if has_error_findings(findings):
        return

    target_component = graph.component_for(target)
    source_component = graph.component_for(source)
    if target_component:
        ERR.print(f"  target component: {', '.join(sorted(target_component))}")
    if source_component and source_component != target_component:
        ERR.print(f"  source component: {', '.join(sorted(source_component))}")

    for finding in findings:
        if finding.severity is TfSeverity.INFO:
            ERR.print(f"  {finding.code.value}: {finding.message}")


def _fmt(value: float) -> str:
    return f"{value:.9g}"
