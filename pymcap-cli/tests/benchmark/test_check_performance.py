"""Performance coverage for recording contract checks.

Run with:
  uv run pytest pymcap-cli/tests/benchmark/test_check_performance.py --benchmark-only -q
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.check import check_mcap, load_check_spec, parse_check_spec
from pymcap_cli.cmd.bridge.check import _graph_results
from robo_ws_bridge import ConnectionGraph

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.benchmark(group="check_nuscenes")
def test_benchmark_check_nuscenes_timing_and_values(benchmark, nuscenes_mcap: Path) -> None:
    if not nuscenes_mcap.exists():
        pytest.skip(f"Benchmark fixture missing: {nuscenes_mcap}")
    spec_path = nuscenes_mcap.parents[2] / "pymcap-cli/examples/check/nuscenes.yaml"
    spec = load_check_spec(spec_path)

    report = benchmark.pedantic(
        lambda: check_mcap(str(nuscenes_mcap), spec, num_workers=4),
        rounds=5,
        iterations=1,
    )

    assert report.error_count == 0


@pytest.mark.benchmark(group="bridge_check_graph")
def test_benchmark_bridge_check_large_connection_graph(benchmark) -> None:
    spec = parse_check_spec(
        """\
version: 1
topics:
  front_camera:
    topic: /CAM_FRONT/.*
    live:
      publishers:
        min: 1
        max: 1
"""
    )
    graph = ConnectionGraph(
        published_topics=tuple(
            {"name": f"/CAM_FRONT/stream_{index}", "publisherIds": ["/camera_driver"]}
            for index in range(1_000)
        ),
        subscribed_topics=(),
        advertised_services=(),
    )

    results = benchmark(_graph_results, spec, {0: {}}, graph)

    assert len(results) == 1_000
