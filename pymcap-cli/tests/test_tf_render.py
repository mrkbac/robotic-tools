"""Tests for the shared TF table renderer."""

from __future__ import annotations

import io
from typing import Any

from pymcap_cli.core.tf_tree import TF_STATIC_TOPIC, TF_TOPIC, TfGraph
from pymcap_cli.display.tf_render import build_tf_table
from rich.console import Console


def _capture(table: Any) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, width=200, color_system=None).print(table)
    return buf.getvalue()


def _graph() -> TfGraph:
    graph = TfGraph()
    graph.add(
        static=True,
        stamp_ns=1_000_000_000,
        parent="base_link",
        child="sensor",
        translation=(1.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    graph.add(
        static=False,
        stamp_ns=2_000_000_000,
        parent="base_link",
        child="wheel",
        translation=(0.0, 1.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    return graph


def test_build_tf_table_none_when_empty() -> None:
    assert build_tf_table({}, {}) is None


def test_build_tf_table_full_has_count_and_timestamp() -> None:
    out = _capture(build_tf_table(_graph().transforms, _graph().counts))
    assert "Count" in out
    assert "Timestamp" in out
    assert "roll" in out
    assert "base_link" in out
    assert "sensor" in out
    assert "wheel" in out


def test_build_tf_table_compact_drops_count_and_timestamp() -> None:
    graph = _graph()
    out = _capture(build_tf_table(graph.transforms, graph.counts, compact=True))
    assert "Count" not in out
    assert "Timestamp" not in out
    # Frame tree and the numeric columns still render.
    assert "base_link" in out
    assert "sensor" in out
    assert "tx" in out
    assert "yaw" in out


def test_build_tf_table_topics_constants_are_standard() -> None:
    assert TF_TOPIC == "/tf"
    assert TF_STATIC_TOPIC == "/tf_static"
