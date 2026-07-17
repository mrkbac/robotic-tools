from __future__ import annotations

import io
from typing import TYPE_CHECKING

from pymcap_cli.check import CheckReport, CheckResult, parse_check_spec
from pymcap_cli.cmd.bridge import check as check_module
from pymcap_cli.cmd.bridge.check import _graph_results, _LiveEvaluator, check
from rich.console import Console
from robo_ws_bridge import ConnectionGraph

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

NS = 1_000_000_000


def _channel(topic: str = "/imu"):
    return {
        "id": 1,
        "topic": topic,
        "encoding": "json",
        "schemaName": "sensor_msgs/msg/Imu",
        "schema": "{}",
        "schemaEncoding": "jsonschema",
    }


def _spec(text: str):
    return parse_check_spec(text, source="live.yaml")


def _result(report: CheckReport, name: str) -> CheckResult:
    return next(result for result in report.results if result.name == name)


def test_graph_results_reuse_topic_selector_and_check_endpoint_nodes() -> None:
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    live:
      publishers:
        min: 1
        max: 1
        node: /imu_driver
      subscribers:
        min: 1
live:
  nodes:
    localization:
      node: /localization
"""
    )
    graph = ConnectionGraph(
        published_topics=({"name": "/imu", "publisherIds": ["/imu_driver"]},),
        subscribed_topics=({"name": "/imu", "subscriberIds": ["/recorder"]},),
        advertised_services=({"name": "/localization/reset", "providerIds": ["/localization"]},),
    )

    results = _graph_results(spec, {0: {}}, graph)

    assert next(result for result in results if result.name.endswith("/publishers")).level == 0
    assert next(result for result in results if result.name.endswith("/subscribers")).level == 0
    assert next(result for result in results if result.name == "localization/node").level == 0


def test_graph_results_report_wrong_publisher_and_missing_capability() -> None:
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    live:
      publishers:
        node: /imu_driver
"""
    )
    graph = ConnectionGraph(
        published_topics=({"name": "/imu", "publisherIds": ["/wrong_driver"]},),
        subscribed_topics=(),
        advertised_services=(),
    )

    results = _graph_results(spec, {0: {}}, graph)

    assert results[0].level == 2
    assert results[0].values["nodes"] == "/wrong_driver"
    assert _graph_results(spec, {0: {}}, None)[0].name == "live/connection_graph"


def test_live_evaluator_checks_advertisement_timing_and_values() -> None:
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    schema:
      name: sensor_msgs/msg/Imu
      encoding: jsonschema
    message_encoding: json
    frequency:
      min: 2
      max: 2
      window: 1s
    timeout: 600ms
    values:
      - path: .acceleration
        max: 3
"""
    )
    evaluator = _LiveEvaluator(spec, 0, NS)
    channel = _channel()
    evaluator.add_channel(channel)
    evaluator.observe(channel, b'{"acceleration": 1}', NS // 4)
    evaluator.observe(channel, b'{"acceleration": 2}', 3 * NS // 4)

    report = evaluator.report("ws://bridge:8765", None)

    assert _result(report, "imu/expected").level == 0
    assert _result(report, "imu:/imu/schema").level == 0
    assert _result(report, "imu:/imu/frequency").level == 0
    assert _result(report, "imu:/imu/timeout").level == 0
    assert _result(report, "imu:/imu/value[0]").level == 0


def test_live_evaluator_supports_cross_message_assertions() -> None:
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - '.acceleration.@@max{<=3}'
"""
    )
    evaluator = _LiveEvaluator(spec, 0, NS)
    channel = _channel()
    evaluator.add_channel(channel)
    evaluator.observe(channel, b'{"acceleration": 1}', NS // 4)
    evaluator.observe(channel, b'{"acceleration": 3}', 3 * NS // 4)

    report = evaluator.report("ws://bridge:8765", None)

    assert _result(report, "imu:/imu/value[0]").level == 0


def test_live_evaluator_decodes_once_for_overlapping_value_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(
        """
version: 1
topics:
  acceleration:
    topic: /imu
    values:
      - path: .acceleration
        max: 3
  state:
    topic: /imu
    values:
      - path: .state
        equals: ok
"""
    )
    evaluator = _LiveEvaluator(spec, 0, NS)
    channel = _channel()
    evaluator.add_channel(channel)
    decode_count = 0
    original = _LiveEvaluator._decode

    def count_decode(self, *args):
        nonlocal decode_count
        decode_count += 1
        return original(self, *args)

    monkeypatch.setattr(_LiveEvaluator, "_decode", count_decode)

    evaluator.observe(channel, b'{"acceleration": 1, "state": "ok"}', NS // 2)

    assert decode_count == 1


def test_bridge_check_command_uses_shared_report_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "recording.yaml"
    spec_path.write_text("version: 1\ntopics: {}\n")
    output = io.StringIO()
    monkeypatch.setattr(
        "pymcap_cli.cmd.check_cmd.console",
        Console(file=output, force_terminal=False, color_system=None, width=160),
    )
    monkeypatch.setattr(
        check_module,
        "collect_bridge_check",
        lambda *_args, **_kwargs: CheckReport(
            path="ws://localhost:8765",
            results=[CheckResult(2, "imu/expected", "expected topic is missing")],
        ),
    )

    assert check("localhost", spec=spec_path, duration=0.1) == 1
    assert "imu/expected" in output.getvalue()
