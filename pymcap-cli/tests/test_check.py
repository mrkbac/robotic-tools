from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING

import pytest
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli import check as check_module
from pymcap_cli.check import (
    ERROR,
    OK,
    CheckResult,
    CheckSpecError,
    check_mcap,
    load_check_spec,
    parse_check_spec,
)
from pymcap_cli.cmd import check_cmd
from rich.console import Console
from small_mcap import CompressionType, McapWriter

if TYPE_CHECKING:
    from pathlib import Path

NS = 1_000_000_000

_IMU_SCHEMA = b"""\
geometry_msgs/Vector3 linear_acceleration

================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z"""


def _write_json_mcap(
    path: Path,
    events: list[tuple[str, int, str]],
) -> None:
    topics = sorted({topic for topic, _, _ in events})
    channel_ids = {topic: index for index, topic in enumerate(topics, start=1)}
    with path.open("wb") as stream:
        writer = McapWriter(
            stream,
            chunk_size=256,
            compression=CompressionType.NONE,
        )
        writer.start()
        writer.add_schema(1, "example/msg/Sample", "jsonschema", b"{}")
        for topic, channel_id in channel_ids.items():
            writer.add_channel(channel_id, topic, "json", 1)
        for topic, timestamp_ns, payload in sorted(events, key=lambda event: event[1]):
            writer.add_message(
                channel_ids[topic],
                timestamp_ns,
                payload.encode(),
                timestamp_ns,
            )
        writer.finish()


def _write_ros2_imu_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(
            stream,
            chunk_size=256,
            compression=CompressionType.NONE,
            encoder_factory=ROS2EncoderFactory(),
        )
        writer.start()
        writer.add_schema(1, "sensor_msgs/msg/Imu", "ros2msg", _IMU_SCHEMA)
        writer.add_channel(1, "/imu", "cdr", 1)
        writer.add_message_encode(
            channel_id=1,
            log_time=0,
            publish_time=0,
            data={"linear_acceleration": {"x": 1.0, "y": 2.0, "z": 2.0}},
        )
        writer.finish()


def _write_multi_channel_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(stream, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(1, "sensor_msgs/msg/Imu", "ros2msg", _IMU_SCHEMA)
        writer.add_schema(2, "example/msg/Sample", "jsonschema", b"{}")
        writer.add_channel(1, "/imu", "cdr", 1)
        writer.add_channel(2, "/imu", "json", 2)
        writer.add_message(1, 0, b"", 0)
        writer.add_message(2, 1, b"{}", 1)
        writer.finish()


def _spec(text: str):
    return parse_check_spec(text, source="test.yaml")


def _result(report, name: str):
    return next(result for result in report.results if result.name == name)


def test_parse_check_spec_accepts_ros_shaped_contract() -> None:
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    schema:
      name: sensor_msgs/msg/Imu
      encoding: ros2msg
    message_encoding: cdr
    frequency:
      min: 95
      max: 105
      tolerance: 0.05
      window: 1s
    timeout: 50ms
    values:
      - path: .temperature
        min: -40
        max: 85
"""
    )

    rule = spec.topics[0]
    assert rule.name == "imu"
    assert rule.frequency is not None
    assert rule.frequency.window_ns == NS
    assert rule.timeout_ns == 50_000_000
    assert rule.values[0].path_source == ".temperature"


def test_parse_check_spec_accepts_messagepath_assertion_shorthand() -> None:
    spec = _spec(
        """
version: 1
topics:
  lidar:
    topic: /lidar
    values:
      - '.fields[:]{name == "z"}.@length{==1}'
"""
    )

    rule = spec.topics[0].values[0]
    assert rule.path_source == '.fields[:]{name == "z"}.@length{==1}'


def test_parse_check_spec_rejects_comparatorless_non_predicate_path() -> None:
    with pytest.raises(CheckSpecError, match="must end in a filter"):
        _spec(
            """
version: 1
topics:
  lidar:
    topic: /lidar
    values:
      - .width
"""
        )


def test_parse_check_spec_reuses_topics_for_live_rules() -> None:
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
      expected: true
      severity: warn
"""
    )

    topic_live = spec.topics[0].live
    assert topic_live is not None
    assert topic_live.publishers is not None
    assert topic_live.publishers.minimum == 1
    assert topic_live.publishers.node_selector is not None
    assert topic_live.publishers.node_selector.matches_topic("/IMU_DRIVER")
    assert spec.live_nodes[0].selector.matches_topic("/localization")
    assert spec.has_live_rules


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("version: 2\ntopics: {}\n", "version must be 1"),
        ("version: 1\ntopics: []\n", "topics must be a mapping"),
        (
            "version: 1\ntopics:\n  x:\n    topic: /x\n    unknown: true\n",
            "unknown key",
        ),
        (
            "version: 1\ntopics:\n  x:\n    topic: '[bad'\n",
            "invalid topic regex",
        ),
        (
            "version: 1\ntopics:\n  x:\n    topic: /x\n    expected: false\n    timeout: 1s\n",
            "forbidden topic rule",
        ),
        (
            "version: 1\ntopics:\n  x:\n    topic: /x\n"
            "    frequency:\n      min: 10\n      window: 0s\n",
            "window must be greater than zero",
        ),
        (
            "version: 1\ntopics:\n  x:\n    topic: /x\n"
            "    frequency:\n      min: 10\n      max: 5\n      window: 1s\n",
            "min must not exceed max",
        ),
        (
            "version: 1\ntopics:\n  x:\n    topic: /x\n"
            "    frequency:\n      min: 10\n      tolerance: 1\n      window: 1s\n",
            "tolerance must be at least 0 and less than 1",
        ),
        (
            "version: 1\ntopics:\n  x:\n    topic: /x\n"
            "    live:\n      publishers:\n        min: 2\n        max: 1\n",
            "min must not exceed max",
        ),
        (
            "version: 1\ntopics:\n  x:\n    topic: /x\n"
            "    live:\n      publishers:\n        node: '[bad'\n",
            "invalid regex",
        ),
    ],
)
def test_parse_check_spec_rejects_invalid_contract(text: str, message: str) -> None:
    with pytest.raises(CheckSpecError, match=message):
        _spec(text)


def test_check_summary_only_does_not_scan_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/imu", 0, '{"value": 1}')])
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    schema:
      name: example/msg/Sample
      encoding: jsonschema
    message_encoding: json
"""
    )

    def fail_scan(*_args, **_kwargs):
        raise AssertionError("summary-only check scanned messages")

    monkeypatch.setattr(check_module, "read_message_decoded", fail_scan)

    report = check_mcap(str(path), spec)

    assert report.error_count == 0
    assert all(result.level == 0 for result in report.results)


def test_check_expected_and_forbidden_topics(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(
        path,
        [
            ("/imu", 0, '{"value": 1}'),
            ("/RADAR_FRONT", NS, '{"value": 2}'),
        ],
    )
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
  gps:
    topic: /sensor/gps
  forbidden_drivers:
    topic: /RADAR_.*
    expected: false
"""
    )

    report = check_mcap(str(path), spec)

    assert _result(report, "imu/expected").level == 0
    assert _result(report, "gps/expected").level == 2
    assert _result(report, "forbidden_drivers/expected").level == 2


def test_recording_check_validates_but_skips_live_constraints(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/imu", 0, "{}")])
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    live:
      publishers:
        min: 1
        node: /imu_driver
"""
    )

    report = check_mcap(str(path), spec)

    assert report.error_count == 0
    assert [result.name for result in report.results] == ["imu/expected"]


def test_check_topic_selector_is_case_insensitive_fullmatch(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/imu/data", 0, "{}"), ("/RADAR_FRONT", 1, "{}")])
    spec = _spec(
        """
version: 1
topics:
  exact_imu:
    topic: /IMU
  radar:
    topic: /radar_.*
"""
    )

    report = check_mcap(str(path), spec)

    assert _result(report, "exact_imu/expected").level == 2
    assert _result(report, "radar/expected").level == 0


def test_check_schema_checks_every_matching_channel(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_multi_channel_mcap(path)
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    schema:
      name: sensor_msgs/msg/Imu
      encoding: ros2msg
    message_encoding: cdr
"""
    )

    result = _result(check_mcap(str(path), spec), "imu:/imu/schema")

    assert result.level == 2
    assert "channel 2" in result.values["observed"]


def test_check_sliding_frequency_tolerance_and_timeout(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    sensor_events = [("/imu", index * NS // 10, f'{{"value": {index}}}') for index in range(21)]
    _write_json_mcap(path, sensor_events)
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      min: 10
      max: 10
      tolerance: 0.01
      window: 1s
    timeout: 110ms
"""
    )

    report = check_mcap(str(path), spec)

    frequency = _result(report, "imu:/imu/frequency")
    timeout = _result(report, "imu:/imu/timeout")
    assert frequency.level == 0
    assert frequency.values["minimum_hz"] == pytest.approx(10.0)
    assert frequency.values["maximum_hz"] == pytest.approx(10.0)
    assert timeout.level == 0
    assert timeout.values["maximum_gap_ns"] == NS // 10


def test_check_sliding_frequency_reports_worst_window(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    events = [
        ("/pose", 0, "{}"),
        ("/imu", 0, "{}"),
        ("/imu", NS // 10, "{}"),
        ("/imu", 2 * NS, "{}"),
        ("/pose", 2 * NS, "{}"),
    ]
    _write_json_mcap(path, events)
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      min: 2
      window: 1s
    timeout: 500ms
"""
    )

    report = check_mcap(str(path), spec)

    assert _result(report, "imu:/imu/frequency").level == 2
    assert _result(report, "imu:/imu/timeout").level == 2


def test_check_timing_does_not_decode_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/imu", 0, "not valid json"), ("/imu", NS, "not valid json")])
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    timeout: 1s
"""
    )

    def fail_decode(_message):
        raise AssertionError("timing-only check decoded a payload")

    monkeypatch.setattr(check_module, "_decoded_payload", fail_decode)

    assert check_mcap(str(path), spec).error_count == 0


def test_check_scan_excludes_irrelevant_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/imu", 0, "{}"), ("/CAM_FRONT/image", NS, "{}")])
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    timeout: 1s
"""
    )
    decisions: dict[str, bool] = {}
    original = check_module.read_message_decoded

    def track_channels(stream, *, should_include, **kwargs):
        def tracked(channel, schema):
            included = should_include(channel, schema)
            decisions[channel.topic] = included
            return included

        return original(stream, should_include=tracked, **kwargs)

    monkeypatch.setattr(check_module, "read_message_decoded", track_channels)

    check_mcap(str(path), spec)

    assert decisions == {"/imu": True, "/CAM_FRONT/image": False}


def test_check_sliding_frequency_state_is_window_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(
        path,
        [("/imu", index * NS // 10, "{}") for index in range(101)],
    )
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      min: 10
      max: 10
      window: 1s
"""
    )
    maximum_state_size = 0
    original = check_module._RateTracker.observe

    def track_state(rate, *args):
        nonlocal maximum_state_size
        original(rate, *args)
        maximum_state_size = max(maximum_state_size, len(rate.timestamps))

    monkeypatch.setattr(check_module._RateTracker, "observe", track_state)

    assert check_mcap(str(path), spec).error_count == 0
    assert maximum_state_size <= 11


def test_check_value_bounds_decode_once_per_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(
        path,
        [
            ("/imu", 0, '{"value": 1, "status": "ok"}'),
            ("/imu", NS, '{"value": 8, "status": "bad"}'),
        ],
    )
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .value
        min: 0
        max: 5
      - path: .status
        one_of: [ok, ready]
"""
    )
    decoded_count = 0
    original = check_module._decoded_payload

    def count_decode(message):
        nonlocal decoded_count
        decoded_count += 1
        return original(message)

    monkeypatch.setattr(check_module, "_decoded_payload", count_decode)

    report = check_mcap(str(path), spec)

    assert decoded_count == 2
    assert _result(report, "imu:/imu/value[0]").level == 2
    assert _result(report, "imu:/imu/value[1]").level == 2
    assert _result(report, "imu:/imu/value[0]").values["observed_maximum"] == 8


def test_check_messagepath_assertion_shorthand(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(
        path,
        [
            ("/lidar", 0, '{"fields": [{"name": "x"}, {"name": "z"}]}'),
            ("/lidar", NS, '{"fields": [{"name": "z"}, {"name": "z"}]}'),
        ],
    )
    spec = _spec(
        """
version: 1
topics:
  lidar:
    topic: /lidar
    values:
      - '.fields[:]{name == "z"}.@length{==1}'
"""
    )

    result = _result(check_mcap(str(path), spec), "lidar:/lidar/value[0]")

    assert result.level == 2
    assert result.values["observed_count"] == 1
    assert result.summary.endswith("has 1 failing values")


def test_check_cross_message_max_assertion(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(
        path,
        [("/temperature", index, f'{{"value": {value}}}') for index, value in enumerate([2, 5, 4])],
    )
    spec = _spec(
        """
version: 1
topics:
  temperature:
    topic: /temperature
    values:
      - '.value.@@max{<=5}'
"""
    )

    result = _result(check_mcap(str(path), spec), "temperature:/temperature/value[0]")

    assert result.level == 0
    assert result.values["observed_maximum"] == 5


def test_check_cross_message_delta_then_max_assertion(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(
        path,
        [("/position", index, f'{{"value": {value}}}') for index, value in enumerate([1, 4, 10])],
    )
    spec = _spec(
        """
version: 1
topics:
  position:
    topic: /position
    values:
      - '.value.@@delta.@@max{<=5}'
"""
    )

    result = _result(check_mcap(str(path), spec), "position:/position/value[0]")

    assert result.level == 2
    assert result.values["failure_samples"]


def test_check_cross_message_count_can_assert_no_matches(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/status", 0, '{"value": "BAD"}')])
    spec = _spec(
        """
version: 1
topics:
  status:
    topic: /status
    values:
      - '.value{=="OK"}.@@count{==0}'
"""
    )

    result = _result(check_mcap(str(path), spec), "status:/status/value[0]")

    assert result.level == 0
    assert result.values["observed_minimum"] == 0


def test_check_exposes_message_timestamp_variables(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/sensor", 100, '{"stamp": 95}')])
    spec = _spec(
        """
version: 1
topics:
  sensor:
    topic: /sensor
    values:
      - '.stamp.@sub($publish_time_ns).@abs{<=5}'
"""
    )

    result = _result(check_mcap(str(path), spec), "sensor:/sensor/value[0]")

    assert result.level == 0


def test_check_ros2_messagepath_and_failure_samples_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_ros2_imu_mcap(path)
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .linear_acceleration.@norm
        equals: 3
"""
    )

    result = _result(check_mcap(str(path), spec), "imu:/imu/value[0]")

    assert result.level == 0
    assert result.values["observed_count"] == 1


def test_check_value_failure_samples_are_capped(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(
        path,
        [("/imu", index, f'{{"value": {index}}}') for index in range(10)],
    )
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .value
        max: -1
"""
    )

    result = _result(check_mcap(str(path), spec), "imu:/imu/value[0]")

    assert result.summary == ".value has 10 failing values"
    assert str(result.values["failure_samples"]).count("; ") == 2


def test_check_warning_does_not_fail_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "recording.mcap"
    _write_json_mcap(path, [("/imu", 0, "{}")])
    spec_path = tmp_path / "check.yaml"
    spec_path.write_text(
        """
version: 1
topics:
  missing:
    topic: /sensor/gps
    severity: warn
"""
    )
    output = io.StringIO()
    monkeypatch.setattr(
        check_cmd,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=160),
    )

    assert check_cmd.check(str(path), spec=spec_path, num_workers=0) == 0
    assert "WARN" in output.getvalue()


def _strip_summary(path: Path) -> None:
    """Drop the summary section and footer, keeping every message."""
    data = path.read_bytes()
    summary_start = struct.unpack_from("<Q", data, len(data) - 28)[0]
    assert summary_start != 0
    path.write_bytes(data[:summary_start])


def test_check_frequency_max_violation_detected_without_summary(tmp_path: Path) -> None:
    events = [("/imu", int(seconds * NS), "{}") for seconds in (0.0, 2.0, 2.1, 2.2, 5.0)]
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      max: 2.5
      window: 1s
"""
    )
    full = tmp_path / "full.mcap"
    _write_json_mcap(full, events)
    torn = tmp_path / "torn.mcap"
    _write_json_mcap(torn, events)
    _strip_summary(torn)

    full_result = _result(check_mcap(str(full), spec, num_workers=0), "imu:/imu/frequency")
    torn_result = _result(check_mcap(str(torn), spec, num_workers=0), "imu:/imu/frequency")

    assert full_result.level == ERROR
    assert torn_result.level == ERROR
    assert torn_result.values["maximum_hz"] == full_result.values["maximum_hz"]


def test_check_recording_end_variable_errors_without_summary(tmp_path: Path) -> None:
    events = [("/imu", index * NS, f'{{"stamp": {index * NS}}}') for index in range(3)]
    spec = _spec(
        """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - '.stamp{<=$recording_end_ns}'
"""
    )
    full = tmp_path / "full.mcap"
    _write_json_mcap(full, events)
    torn = tmp_path / "torn.mcap"
    _write_json_mcap(torn, events)
    _strip_summary(torn)

    assert _result(check_mcap(str(full), spec, num_workers=0), "imu:/imu/value[0]").level == OK
    torn_result = _result(check_mcap(str(torn), spec, num_workers=0), "imu:/imu/value[0]")
    assert torn_result.level == ERROR
    assert "recording_end_ns" in torn_result.summary


def test_check_works_on_in_progress_recording_without_summary(tmp_path: Path) -> None:
    path = tmp_path / "in_progress.mcap"
    events = [("/temp", index * NS // 10, '{"temperature": 20}') for index in range(50)]
    _write_json_mcap(path, events)
    data = path.read_bytes()
    # Simulate a recording torn mid-write: no summary/footer and a truncated tail.
    path.write_bytes(data[: int(len(data) * 0.6)])

    spec = _spec(
        """\
version: 1
topics:
  temp:
    topic: /temp
    values:
      - '.temperature{<=25}'
"""
    )

    report = check_mcap(str(path), spec, num_workers=0)

    expected = _result(report, "temp/expected")
    value_result = _result(report, "temp:/temp/value[0]")
    assert expected.level == OK
    assert value_result.level == OK
    assert 0 < value_result.values["observed_count"] < 50


def test_observed_cell_compacts_ok_frequency_row() -> None:
    result = CheckResult(
        level=OK,
        name="imu:/imu/frequency",
        summary="frequency is within bounds",
        values={
            "minimum_hz": 87.234567,
            "maximum_hz": 109.8,
            "window_ns": 5 * NS,
            "required_minimum_hz": 85.5,
            "minimum_window_start_ns": 1532402934697511000,
            "required_maximum_hz": 115.5,
            "maximum_window_start_ns": 1532402940697148001,
        },
    )

    assert check_cmd._observed_cell(result) == "87.23-109.8 Hz"


def test_observed_cell_compacts_ok_value_row() -> None:
    result = CheckResult(
        level=OK,
        name="imu:/imu/value[0]",
        summary="values are within bounds",
        values={
            "observed_count": 1898,
            "observed_minimum": 8.482338960996273,
            "observed_maximum": 11.210457711811845,
        },
    )

    assert check_cmd._observed_cell(result) == "n=1898 in [8.482, 11.21]"


def test_observed_cell_compacts_ok_timeout_and_expected_rows() -> None:
    timeout = CheckResult(
        level=OK,
        name="imu:/imu/timeout",
        summary="message timeout is within limit",
        values={"maximum_gap_ns": 599937000, "timeout_ns": 650000000},
    )
    expected = CheckResult(
        level=OK,
        name="imu/expected",
        summary="expected topic is present",
        values={"topics": "/imu", "message_count": 1898},
    )

    assert check_cmd._observed_cell(timeout) == "max gap 0.600s"
    assert check_cmd._observed_cell(expected) == "topics=/imu, n=1898"


def test_observed_cell_keeps_full_detail_on_failure() -> None:
    result = CheckResult(
        level=ERROR,
        name="imu:/imu/frequency",
        summary="frequency below bounds",
        values={
            "required_minimum_hz": 85.5,
            "minimum_window_start_ns": 1532402934697511000,
            "minimum_hz": 12.345678,
        },
    )

    cell = check_cmd._observed_cell(result)

    assert cell == (
        "required_minimum_hz=85.5, "
        "minimum_window_start_ns=1532402934697511000, "
        "minimum_hz=12.345678"
    )


def test_load_check_spec_reports_source_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\ntopics: []\n")

    with pytest.raises(CheckSpecError, match=str(path)):
        load_check_spec(path)
