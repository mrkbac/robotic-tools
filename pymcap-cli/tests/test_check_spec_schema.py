"""Agreement tests between schemas/mcap_check_spec.json and the check spec parser.

Invariants:
- Every spec the parser accepts must validate against the JSON schema.
- Every structurally invalid spec must be rejected by BOTH the schema and the parser.
- Semantic-only rules (regex validity, MessagePath validity, min<=max) are parser-only;
  the schema intentionally accepts those specs.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml
from pymcap_cli.check import CheckSpecError, parse_check_spec

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "mcap_check_spec.json"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "check"
VALIDATOR = jsonschema.Draft7Validator(json.loads(SCHEMA_PATH.read_text()))


def _schema_errors(spec_yaml: str) -> list[jsonschema.ValidationError]:
    return list(VALIDATOR.iter_errors(yaml.safe_load(spec_yaml)))


def test_schema_is_valid_draft7():
    jsonschema.Draft7Validator.check_schema(json.loads(SCHEMA_PATH.read_text()))


VALID_SPECS = {
    "empty_topics": """
version: 1
topics: {}
""",
    "minimal_topic": """
version: 1
topics:
  imu:
    topic: /imu
""",
    "forbidden_topic": """
version: 1
topics:
  no_clock:
    topic: /clock
    expected: false
    severity: warn
""",
    "full_featured": """
version: 1
topics:
  imu:
    topic: /imu
    expected: true
    severity: warn
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
      - '.linear_acceleration.@norm{<=30}'
      - '.header.frame_id{=="imu_link"}'
      - path: .temperature
        min: -40
        max: 85
      - path: .status
        equals: OK
      - path: .mode
        one_of: [1, 2, auto]
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
      severity: error
""",
    "bare_number_durations": """
version: 1
topics:
  odom:
    topic: /odom
    frequency:
      max: 60
      window: 5
    timeout: 0.5
""",
    "root_filter_value": """
version: 1
topics:
  lidar:
    topic: /LIDAR_TOP
    values:
      - '{width > 0}'
""",
}


@pytest.mark.parametrize("spec_yaml", VALID_SPECS.values(), ids=VALID_SPECS.keys())
def test_valid_spec_passes_parser_and_schema(spec_yaml):
    parse_check_spec(spec_yaml)
    assert _schema_errors(spec_yaml) == []


@pytest.mark.parametrize("example", sorted(EXAMPLES_DIR.glob("*.yaml")), ids=lambda path: path.name)
def test_example_spec_passes_parser_and_schema(example):
    text = example.read_text()
    parse_check_spec(text, source=str(example))
    assert _schema_errors(text) == []


INVALID_SPECS = {
    "unknown_root_key": """
version: 1
topics: {}
extra: 1
""",
    "missing_version": """
topics: {}
""",
    "version_2": """
version: 2
topics: {}
""",
    "version_true": """
version: true
topics: {}
""",
    "topics_list": """
version: 1
topics:
  - topic: /imu
""",
    "missing_topic_key": """
version: 1
topics:
  imu:
    expected: true
""",
    "unknown_topic_key": """
version: 1
topics:
  imu:
    topic: /imu
    frequency_hz: 100
""",
    "bad_severity": """
version: 1
topics:
  imu:
    topic: /imu
    severity: fatal
""",
    "forbidden_topic_with_checks": """
version: 1
topics:
  no_clock:
    topic: /clock
    expected: false
    timeout: 1s
""",
    "empty_schema_rule": """
version: 1
topics:
  imu:
    topic: /imu
    schema: {}
""",
    "frequency_without_bounds": """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      window: 1s
""",
    "frequency_without_window": """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      min: 10
""",
    "frequency_negative_min": """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      min: -1
      window: 1s
""",
    "tolerance_too_large": """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      min: 10
      tolerance: 1
      window: 1s
""",
    "zero_duration": """
version: 1
topics:
  imu:
    topic: /imu
    timeout: 0ms
""",
    "bad_duration": """
version: 1
topics:
  imu:
    topic: /imu
    timeout: fast
""",
    "empty_values": """
version: 1
topics:
  imu:
    topic: /imu
    values: []
""",
    "value_not_relative": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - 'linear_acceleration{<=30}'
""",
    "value_string_without_predicate": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - '.linear_acceleration.@norm'
""",
    "value_mapping_without_comparators_or_predicate": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .temperature
""",
    "value_mapping_min_and_equals": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .temperature
        min: 0
        equals: 1
""",
    "value_mapping_equals_and_one_of": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .status
        equals: OK
        one_of: [OK, DEGRADED]
""",
    "empty_one_of": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .status
        one_of: []
""",
    "empty_live_topic_rule": """
version: 1
topics:
  imu:
    topic: /imu
    live: {}
""",
    "empty_endpoint_rule": """
version: 1
topics:
  imu:
    topic: /imu
    live:
      publishers: {}
""",
    "negative_endpoint_min": """
version: 1
topics:
  imu:
    topic: /imu
    live:
      publishers:
        min: -1
""",
    "empty_live_nodes": """
version: 1
topics: {}
live:
  nodes: {}
""",
    "live_node_without_node": """
version: 1
topics: {}
live:
  nodes:
    localization:
      expected: true
""",
}


@pytest.mark.parametrize("spec_yaml", INVALID_SPECS.values(), ids=INVALID_SPECS.keys())
def test_invalid_spec_rejected_by_parser_and_schema(spec_yaml):
    with pytest.raises(CheckSpecError):
        parse_check_spec(spec_yaml)
    assert _schema_errors(spec_yaml)


PARSER_ONLY_SPECS = {
    "invalid_topic_regex": """
version: 1
topics:
  imu:
    topic: '['
""",
    "invalid_message_path": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - '.linear_acceleration..bad{<=30}'
""",
    "frequency_min_above_max": """
version: 1
topics:
  imu:
    topic: /imu
    frequency:
      min: 100
      max: 10
      window: 1s
""",
    "value_min_above_max": """
version: 1
topics:
  imu:
    topic: /imu
    values:
      - path: .temperature
        min: 10
        max: -10
""",
}


@pytest.mark.parametrize("spec_yaml", PARSER_ONLY_SPECS.values(), ids=PARSER_ONLY_SPECS.keys())
def test_semantic_rules_are_parser_only(spec_yaml):
    """The schema is structural; these specs need the parser to be rejected."""
    with pytest.raises(CheckSpecError):
        parse_check_spec(spec_yaml)
    assert _schema_errors(spec_yaml) == []
