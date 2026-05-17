"""Tests for ROS 2 QoS-profile parsing from MCAP channel metadata."""

from __future__ import annotations

import yaml
from pymcap_cli.core.qos import parse_qos_profiles
from pymcap_cli.display.display_utils import _format_qos_compact
from pymcap_cli.doctor import _qos_issues as qos_issues
from pymcap_cli.types.qos import (
    Durability,
    History,
    Liveliness,
    QosProfile,
    Reliability,
)
from rich.text import Text


def _plain(markup: str) -> str:
    """Strip Rich colour tags so assertions stay readable."""
    return Text.from_markup(markup).plain


_LEGACY_NUMERIC_YAML = """\
- history: 1
  depth: 10
  reliability: 1
  durability: 2
  deadline:
    sec: 9223372036
    nsec: 854775807
  lifespan:
    sec: 9223372036
    nsec: 854775807
  liveliness: 1
  liveliness_lease_duration:
    sec: 9223372036
    nsec: 854775807
  avoid_ros_namespace_conventions: false
"""


_MODERN_STRING_YAML = """\
- history: keep_last
  depth: 5
  reliability: best_effort
  durability: volatile
  deadline:
    sec: 0
    nanosec: 500000000
  lifespan:
    sec: 9223372036854775807
    nanosec: 0
  liveliness: automatic
  liveliness_lease_duration:
    sec: 9223372036854775807
    nanosec: 0
  avoid_ros_namespace_conventions: false
"""


def test_parses_legacy_numeric_form() -> None:
    profiles = parse_qos_profiles({"offered_qos_profiles": _LEGACY_NUMERIC_YAML})
    assert len(profiles) == 1
    p = profiles[0]
    assert p.reliability is Reliability.RELIABLE
    assert p.durability is Durability.VOLATILE
    assert p.history is History.KEEP_LAST
    assert p.depth == 10
    assert p.liveliness is Liveliness.AUTOMATIC
    # All durations were written as the int64-max-ns sentinel → unspecified.
    assert p.deadline_ns is None
    assert p.lifespan_ns is None
    assert p.liveliness_lease_ns is None
    assert p.avoid_ros_namespace_conventions is False


def test_parses_modern_string_form_with_real_deadline() -> None:
    profiles = parse_qos_profiles({"offered_qos_profiles": _MODERN_STRING_YAML})
    assert len(profiles) == 1
    p = profiles[0]
    assert p.reliability is Reliability.BEST_EFFORT
    assert p.durability is Durability.VOLATILE
    assert p.history is History.KEEP_LAST
    assert p.depth == 5
    # 500 ms deadline rather than sentinel.
    assert p.deadline_ns == 500_000_000
    # Lifespan + liveliness lease are sentinels → unspecified.
    assert p.lifespan_ns is None
    assert p.liveliness_lease_ns is None


def test_unknown_int_flows_through_raw() -> None:
    # 99 is unrecognised; must surface as int rather than being mapped to UNKNOWN.
    raw = yaml.safe_dump([{"reliability": 99, "history": 1, "depth": 1}])
    profile = parse_qos_profiles({"offered_qos_profiles": raw})[0]
    assert profile.reliability == 99
    assert not isinstance(profile.reliability, Reliability)
    assert "UNKNOWN(99)" in _plain(_format_qos_compact(profile))


def test_explicit_unknown_sentinel_maps_to_enum() -> None:
    raw = yaml.safe_dump([{"reliability": 3, "history": 1, "depth": 1}])
    profile = parse_qos_profiles({"offered_qos_profiles": raw})[0]
    # ``3`` is the ROS 2 RELIABILITY_UNKNOWN code — should map to the enum,
    # not flow through as a raw int.
    assert profile.reliability is Reliability.UNKNOWN


def test_unknown_string_maps_to_unknown_enum() -> None:
    raw = yaml.safe_dump([{"reliability": "no_such_policy", "history": 1, "depth": 1}])
    profile = parse_qos_profiles({"offered_qos_profiles": raw})[0]
    # Unrecognised string forms are almost always typos / garbled YAML —
    # mapping to UNKNOWN is the right call (unlike numeric codes, which
    # flow through raw so future ROS 2 policies stay visible).
    assert profile.reliability is Reliability.UNKNOWN


def test_missing_metadata_key_returns_empty() -> None:
    assert parse_qos_profiles({}) == []
    assert parse_qos_profiles({"some_other_key": "x"}) == []
    assert parse_qos_profiles(None) == []


def test_malformed_yaml_returns_empty() -> None:
    assert parse_qos_profiles({"offered_qos_profiles": ": not valid yaml ::"}) == []


def test_non_list_yaml_returns_empty() -> None:
    # Some recorders might write a single dict by mistake — we expect a list.
    assert parse_qos_profiles({"offered_qos_profiles": "history: 1"}) == []


def test_format_qos_compact_includes_depth_for_keep_last() -> None:
    profile = QosProfile(
        reliability=Reliability.RELIABLE,
        durability=Durability.TRANSIENT_LOCAL,
        history=History.KEEP_LAST,
        depth=20,
        liveliness=Liveliness.AUTOMATIC,
        lifespan_ns=None,
        deadline_ns=None,
        liveliness_lease_ns=None,
        avoid_ros_namespace_conventions=False,
    )
    assert _plain(_format_qos_compact(profile)) == "RELIABLE/TRANSIENT_LOCAL/KEEP_LAST(20)"


def test_format_qos_compact_keep_all() -> None:
    profile = QosProfile(
        reliability=Reliability.BEST_EFFORT,
        durability=Durability.VOLATILE,
        history=History.KEEP_ALL,
        depth=0,
        liveliness=Liveliness.AUTOMATIC,
        lifespan_ns=None,
        deadline_ns=None,
        liveliness_lease_ns=None,
        avoid_ros_namespace_conventions=False,
    )
    assert _plain(_format_qos_compact(profile)) == "BEST_EFFORT/VOLATILE/KEEP_ALL"


def test_format_qos_compact_system_default_history_omitted() -> None:
    profile = QosProfile(
        reliability=Reliability.RELIABLE,
        durability=Durability.VOLATILE,
        history=History.SYSTEM_DEFAULT,
        depth=0,
        liveliness=Liveliness.AUTOMATIC,
        lifespan_ns=None,
        deadline_ns=None,
        liveliness_lease_ns=None,
        avoid_ros_namespace_conventions=False,
    )
    assert _plain(_format_qos_compact(profile)) == "RELIABLE/VOLATILE"


def test_best_available_recognised() -> None:
    # ROS 2 Iron+ adds BEST_AVAILABLE at code 4 — should map to the enum.
    raw = yaml.safe_dump([{"reliability": 4, "history": 1, "depth": 1}])
    profile = parse_qos_profiles({"offered_qos_profiles": raw})[0]
    assert profile.reliability is Reliability.BEST_AVAILABLE


def test_zero_duration_is_unspecified() -> None:
    # rosbag2 commonly writes ``{sec: 0, nsec: 0}`` for "policy not applied".
    # That's a sentinel, not a finite 0 ns deadline.
    raw = yaml.safe_dump(
        [
            {
                "history": 1,
                "depth": 1,
                "reliability": 1,
                "durability": 2,
                "deadline": {"sec": 0, "nsec": 0},
                "lifespan": {"sec": 0, "nsec": 0},
                "liveliness": 1,
                "liveliness_lease_duration": {"sec": 0, "nsec": 0},
            }
        ]
    )
    profile = parse_qos_profiles({"offered_qos_profiles": raw})[0]
    assert profile.deadline_ns is None
    assert profile.lifespan_ns is None
    assert profile.liveliness_lease_ns is None


def test_qos_issues_empty_for_clean_yaml() -> None:
    assert qos_issues({"offered_qos_profiles": _LEGACY_NUMERIC_YAML}) == []
    assert qos_issues({"offered_qos_profiles": _MODERN_STRING_YAML}) == []


def test_qos_issues_missing_key() -> None:
    assert qos_issues({}) == []
    assert qos_issues({"other": "data"}) == []
    assert qos_issues(None) == []


def test_qos_issues_malformed_yaml() -> None:
    issues = qos_issues({"offered_qos_profiles": ": not valid ::"})
    assert len(issues) == 1
    assert "malformed" in issues[0]


def test_qos_issues_not_a_list() -> None:
    issues = qos_issues({"offered_qos_profiles": "history: 1"})
    assert any("must be a YAML list" in i for i in issues)


def test_qos_issues_unknown_reliability_code() -> None:
    raw = yaml.safe_dump([{"history": 1, "depth": 1, "reliability": 99}])
    issues = qos_issues({"offered_qos_profiles": raw})
    assert any("reliability code 99" in i for i in issues)


def test_qos_issues_unknown_string() -> None:
    raw = yaml.safe_dump([{"history": 1, "depth": 1, "reliability": "definitely_not_a_policy"}])
    issues = qos_issues({"offered_qos_profiles": raw})
    assert any("reliability" in i and "definitely_not_a_policy" in i for i in issues)


def test_legacy_manual_by_node_recognised() -> None:
    # Liveliness=2 (MANUAL_BY_NODE) was removed in Foxy but pre-Foxy
    # recordings still write it. Should map to the enum, not UNKNOWN(2).
    raw = yaml.safe_dump([{"history": 1, "depth": 1, "liveliness": 2}])
    profile = parse_qos_profiles({"offered_qos_profiles": raw})[0]
    assert profile.liveliness is Liveliness.MANUAL_BY_NODE


def test_format_qos_compact_emits_colour_markup_per_value() -> None:
    profile = QosProfile(
        reliability=Reliability.RELIABLE,
        durability=Durability.VOLATILE,
        history=History.KEEP_LAST,
        depth=10,
        liveliness=Liveliness.AUTOMATIC,
        lifespan_ns=None,
        deadline_ns=None,
        liveliness_lease_ns=None,
        avoid_ros_namespace_conventions=False,
    )
    out = _format_qos_compact(profile)
    # Each axis should carry its own Rich colour tag so a glance
    # distinguishes RELIABLE topics from BEST_EFFORT etc. The exact
    # colour comes from text_to_color hashing, so just check the tags
    # wrap each policy name.
    assert "]RELIABLE[/]" in out
    assert "]VOLATILE[/]" in out
    assert "]KEEP_LAST(10)[/]" in out


def test_multiple_profiles_parsed_in_order() -> None:
    raw = yaml.safe_dump(
        [
            {"reliability": 1, "history": 1, "depth": 5},
            {"reliability": 2, "history": 2, "depth": 0},
        ]
    )
    profiles = parse_qos_profiles({"offered_qos_profiles": raw})
    assert [p.reliability for p in profiles] == [Reliability.RELIABLE, Reliability.BEST_EFFORT]
