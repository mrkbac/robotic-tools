"""Parse ROS 2 QoS profiles from MCAP ``channel.metadata``.

ROS 2 bag recorders stash QoS profiles into the channel metadata
under the key ``offered_qos_profiles`` (and sometimes
``subscribed_qos_profiles``), as a YAML-serialised list of profile
dicts. Both numeric and string policy forms exist in the wild
because ``rosbag2_storage`` branches on a metadata-version field
when serialising policy enums (see
``rosbag2_storage/src/rosbag2_storage/qos.cpp``):

- ``version < 9`` writes the raw ``static_cast<int>(policy)`` value.
- ``version >= 9`` writes the lowercase string name via
  ``rmw_qos_*_policy_to_str``.

Durations come as ``{sec, nsec}`` (legacy) or ``{sec, nanosec}``
(modern) and use ``int64.max`` ns or ``{0, 0}`` as the
"unset / infinite" sentinel.

This module normalises every shape into a :class:`QosProfile`
dataclass; the dataclass and policy enums live in
:mod:`pymcap_cli.types.qos`. Rendering happens in
:mod:`pymcap_cli.display.display_utils`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, TypeAlias, cast

import yaml

from pymcap_cli.types.qos import (
    Durability,
    History,
    Liveliness,
    QosProfile,
    Reliability,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


# Recorders use this as the "unset / infinite" sentinel — it's
# ``int64.max`` worth of nanoseconds split across ``sec`` and
# ``nsec`` fields.
_DURATION_UNSET_NS = 2**63 - 1

YamlScalar: TypeAlias = None | bool | int | float | str
YamlValue: TypeAlias = YamlScalar | list["YamlValue"] | dict[str, "YamlValue"]

_HUMBLE_POLICY_CODES: dict[str, dict[str, int]] = {
    "history": {
        "system_default": 0,
        "keep_last": 1,
        "keep_all": 2,
        "unknown": 3,
    },
    "reliability": {
        "system_default": 0,
        "reliable": 1,
        "best_effort": 2,
        "unknown": 3,
    },
    "durability": {
        "system_default": 0,
        "transient_local": 1,
        "volatile": 2,
        "unknown": 3,
    },
    "liveliness": {
        "system_default": 0,
        "automatic": 1,
        "manual_by_node": 2,
        "manual_by_topic": 3,
        "unknown": 4,
    },
}

_QOS_DURATION_FIELDS = {"deadline", "lifespan", "liveliness_lease_duration"}
_QOS_OVERRIDE_FIELDS = {
    *_HUMBLE_POLICY_CODES,
    *_QOS_DURATION_FIELDS,
    "depth",
    "avoid_ros_namespace_conventions",
}


def parse_qos_profiles(metadata: Mapping[str, str] | None) -> list[QosProfile]:
    """Parse ``offered_qos_profiles`` from a channel's metadata dict.

    Returns ``[]`` when the metadata is missing the key, the YAML
    doesn't parse, or it doesn't shape like a list of profile dicts.
    Never raises — non-ROS2 MCAPs simply yield empty lists.
    """
    if not metadata:
        return []
    raw = metadata.get("offered_qos_profiles")
    if not raw:
        return []
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError:
        return []
    if not isinstance(loaded, list):
        return []

    profiles: list[QosProfile] = []
    for entry in loaded:
        if not isinstance(entry, dict):
            continue
        profiles.append(_profile_from_yaml(cast("dict[str, object]", entry)))
    return profiles


def qos_profiles_to_numeric(raw: str) -> str:
    """Convert Jazzy string policy names to Humble-compatible integer codes.

    The four policy fields are the only values rewritten. Already-numeric and
    empty inputs are returned unchanged so repeated conversion is idempotent.
    """
    if not raw.strip():
        return raw
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"offered_qos_profiles YAML is malformed: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(loaded, list):
        msg = f"offered_qos_profiles must be a YAML list, got {type(loaded).__name__}"
        raise TypeError(msg)
    if not loaded:
        return raw

    profiles = cast("list[YamlValue]", loaded)
    was_changed = False
    for index, entry in enumerate(profiles):
        if not isinstance(entry, dict):
            msg = f"offered_qos_profiles profile {index} must be a mapping"
            raise TypeError(msg)
        profile = cast("dict[str, YamlValue]", entry)
        for field, name_to_code in _HUMBLE_POLICY_CODES.items():
            if field not in profile:
                msg = f"offered_qos_profiles profile {index} is missing {field!r}"
                raise ValueError(msg)
            value = profile[field]
            if isinstance(value, bool):
                msg = f"profile {index} {field} must be a policy name or integer, got boolean"
                raise TypeError(msg)
            if isinstance(value, int):
                if value not in name_to_code.values():
                    msg = f"profile {index} {field} code {value} is not supported by Humble"
                    raise ValueError(msg)
                continue
            if not isinstance(value, str):
                msg = (
                    f"profile {index} {field} must be a policy name or integer, "
                    f"got {type(value).__name__}"
                )
                raise TypeError(msg)
            normalized = value.strip().lower()
            code = name_to_code.get(normalized)
            if code is None:
                msg = f"profile {index} {field} policy {value!r} is not supported by Humble"
                raise ValueError(msg)
            profile[field] = code
            was_changed = True

    if not was_changed:
        return raw
    return yaml.safe_dump(loaded, sort_keys=False)


def parse_qos_override_yaml(raw: str) -> dict[str, dict[str, YamlValue]]:
    """Parse the standard ROS 2 topic-to-QoS-profile override shape."""
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"QoS override YAML is malformed: {exc}"
        raise ValueError(msg) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"QoS override YAML must be a topic mapping, got {type(loaded).__name__}"
        raise TypeError(msg)

    overrides: dict[str, dict[str, YamlValue]] = {}
    for topic, raw_profile in loaded.items():
        if not isinstance(topic, str) or not topic:
            msg = "QoS override topic names must be non-empty strings"
            raise TypeError(msg)
        if not isinstance(raw_profile, dict):
            msg = f"QoS override for topic {topic!r} must be a mapping"
            raise TypeError(msg)
        profile: dict[str, YamlValue] = {}
        for field, value in raw_profile.items():
            if not isinstance(field, str):
                msg = f"QoS override fields for topic {topic!r} must be strings"
                raise TypeError(msg)
            typed_value = cast("YamlValue", value)
            _validate_qos_override_value(field, typed_value)
            profile[field] = typed_value
        overrides[topic] = profile
    return overrides


def parse_qos_set_rule(token: str) -> tuple[str, str, YamlValue]:
    """Parse ``TOPIC_REGEX:FIELD=VALUE`` with VALUE interpreted as YAML."""
    separators = [
        (token.rfind(f":{field}="), field)
        for field in _QOS_OVERRIDE_FIELDS
        if f":{field}=" in token
    ]
    if not separators:
        msg = f"Expected 'TOPIC_REGEX:QOS_POLICY=VALUE' with a supported QoS policy, got {token!r}"
        raise ValueError(msg)
    separator_index, field = max(separators)
    pattern = token[:separator_index]
    raw_value = token[separator_index + len(field) + 2 :]
    if not pattern:
        msg = f"QoS set rule has an empty topic regex: {token!r}"
        raise ValueError(msg)
    try:
        value = cast("YamlValue", yaml.safe_load(raw_value))
    except yaml.YAMLError as exc:
        msg = f"QoS set value is malformed YAML in {token!r}: {exc}"
        raise ValueError(msg) from exc
    _validate_qos_override_value(field, value)
    return pattern, field, value


def qos_profiles_with_overrides(raw: str, overrides: Mapping[str, YamlValue]) -> str:
    """Apply partial overrides to every recorded offered QoS profile."""
    if not raw.strip():
        msg = "offered_qos_profiles is empty"
        raise ValueError(msg)
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"offered_qos_profiles YAML is malformed: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(loaded, list):
        msg = f"offered_qos_profiles must be a YAML list, got {type(loaded).__name__}"
        raise TypeError(msg)
    if not loaded:
        msg = "offered_qos_profiles must contain at least one profile"
        raise ValueError(msg)

    profiles = cast("list[YamlValue]", loaded)
    was_changed = False
    for index, entry in enumerate(profiles):
        if not isinstance(entry, dict):
            msg = f"offered_qos_profiles profile {index} must be a mapping"
            raise TypeError(msg)
        profile = cast("dict[str, YamlValue]", entry)
        for field, value in overrides.items():
            if profile.get(field) != value:
                profile[field] = copy.deepcopy(value)
                was_changed = True

    if not was_changed:
        return raw
    return yaml.safe_dump(loaded, sort_keys=False)


def _validate_qos_override_value(field: str, value: YamlValue) -> None:
    if field not in _QOS_OVERRIDE_FIELDS:
        msg = f"Unsupported QoS policy {field!r}"
        raise ValueError(msg)

    policy_codes = _HUMBLE_POLICY_CODES.get(field)
    if policy_codes is not None:
        if isinstance(value, bool):
            msg = f"QoS policy {field!r} must be a policy name or integer"
            raise TypeError(msg)
        if isinstance(value, int):
            if value not in policy_codes.values():
                msg = f"QoS policy {field!r} code {value} is not supported by Humble"
                raise ValueError(msg)
            return
        if isinstance(value, str) and value.strip().lower() in policy_codes:
            return
        msg = f"QoS policy {field!r} value {value!r} is not supported by Humble"
        raise ValueError(msg)

    if field == "depth":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = "QoS policy 'depth' must be a non-negative integer"
            raise TypeError(msg)
        return

    if field == "avoid_ros_namespace_conventions":
        if not isinstance(value, bool):
            msg = "QoS policy 'avoid_ros_namespace_conventions' must be a boolean"
            raise TypeError(msg)
        return

    if not isinstance(value, dict):
        msg = f"QoS duration policy {field!r} must be a mapping"
        raise TypeError(msg)
    duration = cast("dict[str, YamlValue]", value)
    nsec_field = "nsec" if "nsec" in duration else "nanosec"
    if set(duration) != {"sec", nsec_field}:
        msg = f"QoS duration policy {field!r} requires sec and nsec"
        raise ValueError(msg)
    sec = duration["sec"]
    nsec = duration[nsec_field]
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or sec < 0
        or isinstance(nsec, bool)
        or not isinstance(nsec, int)
        or not 0 <= nsec < 1_000_000_000
    ):
        msg = f"QoS duration policy {field!r} requires non-negative integer sec/nsec"
        raise TypeError(msg)


def _profile_from_yaml(entry: Mapping[str, object]) -> QosProfile:
    return QosProfile(
        reliability=_coerce_policy(entry.get("reliability"), Reliability),
        durability=_coerce_policy(entry.get("durability"), Durability),
        history=_coerce_policy(entry.get("history"), History),
        depth=_int_or(entry.get("depth"), 0),
        liveliness=_coerce_policy(entry.get("liveliness"), Liveliness),
        lifespan_ns=_duration_ns(entry.get("lifespan")),
        deadline_ns=_duration_ns(entry.get("deadline")),
        liveliness_lease_ns=_duration_ns(entry.get("liveliness_lease_duration")),
        avoid_ros_namespace_conventions=bool(entry.get("avoid_ros_namespace_conventions", False)),
    )


def _coerce_policy(
    value: object,
    enum_cls: type[Reliability | Durability | History | Liveliness],
) -> Reliability | Durability | History | Liveliness | int:
    """Convert a recorder-written value into the matching enum or its raw int.

    Unrecognised numeric codes flow through unchanged so future ROS 2
    policies don't get silently flattened into ``UNKNOWN``. Unrecognised
    string names map to the enum's ``UNKNOWN`` — those are almost always
    typos or garbled YAML, not new policies.
    """
    unknown = enum_cls["UNKNOWN"]
    if isinstance(value, bool):
        return unknown
    if isinstance(value, int):
        try:
            return enum_cls(value)
        except ValueError:
            return value
    if isinstance(value, str):
        key = value.strip().upper()
        if key in enum_cls.__members__:
            return enum_cls[key]
    return unknown


def _int_or(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _duration_ns(value: object) -> int | None:
    """Normalise a ROS 2 duration dict to nanoseconds, or ``None`` if unset.

    Two sentinels mean "default / unspecified" and both map to ``None``:

    - ``{sec: 0, nsec: 0}`` — what most recorders write for "policy not
      applied" (e.g. no deadline configured).
    - ``int64.max`` ns split across sec / nsec — what rmw writes as the
      explicit ``RMW_DURATION_INFINITE`` sentinel.
    """
    if not isinstance(value, dict):
        return None
    duration = cast("dict[str, object]", value)
    sec = duration.get("sec")
    # Recorders disagree on the sub-second field name.
    nsec = duration.get("nsec")
    if nsec is None:
        nsec = duration.get("nanosec")
    if not isinstance(sec, int) or not isinstance(nsec, int):
        return None
    total = sec * 1_000_000_000 + nsec
    if total <= 0 or total >= _DURATION_UNSET_NS:
        return None
    return total
