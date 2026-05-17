"""Parse ROS 2 QoS profiles from MCAP ``channel.metadata``.

ROS 2 bag recorders stash QoS profiles into the channel metadata
under the key ``offered_qos_profiles`` (and sometimes
``subscribed_qos_profiles``), as a YAML-serialised list of
profile dicts. Older recorders write numeric enum codes; newer ones
write lowercase strings. Durations come as ``{sec, nsec}`` (legacy)
or ``{sec, nanosec}`` (modern) and use ``int64.max`` ns as the
"unset / infinite" sentinel.

This module normalises both shapes into a :class:`QosProfile`
dataclass; the dataclass and policy enums live in
:mod:`pymcap_cli.types.qos`. Rendering happens in
:mod:`pymcap_cli.display.display_utils`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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
