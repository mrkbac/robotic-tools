"""ROS 2 QoS-profile types.

Pure data: the policy enums and the :class:`QosProfile` dataclass.
Parsing logic lives in :mod:`pymcap_cli.core.qos`; rendering happens
in :mod:`pymcap_cli.display.display_utils`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# Numeric codes mirror ``rmw_qos_*_policy_t`` (see
# https://github.com/ros2/rmw/blob/master/rmw/include/rmw/qos_profiles.h).
# UNKNOWN is ROS 2's own "this was unrecognised at write time" sentinel;
# values *outside* the listed members (e.g. a future policy this CLI
# hasn't been taught about) flow through the parser as raw ints rather
# than being mapped to UNKNOWN — that way we don't conflate the two.
class Reliability(IntEnum):
    SYSTEM_DEFAULT = 0
    RELIABLE = 1
    BEST_EFFORT = 2
    UNKNOWN = 3
    BEST_AVAILABLE = 4  # Iron+


class Durability(IntEnum):
    SYSTEM_DEFAULT = 0
    TRANSIENT_LOCAL = 1
    VOLATILE = 2
    UNKNOWN = 3
    BEST_AVAILABLE = 4  # Iron+


class History(IntEnum):
    SYSTEM_DEFAULT = 0
    KEEP_LAST = 1
    KEEP_ALL = 2
    UNKNOWN = 3


class Liveliness(IntEnum):
    SYSTEM_DEFAULT = 0
    AUTOMATIC = 1
    # MANUAL_BY_NODE was removed in Foxy but still appears in numeric
    # metadata from older bag recorders. Keep the member so those
    # values don't render as UNKNOWN(2).
    MANUAL_BY_NODE = 2
    MANUAL_BY_TOPIC = 3
    UNKNOWN = 4
    BEST_AVAILABLE = 5  # Iron+


# Policy fields are either the matched enum member or the raw int the
# recorder wrote (when ROS 2 has added a code we don't yet know about).
PolicyValue = Reliability | Durability | History | Liveliness | int


@dataclass(frozen=True)
class QosProfile:
    """Normalised ROS 2 QoS profile.

    Policy fields hold the matched enum member when the recorder
    wrote a recognised code; otherwise the raw int flows through so
    display code can still surface it (rather than silently masking
    a future ROS 2 policy as ``UNKNOWN``). Unrecognised string forms
    are mapped to the enum's ``UNKNOWN`` — those almost always mean
    "garbled YAML", not "future policy".

    Duration fields are nanoseconds or ``None`` when the recorder
    wrote the ``int64.max`` sentinel (= "unspecified / infinite").
    """

    reliability: Reliability | int
    durability: Durability | int
    history: History | int
    depth: int
    liveliness: Liveliness | int
    lifespan_ns: int | None
    deadline_ns: int | None
    liveliness_lease_ns: int | None
    avoid_ros_namespace_conventions: bool
