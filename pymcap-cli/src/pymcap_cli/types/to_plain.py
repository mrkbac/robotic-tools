"""Convert decoded ROS2 messages into plain Python primitives.

The ROS2 decoder returns ``memoryview.cast('d')``/``'f'``/``'i'`` etc. for
primitive fixed-size arrays — those memoryviews know their element width and
are valid sequences of typed values. Only byte-format memoryviews (``'B'`` /
``'b'`` for ``uint8[]`` / ``int8[]``) collapse to ``bytes``; for typed
memoryviews we call ``.tolist()`` so downstream consumers see N floats, not
8N bytes.

Values shaped like ``builtin_interfaces/Time`` / ``Duration`` (exactly
``sec`` + ``nanosec`` fields) collapse to int nanoseconds.
"""

from __future__ import annotations

from typing import Any

from pymcap_cli.utils import NS_TO_SEC

_TIME_FIELDS: frozenset[str] = frozenset({"sec", "nanosec"})


def to_plain(obj: Any) -> Any:
    """Recursively convert a decoded ROS message into plain dict/list/primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, memoryview):
        if obj.format in ("B", "b", "c"):
            return obj.tobytes()
        return obj.tolist()
    if isinstance(obj, dict):
        if obj.keys() == _TIME_FIELDS:
            return int(obj["sec"]) * NS_TO_SEC + int(obj["nanosec"])
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        if set(slots) == _TIME_FIELDS:
            return int(obj.sec) * NS_TO_SEC + int(obj.nanosec)
        return {k: to_plain(getattr(obj, k)) for k in slots}
    dct = getattr(obj, "__dict__", None)
    if dct is not None:
        if dct.keys() == _TIME_FIELDS:
            return int(obj.sec) * NS_TO_SEC + int(obj.nanosec)
        return {k: to_plain(v) for k, v in dct.items()}
    return str(obj)
