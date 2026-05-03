"""Schema-name canonicalisation shared across packages."""

from __future__ import annotations


def normalize_schema_name(name: str) -> str:
    """Canonicalise ROS1/ROS2 schema names to the short ``pkg/Type`` form.

    ROS2 IDL names appear as ``pkg/msg/Type`` (or ``pkg/srv/Type``,
    ``pkg/action/Type``); ROS1 + foxglove flatbuffer names are already
    ``pkg/Type``. Stripping the middle segment lets schema sets carry one
    canonical entry per type instead of two.
    """
    parts = name.split("/")
    if len(parts) == 3 and parts[1] in ("msg", "srv", "action"):
        return f"{parts[0]}/{parts[2]}"
    return name
