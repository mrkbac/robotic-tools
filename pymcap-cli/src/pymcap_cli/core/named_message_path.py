"""Named Foxglove message paths shared by plotting and tabular export."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ros_parser.message_path import MessagePath, parse_message_path

_COLUMN_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class NamedMessagePath:
    name: str
    source: str
    path: MessagePath


def parse_path_arg(arg: str) -> tuple[str, str]:
    """Parse ``Label=/path`` or a bare ``/path`` into its label and path."""
    if "=" in arg and not arg.startswith("/"):
        label, _, path_str = arg.partition("=")
        return label, path_str
    return arg, arg


def parse_named_columns(args: list[str] | None) -> tuple[NamedMessagePath, ...]:
    columns: list[NamedMessagePath] = []
    seen: set[tuple[str, str]] = set()
    for arg in args or ():
        name, path_str = parse_path_arg(arg)
        name = name.strip()
        path_str = path_str.strip()
        if name == path_str:
            raise ValueError(f"Column expression {arg!r} must use NAME=/topic.path syntax")
        if _COLUMN_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Invalid column name {name!r}; use letters, numbers, and underscores")
        if not path_str.startswith("/"):
            raise ValueError(f"Column {name!r} must use a topic-qualified message path")
        try:
            path = parse_message_path(path_str)
        except Exception as exc:
            raise ValueError(f"Invalid message path for column {name!r}: {exc}") from exc
        key = (path.topic, name)
        if key in seen:
            raise ValueError(f"Duplicate column name {name!r} for topic {path.topic!r}")
        seen.add(key)
        columns.append(NamedMessagePath(name=name, source=path_str, path=path))
    return tuple(columns)
