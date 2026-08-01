"""Shared formatting helpers for resolved message definitions."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

from rich.console import Console

from pymcap_cli.display.schema_render import render_schema_definition

if TYPE_CHECKING:
    from pymcap_cli.core.msg_resolver import ROS2Distro

_CONSTANT_RE = re.compile(r"^\s*\S+\s+\S+\s*=")


def compact_message_definition(definition: str) -> str:
    """Remove comments, constants, and blank lines from a resolved definition.

    Comment and constant detection is quote-aware so literal ``#`` and ``=``
    characters are preserved. Dependency separators and ``MSG:`` headers are
    structural output and remain in the result.
    """
    compact_lines: list[str] = []
    for raw_line in definition.splitlines(keepends=True):
        body, newline = _split_line_ending(raw_line)
        code = _strip_comment(body).rstrip(" \t")
        if not code or _CONSTANT_RE.match(code):
            continue
        compact_lines.append(code + newline)
    return "".join(compact_lines)


def print_message_definition(definition: str, *, distro: ROS2Distro) -> None:
    """Print a definition as Rich text on a terminal and raw text otherwise."""
    if sys.stdout.isatty():
        console = Console(file=sys.stdout, force_terminal=True)
        console.print(render_schema_definition(definition, distro=distro.value), end="")
    else:
        sys.stdout.write(definition)
        sys.stdout.flush()


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "#":
            return line[:index]
    return line
