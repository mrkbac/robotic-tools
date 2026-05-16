"""Rich rendering helpers for ROS message definitions."""

from __future__ import annotations

import re

from rich.text import Text

from pymcap_cli.display.display_utils import _create_ros_docs_url, _text_to_color

_PRIMITIVE_TYPES = frozenset(
    {
        "bool",
        "byte",
        "char",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "string",
        "wstring",
    }
)

_FIRST_TOKEN_RE = re.compile(r"(?P<type>\S+)(?P<space>\s*)(?P<rest>.*)")
_ARRAY_SUFFIX_RE = re.compile(r"(?P<base>.+?)(?P<array>\[(?:<=)?\d*\])$")
_STRING_BOUND_RE = re.compile(r"(?P<base>w?string)(?P<bound><=\d+)$")


def format_schema_name(schema_name: str, *, distro: str = "jazzy") -> Text:
    """Format a ROS schema/type path using deterministic per-segment colors."""
    if not schema_name:
        return Text("unknown", style="dim")

    parts = schema_name.strip("/").split("/")
    if not parts or parts == [""]:
        return Text(schema_name)

    url = _create_ros_docs_url(schema_name, distro) if "/" in schema_name else ""
    link_style = f" link {url}" if url else ""

    text = Text()
    if schema_name.startswith("/"):
        text.append("/", style=f"dim{link_style}")

    for index, part in enumerate(parts):
        if index:
            text.append("/", style=f"dim{link_style}")
        text.append(part, style=f"{_text_to_color(part)}{link_style}")

    return text


def render_schema_definition(definition: str, *, distro: str = "jazzy") -> Text:
    """Render a resolved ``.msg`` definition with Rich styles.

    The returned ``Text.plain`` is exactly ``definition``, so callers can safely
    choose raw output for pipes and styled output for terminals.
    """
    rendered = Text()
    for line in definition.splitlines(keepends=True):
        body, newline = _split_line_ending(line)
        _append_rendered_line(rendered, body, distro=distro)
        rendered.append(newline)
    return rendered


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _append_rendered_line(rendered: Text, line: str, *, distro: str) -> None:
    if not line:
        return

    stripped = line.strip()
    if stripped and set(stripped) == {"="}:
        rendered.append(line, style="dim")
        return

    leading_len = len(line) - len(line.lstrip(" \t"))
    rendered.append(line[:leading_len])
    content = line[leading_len:]

    if content.startswith("#"):
        rendered.append(content, style="dim")
        return

    if content.startswith("MSG:"):
        _append_msg_header(rendered, content, distro=distro)
        return

    code, comment = _split_comment(content)
    if not code.strip():
        rendered.append(code)
        rendered.append(comment, style="dim")
        return

    trailing_len = len(code) - len(code.rstrip(" \t"))
    code_body = code[: len(code) - trailing_len] if trailing_len else code
    trailing = code[len(code_body) :]

    match = _FIRST_TOKEN_RE.fullmatch(code_body)
    if match is None:
        rendered.append(code)
        rendered.append(comment, style="dim")
        return

    _append_type(rendered, match.group("type"), distro=distro)
    rendered.append(match.group("space"))
    _append_field_or_constant(rendered, match.group("rest"))
    rendered.append(trailing)
    rendered.append(comment, style="dim")


def _append_msg_header(rendered: Text, content: str, *, distro: str) -> None:
    marker = "MSG:"
    rendered.append(marker, style="bold magenta")
    rest = content[len(marker) :]
    leading_len = len(rest) - len(rest.lstrip(" \t"))
    rendered.append(rest[:leading_len])
    rest_body = rest[leading_len:]
    type_name, space, tail = rest_body.partition(" ")
    if not type_name:
        return
    rendered.append(format_schema_name(type_name, distro=distro))
    rendered.append(space)
    rendered.append(tail)


def _split_comment(content: str) -> tuple[str, str]:
    code, marker, comment = content.partition("#")
    if not marker:
        return content, ""
    return code, f"{marker}{comment}"


def _append_type(rendered: Text, type_token: str, *, distro: str) -> None:
    base, array_suffix = _split_array_suffix(type_token)
    base, bound_suffix = _split_string_bound(base)

    if base in _PRIMITIVE_TYPES:
        rendered.append(base, style="green")
    elif "/" in base:
        rendered.append(format_schema_name(base, distro=distro))
    else:
        rendered.append(base, style="cyan")

    if bound_suffix:
        rendered.append(bound_suffix, style="yellow")
    if array_suffix:
        rendered.append(array_suffix, style="yellow")


def _split_array_suffix(type_token: str) -> tuple[str, str]:
    match = _ARRAY_SUFFIX_RE.fullmatch(type_token)
    if match is None:
        return type_token, ""
    return match.group("base"), match.group("array")


def _split_string_bound(type_token: str) -> tuple[str, str]:
    match = _STRING_BOUND_RE.fullmatch(type_token)
    if match is None:
        return type_token, ""
    return match.group("base"), match.group("bound")


def _append_field_or_constant(rendered: Text, rest: str) -> None:
    name, equals, value = rest.partition("=")
    if not equals:
        rendered.append(rest, style="bold white" if rest else "")
        return

    rendered.append(name, style="bold magenta")
    rendered.append(equals, style="dim")
    rendered.append(value, style="yellow")
