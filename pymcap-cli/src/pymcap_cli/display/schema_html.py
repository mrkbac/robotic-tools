"""HTML rendering of ROS2 ``.msg`` definitions.

Mirrors :mod:`pymcap_cli.display.schema_render` (which produces Rich
``Text``) but emits ``<span class="...">`` tokens, and turns every
``<pkg>/<Type>`` reference into a hyperlink so the ``msg serve`` UI
can drill in.
"""

from __future__ import annotations

import html
import re

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

# ROS1 legacy syntax still seen in some .msg files. The parser resolves
# these to ``builtin_interfaces/Time`` / ``Duration``; treat them as
# unlinkable so we don't generate dead ``/msg/<pkg>/time`` URLs.
_BUILTIN_UNQUALIFIED = frozenset({"time", "duration"})

_FIRST_TOKEN_RE = re.compile(r"(?P<type>\S+)(?P<space>\s*)(?P<rest>.*)")
_ARRAY_SUFFIX_RE = re.compile(r"(?P<base>.+?)(?P<array>\[(?:<=)?\d*\])$")
_STRING_BOUND_RE = re.compile(r"(?P<base>w?string)(?P<bound><=\d+)$")


def render_msg_definition_html(definition: str, *, current_pkg: str | None = None) -> str:
    """Render a resolved ``.msg`` definition as syntax-highlighted HTML.

    Fully-qualified ``<pkg>/<Type>`` references always become anchors
    pointing at ``/msg/<pkg>/<Type>``. Unqualified custom types (like
    ``VelodynePacket``) are resolved relative to ``current_pkg`` when
    provided — ROS2 .msg semantics: unqualified names refer to the
    same package.
    """
    out: list[str] = []
    for line in definition.splitlines(keepends=True):
        body, newline = _split_line_ending(line)
        _render_line(out, body, current_pkg=current_pkg)
        out.append(html.escape(newline))
    return "".join(out)


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _render_line(out: list[str], line: str, *, current_pkg: str | None) -> None:
    if not line:
        return

    stripped = line.strip()
    if stripped and set(stripped) == {"="}:
        out.append(_span("separator", line))
        return

    leading_len = len(line) - len(line.lstrip(" \t"))
    out.append(html.escape(line[:leading_len]))
    content = line[leading_len:]

    if content.startswith("#"):
        out.append(_span("comment", content))
        return

    if content.startswith("MSG:"):
        _render_msg_header(out, content)
        return

    code, comment = _split_comment(content)
    if not code.strip():
        out.append(html.escape(code))
        if comment:
            out.append(_span("comment", comment))
        return

    trailing_len = len(code) - len(code.rstrip(" \t"))
    code_body = code[: len(code) - trailing_len] if trailing_len else code
    trailing = code[len(code_body) :]

    match = _FIRST_TOKEN_RE.fullmatch(code_body)
    if match is None:
        out.append(html.escape(code))
        if comment:
            out.append(_span("comment", comment))
        return

    _render_type(out, match.group("type"), current_pkg=current_pkg)
    out.append(html.escape(match.group("space")))
    _render_field_or_constant(out, match.group("rest"))
    out.append(html.escape(trailing))
    if comment:
        out.append(_span("comment", comment))


def _render_msg_header(out: list[str], content: str) -> None:
    marker = "MSG:"
    out.append(_span("msg-marker", marker))
    rest = content[len(marker) :]
    leading_len = len(rest) - len(rest.lstrip(" \t"))
    out.append(html.escape(rest[:leading_len]))
    rest_body = rest[leading_len:]
    type_name, space, tail = rest_body.partition(" ")
    if not type_name:
        return
    out.append(_render_schema_name(type_name))
    out.append(html.escape(space))
    out.append(html.escape(tail))


def _split_comment(content: str) -> tuple[str, str]:
    code, marker, comment = content.partition("#")
    if not marker:
        return content, ""
    return code, f"{marker}{comment}"


def _render_type(out: list[str], type_token: str, *, current_pkg: str | None) -> None:
    base, array_suffix = _split_array_suffix(type_token)
    base, bound_suffix = _split_string_bound(base)

    if base in _PRIMITIVE_TYPES:
        out.append(_span("primitive", base))
    elif "/" in base:
        out.append(_render_schema_name(base))
    elif current_pkg and base not in _BUILTIN_UNQUALIFIED:
        out.append(_render_schema_name(f"{current_pkg}/{base}", display=base))
    else:
        out.append(_span("custom-type", base))

    if bound_suffix:
        out.append(_span("bound", bound_suffix))
    if array_suffix:
        out.append(_span("array", array_suffix))


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


def _render_field_or_constant(out: list[str], rest: str) -> None:
    name, equals, value = rest.partition("=")
    if not equals:
        if rest:
            out.append(_span("field-name", rest))
        return

    out.append(_span("constant-name", name))
    out.append(_span("constant-equals", equals))
    out.append(_span("constant-value", value))


def _render_schema_name(schema_name: str, *, display: str | None = None) -> str:
    """Render a ``pkg/Type`` reference as a hyperlink to its page.

    ``display`` overrides the visible text (e.g. to keep an unqualified
    rendering while linking to the fully-qualified URL).
    """
    parts = schema_name.split("/")
    if len(parts) == 2:
        pkg, type_name = parts
    elif len(parts) == 3 and parts[1] == "msg":
        pkg, type_name = parts[0], parts[2]
    else:
        return _span("custom-type", display or schema_name)

    if not pkg or not type_name:
        return _span("custom-type", display or schema_name)

    href = f"/msg/{html.escape(pkg, quote=True)}/{html.escape(type_name, quote=True)}"
    text = html.escape(display) if display is not None else html.escape(schema_name)
    return f'<a class="schema-link" href="{href}">{text}</a>'


def _span(css_class: str, text: str) -> str:
    return f'<span class="{css_class}">{html.escape(text)}</span>'
