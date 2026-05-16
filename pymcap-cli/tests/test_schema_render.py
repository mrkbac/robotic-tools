"""Tests for ROS message definition rendering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.display.schema_render import format_schema_name, render_schema_definition

if TYPE_CHECKING:
    from rich.text import Text


def _styled_fragments(text: Text) -> list[tuple[str, str]]:
    return [(text.plain[span.start : span.end], str(span.style)) for span in text.spans]


def test_format_schema_name_splits_and_links_ros_schema() -> None:
    rendered = format_schema_name("sensor_msgs/msg/Image", distro="humble")

    assert rendered.plain == "sensor_msgs/msg/Image"
    fragments = _styled_fragments(rendered)
    assert any(
        fragment == "sensor_msgs"
        and style.endswith("link https://docs.ros.org/en/humble/p/sensor_msgs/msg/Image.html")
        for fragment, style in fragments
    )
    assert (
        "/",
        "dim link https://docs.ros.org/en/humble/p/sensor_msgs/msg/Image.html",
    ) in fragments


def test_render_schema_definition_preserves_plain_text() -> None:
    definition = (
        "uint32 height # pixels\n"
        "sensor_msgs/msg/Image[] images\n"
        "string<=8 NAME=front\n"
        "========================================\n"
        "MSG: std_msgs/msg/Header\n"
    )

    rendered = render_schema_definition(definition, distro="humble")

    assert rendered.plain == definition


def test_render_schema_definition_styles_schema_tokens() -> None:
    rendered = render_schema_definition(
        (
            "uint32 height # pixels\n"
            "sensor_msgs/msg/Image[] images\n"
            "string<=8 NAME=front\n"
            "========================================\n"
            "MSG: std_msgs/msg/Header\n"
        ),
        distro="humble",
    )

    fragments = _styled_fragments(rendered)
    assert ("uint32", "green") in fragments
    assert ("# pixels", "dim") in fragments
    assert ("[]", "yellow") in fragments
    assert ("<=", "yellow") not in fragments
    assert ("<=8", "yellow") in fragments
    assert ("NAME", "bold magenta") in fragments
    assert ("front", "yellow") in fragments
    assert ("========================================", "dim") in fragments
    assert ("MSG:", "bold magenta") in fragments
    assert any(
        fragment == "std_msgs"
        and style.endswith("link https://docs.ros.org/en/humble/p/std_msgs/msg/Header.html")
        for fragment, style in fragments
    )
