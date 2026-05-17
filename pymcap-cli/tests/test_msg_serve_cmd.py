"""Tests for `pymcap-cli msg serve`."""

from __future__ import annotations

import threading
import urllib.request
from contextlib import closing
from http.server import ThreadingHTTPServer

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd.msg import serve_cmd as msg_serve_cmd
from pymcap_cli.cmd.msg.serve_cmd import _make_handler, _route
from pymcap_cli.core.msg_resolver import PackageInfo, ROS2Distro
from pymcap_cli.display.schema_html import render_msg_definition_html


class TestRenderHtml:
    def test_primitive_field_gets_primitive_span(self) -> None:
        out = render_msg_definition_html("int32 height\n")
        assert '<span class="primitive">int32</span>' in out
        assert '<span class="field-name">height</span>' in out

    def test_cross_reference_becomes_link(self) -> None:
        out = render_msg_definition_html("std_msgs/Header header\n")
        assert '<a class="schema-link" href="/msg/std_msgs/Header">' in out
        assert "std_msgs/Header" in out
        assert '<span class="field-name">header</span>' in out

    def test_unqualified_type_links_to_current_pkg(self) -> None:
        out = render_msg_definition_html("VelodynePacket[] packets\n", current_pkg="velodyne_msgs")
        assert '<a class="schema-link" href="/msg/velodyne_msgs/VelodynePacket">' in out
        assert ">VelodynePacket<" in out  # display stays unqualified
        assert '<span class="array">[]</span>' in out

    def test_unqualified_type_without_current_pkg_has_no_link(self) -> None:
        out = render_msg_definition_html("VelodynePacket[] packets\n")
        assert '<span class="custom-type">VelodynePacket</span>' in out
        assert "schema-link" not in out

    def test_legacy_time_builtin_is_not_linked(self) -> None:
        out = render_msg_definition_html("time stamp\n", current_pkg="some_pkg")
        assert '<span class="custom-type">time</span>' in out
        assert "schema-link" not in out

    def test_array_and_bound_suffixes_styled(self) -> None:
        out = render_msg_definition_html("string<=32[2] data\n")
        assert '<span class="primitive">string</span>' in out
        assert '<span class="bound">&lt;=32</span>' in out
        assert '<span class="array">[2]</span>' in out

    def test_constants_styled(self) -> None:
        out = render_msg_definition_html("uint8 OK=0\n")
        assert '<span class="primitive">uint8</span>' in out
        assert '<span class="constant-name">OK</span>' in out
        assert '<span class="constant-equals">=</span>' in out
        assert '<span class="constant-value">0</span>' in out

    def test_comments_styled(self) -> None:
        out = render_msg_definition_html("# a header comment\nint32 x  # inline\n")
        assert '<span class="comment"># a header comment</span>' in out
        assert '<span class="comment"># inline</span>' in out

    def test_msg_header_separator(self) -> None:
        text = (
            "uint8 x\n========================================\nMSG: std_msgs/Header\nuint32 seq\n"
        )
        out = render_msg_definition_html(text)
        assert '<span class="separator">========================================</span>' in out
        assert '<span class="msg-marker">MSG:</span>' in out
        # Header in the MSG: line should be a hyperlink.
        assert 'href="/msg/std_msgs/Header"' in out

    def test_html_escaping_of_text_chars(self) -> None:
        out = render_msg_definition_html("# 1 < 2 & 3 > 0\n")
        assert "1 &lt; 2 &amp; 3 &gt; 0" in out


class TestRouting:
    def test_index_lists_packages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            msg_serve_cmd, "list_distro_packages", lambda **_kw: ["sensor_msgs", "std_msgs"]
        )
        status, ctype, body = _route("/", ROS2Distro.HUMBLE, ())
        assert status == 200
        assert ctype.startswith("text/html")
        assert '<a href="/pkg/sensor_msgs">sensor_msgs</a>' in body
        assert '<a href="/pkg/std_msgs">std_msgs</a>' in body
        assert "2 packages indexed" in body

    def test_index_503_when_distro_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(msg_serve_cmd, "list_distro_packages", lambda **_kw: None)
        status, _ctype, body = _route("/", ROS2Distro.HUMBLE, ())
        assert status == 503
        assert "Could not load the rosdistro index" in body

    def test_pkg_page_lists_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            msg_serve_cmd,
            "list_package_messages",
            lambda _package_name, **_kw: ["Image", "PointCloud2"],
        )
        monkeypatch.setattr(msg_serve_cmd, "get_package_info", lambda _pkg, **_kw: None)
        status, _ctype, body = _route("/pkg/sensor_msgs", ROS2Distro.HUMBLE, ())
        assert status == 200
        assert '<a href="/msg/sensor_msgs/Image">Image</a>' in body
        assert '<a href="/msg/sensor_msgs/PointCloud2">PointCloud2</a>' in body

    def test_pkg_page_empty_pkg_shows_info_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            msg_serve_cmd,
            "list_package_messages",
            lambda _package_name, **_kw: [],
        )
        monkeypatch.setattr(
            msg_serve_cmd,
            "get_package_info",
            lambda _pkg, **_kw: PackageInfo(
                name="off_highway_mm7p10",
                repo_name="off_highway_sensor_drivers",
                source_url="https://github.com/bosch-engineering/off_highway_sensor_drivers.git",
                source_version="humble-devel",
                release_url="https://github.com/ros2-gbp/off_highway_sensor_drivers-release.git",
                release_version="0.12.0-1",
                release_tag="release/humble/off_highway_mm7p10/0.12.0-1",
            ),
        )
        status, _ctype, body = _route("/pkg/off_highway_mm7p10", ROS2Distro.HUMBLE, ())
        assert status == 200
        assert "defines no .msg types" in body
        # repo metadata
        assert "off_highway_sensor_drivers" in body
        # source repo link, with branch ref
        assert (
            'href="https://github.com/bosch-engineering/off_highway_sensor_drivers/tree/humble-devel"'
            in body
        )
        # release tag link
        assert (
            'href="https://github.com/ros2-gbp/off_highway_sensor_drivers-release'
            '/tree/release/humble/off_highway_mm7p10/0.12.0-1"' in body
        )
        # index.ros.org link
        assert 'href="https://index.ros.org/p/off_highway_mm7p10/"' in body

    def test_pkg_page_404_for_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            msg_serve_cmd,
            "list_package_messages",
            lambda _package_name, **_kw: None,
        )
        monkeypatch.setattr(msg_serve_cmd, "get_package_info", lambda _pkg, **_kw: None)
        status, _ctype, body = _route("/pkg/missing_msgs", ROS2Distro.HUMBLE, ())
        assert status == 404
        assert "Could not resolve package" in body

    def test_msg_page_renders_definition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            msg_serve_cmd,
            "get_message_text",
            lambda _msg_type, **_kw: ("std_msgs/Header header\nuint8[] data\n", []),
        )
        status, _ctype, body = _route("/msg/sensor_msgs/Image", ROS2Distro.HUMBLE, ())
        assert status == 200
        assert 'class="definition"' in body
        assert 'href="/msg/std_msgs/Header"' in body
        assert 'class="copy-btn"' in body

    def test_msg_page_accepts_pkg_msg_typename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The /msg/<pkg>/msg/<Name> form must also work for symmetry."""
        monkeypatch.setattr(
            msg_serve_cmd,
            "get_message_text",
            lambda _msg_type, **_kw: ("uint32 x\n", []),
        )
        status, _ctype, _body = _route("/msg/sensor_msgs/msg/Image", ROS2Distro.HUMBLE, ())
        assert status == 200

    def test_msg_page_404_for_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            msg_serve_cmd,
            "get_message_text",
            lambda _msg_type, **_kw: None,
        )
        status, _ctype, body = _route("/msg/sensor_msgs/Nope", ROS2Distro.HUMBLE, ())
        assert status == 404
        assert "Could not resolve message" in body

    def test_style_css_is_served(self) -> None:
        status, ctype, body = _route("/style.css", ROS2Distro.HUMBLE, ())
        assert status == 200
        assert ctype.startswith("text/css")
        assert "--bg" in body  # CSS variable from the theme
        assert "prefers-color-scheme" in body  # dark mode hook
        assert "auto-fill" in body  # responsive grid for both pages
        assert "info-card" in body  # info card for package pages
        assert "pre-wrap" in body  # wraps long .msg comment lines

    def test_unknown_path_is_404(self) -> None:
        status, _ctype, body = _route("/nope", ROS2Distro.HUMBLE, ())
        assert status == 404
        assert "Not found." in body


def test_full_http_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind to an ephemeral port and verify the index page comes back."""
    monkeypatch.setattr(msg_serve_cmd, "list_distro_packages", lambda **_kw: ["std_msgs"])

    handler_cls = _make_handler(ROS2Distro.HUMBLE, ())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with closing(urllib.request.urlopen(f"http://127.0.0.1:{port}/")) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
        assert "std_msgs" in body
        assert "<!doctype html>" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_msg_serve_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["msg", "serve", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli msg serve" in output
