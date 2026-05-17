"""Serve a tiny HTML browser for ROS2 message definitions."""

import html
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter

from pymcap_cli.core.msg_resolver import (
    PackageInfo,
    ROS2Distro,
    get_message_text,
    get_package_info,
    list_distro_packages,
    list_package_messages,
)
from pymcap_cli.display.schema_html import render_msg_definition_html
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

MSG_SERVE_OPTIONS_GROUP = Group("Message Serve Options")


_STYLE_CSS = """\
:root {
    color-scheme: light dark;
    --bg: #fafafa;
    --fg: #1a1a1a;
    --muted: #777;
    --link: #1e6fdc;
    --link-hover: #003a8c;
    --border: #e2e2e2;
    --card: #ffffff;
    --primitive: #2e8a3a;
    --constant: #b35900;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0f0f10;
        --fg: #e8e8e8;
        --muted: #9a9a9a;
        --link: #6fb0ff;
        --link-hover: #aacaff;
        --border: #2a2a2a;
        --card: #19191b;
        --primitive: #71c97e;
        --constant: #e8a060;
    }
}
body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font: 15px/1.6 system-ui, sans-serif;
}
.container { max-width: 1100px; margin: 0 auto; padding: clamp(1rem, 4vw, 2.5rem); }
a { color: var(--link); text-decoration: none; }
a:hover { color: var(--link-hover); text-decoration: underline; }
header { margin-bottom: 1.5rem; color: var(--muted); font-size: 0.9rem; }
header .sep { margin: 0 0.5rem; color: var(--border); }
h1 { margin: 0 0 1.5rem; font-size: clamp(1.4rem, 3vw, 1.9rem); }
.subtitle { color: var(--muted); margin: -0.75rem 0 1.5rem; }
.empty { color: var(--muted); text-align: center; padding: 2.5rem 0; }
.grid {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 0.5rem;
    grid-template-columns: repeat(auto-fill, minmax(min(320px, 100%), 1fr));
}
.grid a {
    display: block;
    padding: 0.65rem 0.85rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--card);
    font: 14px/1.4 ui-monospace, monospace;
    overflow-wrap: anywhere;
}
.grid a:hover { border-color: var(--link); text-decoration: none; }
.info-card {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.4rem 1.5rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    margin: 0 0 1.75rem;
}
.info-card dt { color: var(--muted); }
.info-card dd { margin: 0; overflow-wrap: anywhere; }
.info-card code { font: 12.5px/1.4 ui-monospace, monospace; }
.def-wrap { position: relative; }
pre.definition {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    font: 13px/1.55 ui-monospace, monospace;
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.copy-btn {
    position: absolute;
    top: 0.6rem;
    right: 0.6rem;
    padding: 0.3rem 0.7rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg);
    color: var(--muted);
    font: 12px/1 ui-monospace, monospace;
    cursor: pointer;
}
.copy-btn:hover { border-color: var(--link); color: var(--link); }
.comment { color: var(--muted); }
.primitive { color: var(--primitive); }
.field-name { font-weight: 600; }
.constant-name { color: var(--constant); font-weight: 600; }
.constant-equals { color: var(--muted); }
.constant-value { color: var(--constant); }
.bound, .array, .msg-marker { color: var(--muted); }
.separator { color: var(--border); }
.schema-link { border-bottom: 1px dotted; }
"""

_COPY_SCRIPT = """\
<script>
(() => {
  for (const btn of document.querySelectorAll('.copy-btn')) {
    btn.addEventListener('click', async () => {
      const target = document.getElementById(btn.dataset.copyTarget);
      if (!target || !navigator.clipboard) return;
      try {
        await navigator.clipboard.writeText(target.textContent);
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = original; }, 1200);
      } catch (_e) {
        btn.textContent = 'Failed';
      }
    });
  }
})();
</script>
"""


def _page(title: str, breadcrumb: str, content: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{html.escape(title)}</title>\n"
        '<link rel="stylesheet" href="/style.css">\n'
        "</head>\n"
        "<body>\n"
        '<div class="container">\n'
        f"<header>{breadcrumb}</header>\n"
        f"<main>{content}</main>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )


def _breadcrumb(*parts: tuple[str, str | None]) -> str:
    """Build the header breadcrumb from (label, href) pairs.

    ``href=None`` renders the label as plain text (the current page).
    """
    rendered: list[str] = []
    for label, href in parts:
        escaped = html.escape(label)
        if href is None:
            rendered.append(escaped)
        else:
            rendered.append(f'<a href="{html.escape(href, quote=True)}">{escaped}</a>')
    return '<span class="sep">/</span>'.join(rendered)


def _render_index(distro: ROS2Distro) -> tuple[int, str]:
    packages = list_distro_packages(distro=distro)
    if packages is None:
        body = (
            '<p class="empty">Could not load the rosdistro index for '
            f"<strong>{html.escape(distro.value)}</strong>.</p>"
        )
        return 503, _page("packages", _breadcrumb(("packages", None)), body)

    if not packages:
        body = '<p class="empty">No packages found in this distro.</p>'
    else:
        items = "\n".join(
            f'<li><a href="/pkg/{html.escape(p, quote=True)}">{html.escape(p)}</a></li>'
            for p in packages
        )
        body = f'<ul class="grid">\n{items}\n</ul>'

    title = f"ROS2 {distro.value} packages"
    return 200, _page(
        title,
        _breadcrumb(("packages", None)),
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="subtitle">{len(packages)} packages indexed.</p>'
        f"{body}",
    )


def _strip_git_suffix(url: str) -> str:
    return url.removesuffix(".git")


def _github_browse_url(url: str, ref: str | None = None) -> str | None:
    """Convert a ``https://github.com/owner/repo.git`` URL into a browse URL.

    With ``ref``, points at the corresponding ``tree/<ref>`` view.
    Returns ``None`` for non-GitHub URLs.
    """
    if not url.startswith("https://github.com/"):
        return None
    base = _strip_git_suffix(url)
    if ref is None:
        return base
    return f"{base}/tree/{ref}"


def _render_info_card(info: PackageInfo, distro: ROS2Distro) -> str:
    rows: list[tuple[str, str]] = []

    rows.append(("distro", f"<code>{html.escape(distro.value)}</code>"))

    if info.repo_name and info.repo_name != info.name:
        rows.append(("repo", f"<code>{html.escape(info.repo_name)}</code>"))

    if info.source_url:
        repo_url = _github_browse_url(info.source_url)
        tree_url = _github_browse_url(info.source_url, info.source_version)
        label = _strip_git_suffix(info.source_url).removeprefix("https://github.com/")
        ref = info.source_version or ""
        if tree_url:
            inner = (
                f'<a href="{html.escape(tree_url, quote=True)}" target="_blank" rel="noopener">'
                f"<code>{html.escape(label)}</code>"
                f"{' @ ' + html.escape(ref) if ref else ''}"
                "</a>"
            )
        elif repo_url:
            inner = (
                f'<a href="{html.escape(repo_url, quote=True)}" target="_blank" rel="noopener">'
                f"<code>{html.escape(label)}</code></a>"
            )
        else:
            inner = f"<code>{html.escape(info.source_url)}</code>"
        rows.append(("source", inner))

    if info.release_url and info.release_tag:
        tag_url = _github_browse_url(info.release_url, info.release_tag)
        if tag_url:
            inner = (
                f'<a href="{html.escape(tag_url, quote=True)}" target="_blank" rel="noopener">'
                f"<code>{html.escape(info.release_tag)}</code></a>"
            )
        else:
            inner = f"<code>{html.escape(info.release_tag)}</code>"
        rows.append(("release", inner))

    # index.ros.org canonical page for the package.
    rows.append(
        (
            "ros.org",
            f'<a href="https://index.ros.org/p/{html.escape(info.name, quote=True)}/" '
            f'target="_blank" rel="noopener">index.ros.org</a>',
        )
    )

    items = "\n".join(
        f"<dt>{html.escape(label)}</dt><dd>{html_body}</dd>" for label, html_body in rows
    )
    return f'<dl class="info-card">\n{items}\n</dl>'


def _render_pkg(pkg: str, distro: ROS2Distro, extra_paths: tuple[Path, ...]) -> tuple[int, str]:
    names = list_package_messages(pkg, distro=distro, extra_paths=extra_paths)
    info = get_package_info(pkg, distro=distro)

    if names is None and info is None:
        body = (
            f'<p class="empty">Could not resolve package <strong>{html.escape(pkg)}</strong>.</p>'
        )
        return 404, _page(pkg, _breadcrumb(("packages", "/"), (pkg, None)), body)

    sections: list[str] = []
    if info is not None:
        sections.append(_render_info_card(info, distro))

    if names:
        items = "\n".join(
            f'<li><a href="/msg/{html.escape(pkg, quote=True)}/{html.escape(n, quote=True)}">'
            f"{html.escape(n)}</a></li>"
            for n in names
        )
        subtitle = f'<p class="subtitle">{len(names)} message types.</p>'
        sections.append(f'{subtitle}<ul class="grid">\n{items}\n</ul>')
    else:
        sections.append('<p class="empty">This package defines no .msg types.</p>')

    return 200, _page(
        f"{pkg} — messages",
        _breadcrumb(("packages", "/"), (pkg, None)),
        f"<h1>{html.escape(pkg)}</h1>{''.join(sections)}",
    )


def _render_msg(
    pkg: str,
    msg_name: str,
    distro: ROS2Distro,
    extra_paths: tuple[Path, ...],
) -> tuple[int, str]:
    full_type = f"{pkg}/{msg_name}"
    result = get_message_text(full_type, distro=distro, extra_paths=extra_paths)
    if result is None:
        body = (
            '<p class="empty">Could not resolve message '
            f"<strong>{html.escape(full_type)}</strong>.</p>"
        )
        return 404, _page(
            full_type,
            _breadcrumb(("packages", "/"), (pkg, f"/pkg/{pkg}"), (msg_name, None)),
            body,
        )

    msg_text, _deps = result
    highlighted = render_msg_definition_html(msg_text, current_pkg=pkg)
    return 200, _page(
        full_type,
        _breadcrumb(("packages", "/"), (pkg, f"/pkg/{pkg}"), (msg_name, None)),
        f"<h1>{html.escape(full_type)}</h1>"
        f'<div class="def-wrap">'
        f'<button type="button" class="copy-btn" data-copy-target="def">Copy</button>'
        f'<pre id="def" class="definition">{highlighted}</pre>'
        f"</div>"
        f"{_COPY_SCRIPT}",
    )


def _route(
    path: str,
    distro: ROS2Distro,
    extra_paths: tuple[Path, ...],
) -> tuple[int, str, str]:
    """Pure routing layer: return ``(status, content_type, body)``."""
    if path in {"", "/"}:
        status, body = _render_index(distro)
        return status, "text/html; charset=utf-8", body

    if path == "/style.css":
        return 200, "text/css; charset=utf-8", _STYLE_CSS

    if path == "/favicon.ico":
        return 204, "image/x-icon", ""

    if path.startswith("/pkg/"):
        pkg = path[len("/pkg/") :].rstrip("/")
        if not pkg or "/" in pkg:
            return (
                404,
                "text/html; charset=utf-8",
                _page(
                    "not found", _breadcrumb(("packages", "/")), '<p class="empty">Not found.</p>'
                ),
            )
        status, body = _render_pkg(pkg, distro, extra_paths)
        return status, "text/html; charset=utf-8", body

    if path.startswith("/msg/"):
        rest = path[len("/msg/") :].strip("/")
        parts = rest.split("/")
        if len(parts) == 2:
            pkg, msg_name = parts
        elif len(parts) == 3 and parts[1] == "msg":
            pkg, msg_name = parts[0], parts[2]
        else:
            return (
                404,
                "text/html; charset=utf-8",
                _page(
                    "not found", _breadcrumb(("packages", "/")), '<p class="empty">Not found.</p>'
                ),
            )
        status, body = _render_msg(pkg, msg_name, distro, extra_paths)
        return status, "text/html; charset=utf-8", body

    return (
        404,
        "text/html; charset=utf-8",
        _page("not found", _breadcrumb(("packages", "/")), '<p class="empty">Not found.</p>'),
    )


def _make_handler(
    distro: ROS2Distro,
    extra_paths: tuple[Path, ...],
) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, content_type, body = _route(self.path, distro, extra_paths)
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("%s - %s", self.address_string(), format % args)

    return _Handler


def msg_serve(
    *,
    host: Annotated[
        str,
        Parameter(
            name=["--host"],
            group=MSG_SERVE_OPTIONS_GROUP,
            help="Interface to bind the server to.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        Parameter(
            name=["--port", "-p"],
            group=MSG_SERVE_OPTIONS_GROUP,
            help="TCP port to listen on.",
        ),
    ] = 8765,
    distro: Annotated[
        ROS2Distro,
        Parameter(
            name=["--distro", "-d"],
            group=MSG_SERVE_OPTIONS_GROUP,
            help="ROS2 distribution to serve.",
        ),
    ] = ROS2Distro.HUMBLE,
    extra_path: Annotated[
        list[Path],
        Parameter(
            name=["--extra-path", "-I"],
            group=MSG_SERVE_OPTIONS_GROUP,
            help="Additional root paths to search for custom message definitions.",
        ),
    ] = [],  # noqa: B006
    no_browser: Annotated[
        bool,
        Parameter(
            name=["--no-browser"],
            group=MSG_SERVE_OPTIONS_GROUP,
            help="Don't auto-open a browser tab on start.",
            negative="",
        ),
    ] = False,
) -> int:
    """Browse ROS2 packages and message definitions in a local web UI.

    Lists every package in the rosdistro on the index page. Clicking a
    package shows its messages (downloads the release-branch zip on
    first hit); clicking a message renders its definition with
    syntax highlighting and hyperlinked cross-references.
    """
    extra_paths = tuple(extra_path)
    handler_cls = _make_handler(distro, extra_paths)
    url = f"http://{host}:{port}/"

    try:
        httpd = ThreadingHTTPServer((host, port), handler_cls)
    except OSError as exc:
        ERR.print(f"[red]Error:[/red] could not bind to {host}:{port}: {exc}")
        return 1

    ERR.print(f"Serving ROS2 [bold]{distro.value}[/bold] messages on [link={url}]{url}[/link]")
    ERR.print("Press Ctrl-C to stop.")

    if not no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        ERR.print("Stopping.")
    finally:
        httpd.server_close()
    return 0
