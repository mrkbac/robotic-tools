"""Datasette render plugin for the ``pymcap-cli index`` sidecar DB.

Loaded by ``pymcap index serve`` via ``--plugins-dir`` (run inside the project env
so ``pymcap_cli`` imports resolve). Humanizes columns Datasette would otherwise
show raw:

  * ``*_ns`` (sane timestamps)  -> UTC timestamp
  * ``duration_ns`` / ``duration`` -> ``1h2m`` style
  * ``*_bytes``                 -> human size           (reuses bytes_to_human)
  * ``error_kind``              -> explanatory label
  * ``distribution_blob``       -> unicode sparkline     (reuses unpack_distribution_blob)
  * ``metadata_json_zlib``      -> inflated + pretty JSON
"""

from __future__ import annotations

# Datasette discovers plugins by file under --plugins-dir; no package __init__.py.
# ruff: noqa: INP001
import json
import zlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from datasette import hookimpl
from markupsafe import Markup, escape
from pymcap_cli.index import SANE_EPOCH_NS
from pymcap_cli.index.scanner import unpack_distribution_blob
from pymcap_cli.utils import bytes_to_human

if TYPE_CHECKING:
    import sqlite3

# Mirror _helpers._MAX_PLAUSIBLE_DURATION_NS: spans beyond this are clock skew, not real.
_MAX_PLAUSIBLE_DURATION_NS = 30 * 24 * 60 * 60 * 1_000_000_000

# Mirror _helpers._ERROR_KIND_LABEL.
_ERROR_KIND_LABEL = {
    "io": "io — file could not be read",
    "corrupt": "corrupt — not a valid MCAP",
    "no_summary": "no_summary — no usable summary; rerun scan with --rebuild-missing",
}

_SPARK = "▁▂▃▄▅▆▇█"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@hookimpl
def prepare_connection(conn: sqlite3.Connection, database: str) -> None:
    if database != "index":
        return
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA mmap_size=268435456")


def _link_params(params_value: JsonValue | None) -> dict[str, str] | None:
    if params_value is None:
        return {}
    if not isinstance(params_value, dict):
        return None

    params: dict[str, str] = {}
    for key, value in params_value.items():
        if value is None:
            continue
        if isinstance(value, bool):
            params[key] = "true" if value else "false"
            continue
        if isinstance(value, (str, int, float)):
            params[key] = str(value)
            continue
        return None
    return params


def _json_link(value: str) -> Markup | None:
    stripped = value.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        data: JsonValue = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    href_value = data.get("href")
    label_value = data.get("label")
    params = _link_params(data.get("params"))
    if not isinstance(href_value, str) or not href_value.startswith("/") or params is None:
        return None
    if isinstance(label_value, (dict, list)):
        return None

    query = urlencode(params)
    href = href_value if not query else f"{href_value}{'&' if '?' in href_value else '?'}{query}"
    label = "" if label_value is None else str(label_value)
    escaped_label = escape(label) if label else Markup("&nbsp;")
    # href and label are escaped before interpolation, so this is safe HTML.
    return Markup(f'<a href="{escape(href)}">{escaped_label}</a>')  # noqa: S704


def _sparkline(counts: list[int]) -> Markup:
    if not counts:
        return Markup('<span style="opacity:.5">empty</span>')
    peak = max(counts) or 1
    bars = "".join(_SPARK[min(len(_SPARK) - 1, c * (len(_SPARK) - 1) // peak)] for c in counts)
    title = f"{len(counts)} bins · peak {peak:,}"
    # bars is built from the _SPARK literal and title is escaped, so this is safe HTML.
    return Markup(f'<span title="{escape(title)}" style="font-size:1.2em">{bars}</span>')  # noqa: S704


def _format_duration_ns(span: int) -> str | None:
    if span <= 0 or span > _MAX_PLAUSIBLE_DURATION_NS:
        return None  # leave raw — matches the UI dimming bogus spans
    secs = span / 1e9
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        m, s = divmod(int(secs), 60)
        return f"{m}m{s}s"
    if secs < 86400:
        h, rem = divmod(int(secs), 3600)
        return f"{h}h{rem // 60}m"
    d, rem = divmod(int(secs), 86400)
    return f"{d}d{rem // 3600}h"


@hookimpl
def render_cell(value: str | float | bytes | None, column: str) -> str | Markup | None:
    if value is None:
        return None

    if isinstance(value, str):
        link = _json_link(value)
        if link is not None:
            return link

    if column == "distribution_blob" and isinstance(value, bytes):
        return _sparkline(unpack_distribution_blob(value) or [])

    if column == "metadata_json_zlib" and isinstance(value, bytes):
        text = zlib.decompress(value).decode("utf-8")
        pretty = json.dumps(json.loads(text), indent=2, sort_keys=True)
        # The inflated JSON is escaped before interpolation, so this is safe HTML.
        return Markup(f"<pre style='margin:0;white-space:pre-wrap'>{escape(pretty)}</pre>")  # noqa: S704

    if column == "error_kind" and isinstance(value, str):
        return _ERROR_KIND_LABEL.get(value, value)

    # Duration must be checked before the generic ``_ns`` timestamp branch.
    if column in ("duration_ns", "duration") and isinstance(value, int):
        return _format_duration_ns(value)

    if column.endswith("_ns") and isinstance(value, int) and value >= SANE_EPOCH_NS:
        dt = datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
        return f"{dt:%Y-%m-%d %H:%M:%S}Z"

    if column.endswith("_bytes") and isinstance(value, int):
        return bytes_to_human(value)

    return None
