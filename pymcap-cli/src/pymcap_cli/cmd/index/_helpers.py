"""Shared helpers, constants, and SQL fragments for ``pymcap-cli index``."""

from __future__ import annotations

import importlib.metadata
import json as _json
import os
import sqlite3
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from rich.console import Console

from pymcap_cli.display.display_utils import ChannelTableData, _text_to_color
from pymcap_cli.index import SANE_EPOCH_NS
from pymcap_cli.index.db import (
    CURRENT_SCHEMA_VERSION,
    IndexDbNeedsMigrationError,
    default_db_path,
)
from pymcap_cli.utils import bytes_to_human, parse_time_arg

if TYPE_CHECKING:
    from pymcap_cli.types.info_types import ChannelInfo, SchemaInfo

OutputFormat = Literal["table", "json", "paths-only"]

console = Console()


_ERROR_KIND_LABEL: dict[str, str] = {
    "io": "io — file could not be read",
    "corrupt": "corrupt — not a valid MCAP",
    "no_summary": "no_summary — no usable summary; rerun scan with --rebuild-missing",
}


# Second sanity gate beyond ``SANE_EPOCH_NS``: even when both
# ``message_start_time`` and ``message_end_time`` are post-epoch, the *span*
# between them can still be bogus when two channels publish on desynced clocks
# (e.g. GPS stuck at 2011, system clock at 2017, both messages legitimately
# written into the same MCAP in 2024 — chunk-level MIN/MAX can't recover from
# that because every chunk contains a mix). 30 days comfortably catches multi-
# year spans while letting unusually-long single-recording datasets through.
_MAX_PLAUSIBLE_DURATION_NS = 30 * 24 * 60 * 60 * 1_000_000_000

# Pick whichever value (raw vs precomputed sane fallback) is post-epoch. The
# ``sane_message_*`` columns are populated by the scanner from
# ``ChunkIndex`` records, with the same threshold applied (see
# ``scanner._build_content_row``).
_EFF_START_SQL = (
    f"CASE WHEN c.message_start_time_ns >= {SANE_EPOCH_NS} "
    "THEN c.message_start_time_ns ELSE c.sane_message_start_time_ns END"
)
_EFF_END_SQL = (
    f"CASE WHEN c.message_start_time_ns >= {SANE_EPOCH_NS} "
    "THEN c.message_end_time_ns ELSE c.sane_message_end_time_ns END"
)


def _format_compression_cell(
    compression: str | None,
    compressed: int | None,
    uncompressed: int | None,
) -> str:
    """Human-readable summary for the per-content compression aggregates."""
    if compression is None:
        # Pre-0005 content row, or a ``scan_kind = 'rebuilt'`` file with no
        # chunk indexes — we genuinely don't know.
        return "[dim]-[/]"
    if compression in ("", "none"):
        # Chunks exist, none compress. Still show the byte total so the user
        # can confirm the size on disk matches the message payload.
        size = (
            bytes_to_human(uncompressed) if isinstance(uncompressed, int) and uncompressed else "-"
        )
        return f"[dim]none[/]  [yellow]{size}[/]"
    ratio_part = ""
    if (
        isinstance(uncompressed, int)
        and isinstance(compressed, int)
        and compressed > 0
        and uncompressed > 0
    ):
        ratio = uncompressed / compressed
        ratio_part = (
            f"  [cyan]{ratio:.2f}x[/]"
            f"  [dim]({bytes_to_human(uncompressed)} -> {bytes_to_human(compressed)})[/]"
        )
    # ``compression`` is a comma-joined sorted set of codec names. Single
    # codec -> ``"zstd"``; mixed -> ``"lz4,zstd"`` or ``"none,zstd"`` when some
    # chunks are uncompressed. We render verbatim so the user can see the mix.
    codec_label = (
        f"[green]{compression}[/]"
        if "," not in compression
        else f"[yellow]mixed[/] [dim]({compression})[/]"
    )
    return f"{codec_label}{ratio_part}"


def _safe_duration_ns(start_ns: int | None, end_ns: int | None) -> int | None:
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return None
    if start_ns < SANE_EPOCH_NS:
        return None
    span = end_ns - start_ns
    if span > _MAX_PLAUSIBLE_DURATION_NS:
        return None
    return span


_SHORT_ID_LEN = 8
_FULL_FP_LEN = len("s1:") + 32  # ``s1:`` + 32 hex chars


def _short_id_from_fingerprint(fp: str) -> str:
    """Return the first ``_SHORT_ID_LEN`` hex chars after the ``s1:`` scheme prefix.

    Empty string when ``fp`` doesn't start with a recognised scheme — callers
    treat that as "unknown" and render ``-``.
    """
    if fp.startswith("s1:"):
        return fp[3 : 3 + _SHORT_ID_LEN]
    return ""


def _resolve_target_to_summary_fp(
    conn: sqlite3.Connection, target: str
) -> tuple[str | None, str | None, str | None]:
    """Resolve a user-supplied target to a summary fingerprint.

    Returns ``(summary_fp, abs_path, error_message)`` — exactly one of
    ``summary_fp`` or ``error_message`` is set. ``abs_path`` is filled in
    when the target resolved as a filesystem path so the caller can render
    it in the Identity table.

    The target can be:

    - A filesystem path (exists, or contains ``os.sep``).
    - A full ``s1:...`` fingerprint.
    - A short hex prefix, with or without the ``s1:`` scheme. Case-insensitive.
      Resolved via ``LIKE`` against ``content.summary_fingerprint``; ambiguity
      surfaces as an error listing candidates.
    """
    candidate = Path(target).expanduser()
    if candidate.exists() or os.sep in target:
        abs_path = str(candidate.resolve())
        row = conn.execute(
            "SELECT summary_fingerprint FROM current_file WHERE abs_path = ?",
            (abs_path,),
        ).fetchone()
        if row is None or row[0] is None:
            return None, abs_path, f"no indexed content for {abs_path}"
        return row[0], abs_path, None

    if len(target) == _FULL_FP_LEN and target.startswith("s1:"):
        row = conn.execute(
            "SELECT 1 FROM content WHERE summary_fingerprint = ?", (target,)
        ).fetchone()
        if row is None:
            return None, None, f"unknown summary fingerprint {target}"
        return target, None, None

    prefix = target.lower().removeprefix("s1:")
    if not prefix or not all(c in "0123456789abcdef" for c in prefix):
        return (
            None,
            None,
            f"unrecognised target '{target}' - expected path, fingerprint, or short id",
        )
    matches = conn.execute(
        "SELECT summary_fingerprint FROM content WHERE summary_fingerprint LIKE ? LIMIT 5",
        (f"s1:{prefix}%",),
    ).fetchall()
    if not matches:
        return None, None, f"no content matching id '{target}'"
    if len(matches) > 1:
        candidates = ", ".join(row[0][: 3 + _SHORT_ID_LEN + 4] + "…" for row in matches)
        return None, None, f"id '{target}' is ambiguous ({len(matches)} matches): {candidates}"
    return matches[0][0], None, None


def _pymcap_cli_version() -> str:
    try:
        return importlib.metadata.version("pymcap-cli")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class _IndexedTopicPayload(TypedDict):
    channel_id: int
    topic: str
    schema_pk_id: int | None
    schema: str | None
    encoding: str | None
    message_count: int | None
    uncompressed_size_bytes: int | None
    message_start_time_ns: int | None
    message_end_time_ns: int | None
    duration_ns: int | None
    distribution: list[int] | None


def _topics_to_channel_table_data(
    topics_payload: Sequence[_IndexedTopicPayload],
    schema_dim_rows: Sequence[tuple[int, str | None, str | None, int | None]],
    file_duration_ns: int | None,
) -> ChannelTableData:
    """Adapt index-DB rows into the channel-table shape.

    The standalone ``pymcap-cli info`` command builds this dict from a fresh
    file read; we build the same shape from cached index rows so the channel
    table looks identical without duplicating the renderer.

    Each channel's ``schema_id`` is the index's ``schema_pk_id`` (or ``-1``
    when the channel has no schema), and we expose one ``SchemaInfo`` row per
    distinct ``schema_pk_id`` so the renderer's ``schema_map`` lookup hits.
    """
    schemas: list[SchemaInfo] = [
        {"id": pk, "name": name or "", "encoding": enc or "", "data": ""}
        for pk, name, enc, _sz in schema_dim_rows
    ]
    channels: list[ChannelInfo] = [
        {
            "id": entry["channel_id"],
            "topic": entry["topic"],
            "schema_id": entry["schema_pk_id"] if entry["schema_pk_id"] is not None else -1,
            "message_count": entry["message_count"] or 0,
            "size_bytes": entry["uncompressed_size_bytes"],
            "duration_ns": entry["duration_ns"],
            "message_start_time": entry["message_start_time_ns"],
            "message_end_time": entry["message_end_time_ns"],
            "message_distribution": entry["distribution"] or [],
            "estimated_sizes": True,
        }
        for entry in topics_payload
    ]
    return {
        "statistics": {"duration_ns": file_duration_ns or 0},
        "schemas": schemas,
        "channels": channels,
    }


def _resolve_db(db: Path | None) -> Path:
    return (db or default_db_path()).expanduser()


def _print_db_needs_migration(exc: IndexDbNeedsMigrationError) -> None:
    console.print(
        "[red]Error:[/] "
        f"index DB at {exc.db_path} is schema v{exc.current_version}; "
        f"this pymcap-cli expects v{exc.expected_version}. "
        f"Run `pymcap-cli index scan <path> --db {exc.db_path}` to migrate it."
    )


def _path_prefix_where(path: Path) -> tuple[str, tuple[str | int, ...]]:
    """Build a prefix-match WHERE clause that is safe against LIKE wildcards.

    Uses ``substr(abs_path, 1, ?) = ?`` so paths containing ``_`` or ``%`` do
    not accidentally match unrelated siblings.
    """
    resolved = str(path.expanduser().resolve())
    child_prefix = resolved if resolved.endswith(os.sep) else f"{resolved}{os.sep}"
    return (
        "WHERE (abs_path = ? OR substr(abs_path, 1, ?) = ?)",
        (resolved, len(child_prefix), child_prefix),
    )


def _optional_path_filter_params(path: Path | None) -> tuple[str | None, str, int, str]:
    if path is None:
        return None, "", 0, ""
    resolved = str(path.expanduser().resolve())
    child_prefix = resolved if resolved.endswith(os.sep) else f"{resolved}{os.sep}"
    return resolved, resolved, len(child_prefix), child_prefix


SqlValue = str | int | None
SqlParams = Sequence[SqlValue]


def _connect_status_db(db_path: Path) -> sqlite3.Connection:
    """Open an index DB read-only without enforcing schema version.

    ``status`` is the one read command that needs to keep working on DBs at
    other ``user_version`` values — it's how the user discovers they need to
    run a migration. Skip the version check from :func:`connect` and rely on
    the ``_status_*`` query wrappers to convert missing-table errors into
    "unavailable" cells.
    """
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _status_fetchone(
    conn: sqlite3.Connection,
    label: str,
    sql: str,
    params: SqlParams,
    warnings: list[str],
) -> tuple[SqlValue, ...] | None:
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.DatabaseError as exc:
        warnings.append(f"{label}: {exc}")
        return None
    return tuple(row) if row is not None else None


def _status_fetchall(
    conn: sqlite3.Connection,
    label: str,
    sql: str,
    params: SqlParams,
    warnings: list[str],
) -> list[tuple[SqlValue, ...]]:
    try:
        return [tuple(row) for row in conn.execute(sql, params).fetchall()]
    except sqlite3.DatabaseError as exc:
        warnings.append(f"{label}: {exc}")
        return []


def _status_int(
    conn: sqlite3.Connection,
    label: str,
    sql: str,
    params: SqlParams,
    warnings: list[str],
) -> int | None:
    row = _status_fetchone(conn, label, sql, params, warnings)
    if row is None or row[0] is None:
        warnings.append(f"{label}: no value returned")
        return None
    return int(row[0])


def _status_unavailable(label: str, reason: str, warnings: list[str]) -> None:
    warnings.append(f"{label}: {reason}")


def _format_optional_count(value: int | None) -> str:
    if value is None:
        return "[dim]unavailable[/]"
    return f"[green]{value:,}[/]"


def _format_optional_compact_count(value: int | None) -> str:
    if value is None:
        return "[dim]unavailable[/]"
    return f"[green]{_format_count(value)}[/]"


def _format_user_version(user_version: int) -> str:
    expected = CURRENT_SCHEMA_VERSION
    if user_version == expected:
        return f"[green]{user_version} / {expected}[/]"
    if user_version < expected:
        return f"[yellow]{user_version} / {expected}[/] [dim](older than this CLI)[/]"
    return f"[yellow]{user_version} / {expected}[/] [dim](newer than this CLI)[/]"


def _format_count(n: int) -> str:
    """Compact human-readable count (`19`, `258.0K`, `14.7M`, `1.2B`)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _format_ts_ns(time_ns: int | None) -> str:
    """Render UTC timestamp as ``<colored date> <cyan time>``.

    Date gets a deterministic color so different days visually pop; the time
    portion is uniform cyan to keep the row readable.
    """
    if time_ns is None or time_ns == 0:
        return "-"
    dt = datetime.fromtimestamp(time_ns / 1e9, tz=timezone.utc)
    date = dt.strftime("%Y-%m-%d")
    time = dt.strftime("%H:%M:%S")
    return f"[{_text_to_color(date)}]{date}[/] [cyan]{time}[/]"


def _format_duration_ns(start_ns: int | None, end_ns: int | None) -> str:
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return "-"
    secs = (end_ns - start_ns) / 1e9
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"
    if secs < 86400:
        h, rem = divmod(int(secs), 3600)
        return f"{h}h {rem // 60}m"
    d, rem = divmod(int(secs), 86400)
    return f"{d}d {rem // 3600}h"


def _describe_error_kind(kind: str) -> str:
    return _ERROR_KIND_LABEL.get(kind, kind)


def _path_prefix_predicate(path: Path) -> tuple[str, tuple[str | int, ...]]:
    """Same shape as ``_path_prefix_where`` but without the leading ``WHERE``."""
    where, params = _path_prefix_where(path)
    return where.removeprefix("WHERE "), params


def _like_prefix_param(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def _stdout_line(text: str) -> None:
    sys.stdout.write(f"{text}\n")


def _emit_non_table(
    fmt: OutputFormat,
    rows: list[dict[str, object]],
    path_key: str = "path",
) -> bool:
    """If fmt is json or paths-only, emit to stdout and return True.

    Returns False for ``table`` so the caller can build a Rich table itself.
    Non-table output bypasses the Rich console so shell pipes get clean text.
    """
    if fmt == "json":
        _stdout_line(_json.dumps(rows, default=str))
        return True
    if fmt == "paths-only":
        for row in rows:
            value = row.get(path_key)
            if value is not None:
                _stdout_line(str(value))
        return True
    return False


def _parse_time_or_exit(time_str: str, flag: str) -> int:
    try:
        return parse_time_arg(time_str)
    except ValueError as exc:
        console.print(f"[red]Error:[/] --{flag}: {exc}")
        raise SystemExit(2) from exc
