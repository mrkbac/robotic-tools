"""`pymcap-cli index` — sidecar catalog of MCAP summaries."""

from __future__ import annotations

import importlib.metadata
import json as _json
import logging
import os
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from cyclopts import App, Parameter
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

from pymcap_cli.display.display_utils import (
    ChannelTableData,
    _format_parts_with_colors,
    _format_schema_with_link,
    _text_to_color,
    display_channels_table,
)
from pymcap_cli.index import SANE_EPOCH_NS
from pymcap_cli.index.db import (
    CURRENT_SCHEMA_VERSION,
    IndexDbNeedsMigrationError,
    default_db_path,
    open_db,
)
from pymcap_cli.index.scanner import ScanStats, scan, unpack_distribution_blob
from pymcap_cli.utils import bytes_to_human, parse_time_arg

if TYPE_CHECKING:
    from pymcap_cli.types.info_types import ChannelInfo, SchemaInfo

OutputFormat = Literal["table", "json", "paths-only"]

logger = logging.getLogger(__name__)
console = Console()

index_app = App(
    name="index",
    help="Maintain a sidecar SQLite catalog of MCAP summaries for fast recovery.",
)


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


@index_app.command(name="scan")
def scan_cmd(
    folder: Path,
    *,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
    jobs: Annotated[
        int,
        Parameter(name=["-j", "--jobs"], help="Parallel I/O workers."),
    ] = 8,
    retry_errors: Annotated[
        bool,
        Parameter(
            name=["--retry-errors"],
            help="Retry files previously recorded in scan_error.",
        ),
    ] = False,
    rebuild_missing: Annotated[
        bool,
        Parameter(
            help=(
                "Rebuild summaries in memory for files without usable summary data. "
                "This can read entire MCAP files."
            ),
        ),
    ] = False,
    force_rebuild: Annotated[
        bool,
        Parameter(
            name=["--force-rebuild"],
            help=(
                "Re-read every discovered file end-to-end and refresh the "
                "derived columns on its ``content`` row (compression, sane "
                "times, …). Use after a migration that adds new aggregates."
            ),
        ),
    ] = False,
    root_only: Annotated[
        bool,
        Parameter(name=["--root-only"], help="Do not recurse into subfolders."),
    ] = False,
) -> int:
    """Scan FOLDER recursively and index every .mcap into the sidecar DB.

    Unchanged files (matching path + size + mtime) are skipped without any
    file I/O. Files that moved or were duplicated can reuse an existing
    summary snapshot via the cheap byte probe.
    """
    folder = folder.expanduser().resolve()
    if not folder.exists():
        console.print(f"[red]Error:[/] {folder} does not exist")
        return 1

    db_path = _resolve_db(db)
    console.print(f"[dim]DB:[/dim] {db_path}")
    console.print(f"[dim]Root:[/dim] {folder}")

    with Progress(
        TextColumn("[bold blue]Scanning"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn(
            "[dim]walked={task.fields[walked]} dirs[/] "
            "[green]indexed={task.fields[indexed]}[/] "
            "[cyan]reused={task.fields[reused]}[/] "
            "[dim]skip={task.fields[skip]}[/] "
            "[red]err={task.fields[errored]}[/]"
        ),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress_bar:
        task_id = progress_bar.add_task(
            "scan", total=None, walked=0, indexed=0, reused=0, skip=0, errored=0
        )
        seen = 0

        def _progress(stats: ScanStats) -> None:
            nonlocal seen
            total = stats.discovered + stats.deleted
            if progress_bar.tasks[task_id].total != total:
                progress_bar.update(task_id, total=total)
            done = (
                stats.deleted
                + stats.stat_skipped
                + stats.error_skipped
                + stats.indexed
                + stats.fingerprint_reused
                + stats.errored
            )
            progress_bar.update(
                task_id,
                advance=done - seen,
                walked=stats.dirs_walked,
                indexed=stats.indexed,
                reused=stats.fingerprint_reused,
                skip=stats.stat_skipped + stats.error_skipped,
                errored=stats.errored,
            )
            seen = done

        with open_db(db_path) as conn:
            stats = scan(
                folder,
                conn,
                pymcap_cli_version=_pymcap_cli_version(),
                jobs=jobs,
                recurse=not root_only,
                retry_errors=retry_errors,
                rebuild_missing=rebuild_missing,
                force_rebuild=force_rebuild,
                progress=_progress,
            )

    table = Table(title="Scan summary", show_header=False)
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Discovered", f"{stats.discovered:,}")
    table.add_row("Stat-skipped (unchanged)", f"{stats.stat_skipped:,}")
    table.add_row("Error-skipped (previously failed)", f"{stats.error_skipped:,}")
    table.add_row("Fingerprint-reused (moved/dup)", f"{stats.fingerprint_reused:,}")
    table.add_row("Indexed (new content)", f"{stats.indexed:,}")
    table.add_row("Deleted/stale paths", f"{stats.deleted:,}")
    table.add_row("Errored", f"{stats.errored:,}")
    for kind, count in sorted(stats.errored_by_kind.items()):
        table.add_row(f"  [dim]└ {_describe_error_kind(kind)}[/]", f"{count:,}")
    console.print(table)
    return 0


@index_app.command(name="status")
def status_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to filter observations by."),
    ] = None,
    *,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Show coverage stats from the sidecar DB."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    path_filter_params = _optional_path_filter_params(folder)
    warnings: list[str] = []
    # ``shape_row`` / ``last_scan_row`` populate the topic/schema/scan rows
    # below; the rest start as None so the renderer falls back to "unavailable"
    # if a query fails.
    topic_count: int | None = None
    schema_count: int | None = None
    content_channel_count: int | None = None
    last_scan_id: int | None = None
    last_scan_started_at: int | None = None
    last_scan_finished_at: int | None = None
    last_scan_root: str | None = None
    last_scan_version: str | None = None

    try:
        conn = _connect_status_db(db_path)
    except (OSError, sqlite3.DatabaseError) as exc:
        console.print(f"[red]Error:[/] could not open index DB at {db_path}: {exc}")
        return 1
    try:
        user_version = _status_int(conn, "User version", "PRAGMA user_version", (), warnings) or 0

        # All queries target the current (v7) schema. Older / partial DBs
        # raise ``sqlite3.DatabaseError`` on missing tables or columns; the
        # ``_status_*`` wrappers convert that into a warning + ``None``, which
        # the renderer surfaces as "unavailable". No legacy dispatch.

        files = _status_int(
            conn,
            "Files tracked",
            "SELECT COUNT(DISTINCT abs_path) FROM current_file "
            "WHERE (? IS NULL OR abs_path = ? OR substr(abs_path, 1, ?) = ?)",
            path_filter_params,
            warnings,
        )

        with_content = _status_int(
            conn,
            "Files with summary",
            "SELECT COUNT(DISTINCT abs_path) FROM current_file "
            "WHERE (? IS NULL OR abs_path = ? OR substr(abs_path, 1, ?) = ?) "
            "AND content_id IS NOT NULL",
            path_filter_params,
            warnings,
        )

        contents = _status_int(
            conn,
            "Distinct content rows",
            "SELECT COUNT(DISTINCT content_id) FROM current_file "
            "WHERE (? IS NULL OR abs_path = ? OR substr(abs_path, 1, ?) = ?) "
            "AND content_id IS NOT NULL",
            path_filter_params,
            warnings,
        )

        total_bytes = _status_int(
            conn,
            "Total bytes",
            "SELECT COALESCE(SUM(size_bytes),0) FROM current_file "
            "WHERE (? IS NULL OR abs_path = ? OR substr(abs_path, 1, ?) = ?)",
            path_filter_params,
            warnings,
        )

        total_messages = _status_int(
            conn,
            "Total messages",
            "SELECT COALESCE(SUM(c.message_count),0) "
            "FROM current_file cf JOIN content c ON c.id = cf.content_id "
            "WHERE (? IS NULL OR cf.abs_path = ? OR substr(cf.abs_path, 1, ?) = ?)",
            path_filter_params,
            warnings,
        )

        if folder is None:
            errors = _status_int(
                conn,
                "Scan errors recorded",
                "SELECT COUNT(*) FROM scan_error",
                (),
                warnings,
            )
            error_breakdown = [
                (str(kind), int(count or 0))
                for kind, count in _status_fetchall(
                    conn,
                    "Scan error breakdown",
                    "SELECT error_kind, COUNT(*) FROM scan_error "
                    "GROUP BY error_kind ORDER BY error_kind",
                    (),
                    warnings,
                )
            ]
        else:
            errors = _status_int(
                conn,
                "Scan errors recorded",
                "SELECT COUNT(*) FROM scan_error se "
                "JOIN file_path fp ON fp.id = se.file_path_id "
                "WHERE (? IS NULL OR fp.value = ? OR substr(fp.value, 1, ?) = ?)",
                path_filter_params,
                warnings,
            )
            error_breakdown = [
                (str(kind), int(count or 0))
                for kind, count in _status_fetchall(
                    conn,
                    "Scan error breakdown",
                    "SELECT se.error_kind, COUNT(*) FROM scan_error se "
                    "JOIN file_path fp ON fp.id = se.file_path_id "
                    "WHERE (? IS NULL OR fp.value = ? OR substr(fp.value, 1, ?) = ?) "
                    "GROUP BY se.error_kind ORDER BY se.error_kind",
                    path_filter_params,
                    warnings,
                )
            ]

        shape_row = _status_fetchone(
            conn,
            "Dataset shape",
            "SELECT COUNT(DISTINCT sig.topic_id), "
            "       COUNT(DISTINCT sig.schema_id), "
            "       COUNT(*) "
            "FROM current_file cf "
            "JOIN content_channel cc ON cc.content_id = cf.content_id "
            "JOIN channel_signature sig ON sig.id = cc.channel_signature_id "
            "WHERE (? IS NULL OR cf.abs_path = ? OR substr(cf.abs_path, 1, ?) = ?)",
            path_filter_params,
            warnings,
        )
        if shape_row is not None:
            topic_count = int(shape_row[0] or 0)
            schema_count = int(shape_row[1] or 0)
            content_channel_count = int(shape_row[2] or 0)

        last_scan_row = _status_fetchone(
            conn,
            "Last DB scan",
            "SELECT s.id, s.started_at_ns, s.finished_at_ns, "
            "       fp.value, s.pymcap_cli_version "
            "FROM scan_session s "
            "LEFT JOIN file_path fp ON fp.id = s.root_file_path_id "
            "ORDER BY s.id DESC LIMIT 1",
            (),
            warnings,
        )
        if last_scan_row is not None:
            last_scan_id = int(last_scan_row[0]) if last_scan_row[0] is not None else None
            last_scan_started_at = int(last_scan_row[1]) if last_scan_row[1] is not None else None
            last_scan_finished_at = int(last_scan_row[2]) if last_scan_row[2] is not None else None
            last_scan_root = str(last_scan_row[3]) if last_scan_row[3] is not None else None
            last_scan_version = str(last_scan_row[4]) if last_scan_row[4] is not None else None
    finally:
        conn.close()

    if files is None or with_content is None:
        coverage = "[dim]unavailable[/]"
    elif files:
        coverage = f"{with_content:,} / {files:,}"
        coverage += f"  [dim]({with_content * 100 // files}%)[/]"
    else:
        coverage = "0 / 0"

    # Sum of the main DB file plus the WAL/SHM sidecars. WAL can dominate the
    # apparent on-disk footprint between checkpoints, so showing the total
    # avoids a "but ls says X" surprise.
    sidecars = (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    )
    db_size_bytes = sum(s.stat().st_size for s in sidecars if s.exists())

    table = Table(title="Index status", show_header=False)
    table.add_column(style="bold blue")
    table.add_column()
    table.add_row("DB", _format_parts_with_colors(str(db_path)))
    table.add_row(
        "DB size",
        f"[yellow]{bytes_to_human(db_size_bytes)}[/]  [dim]({db_size_bytes:,} B)[/]",
    )
    table.add_row("User version", _format_user_version(user_version))
    if folder is not None:
        table.add_row("Filter", _format_parts_with_colors(str(folder)))
    table.add_row("Files tracked", _format_optional_count(files))
    table.add_row("Files with summary", f"[green]{coverage}[/]")
    table.add_row("Distinct content rows", _format_optional_count(contents))
    table.add_row(
        "Scan errors recorded",
        "[dim]unavailable[/]"
        if errors is None
        else (f"[red]{errors:,}[/]" if errors else f"[green]{errors:,}[/]"),
    )
    for kind, count in error_breakdown:
        table.add_row(f"  [dim]└ {_describe_error_kind(kind)}[/]", f"[dim]{count:,}[/]")
    table.add_row("Total messages", _format_optional_compact_count(total_messages))
    table.add_row(
        "Total bytes",
        "[dim]unavailable[/]"
        if total_bytes is None
        else f"[yellow]{bytes_to_human(total_bytes)}[/]  [dim]({total_bytes:,} B)[/]",
    )
    if topic_count is None or schema_count is None:
        table.add_row("Topics / schemas", "[dim]unavailable[/]")
    else:
        table.add_row("Topics / schemas", f"[green]{topic_count:,}[/] / [cyan]{schema_count:,}[/]")
    table.add_row("Content channels", _format_optional_count(content_channel_count))
    if last_scan_id is None:
        table.add_row("Last DB scan", "[dim]none[/]")
    else:
        state = (
            "[yellow]RUNNING[/]"
            if last_scan_finished_at is None
            else f"[cyan]{_format_duration_ns(last_scan_started_at, last_scan_finished_at)}[/]"
        )
        table.add_row("Last DB scan", f"[green]#{last_scan_id}[/]  {state}")
        table.add_row("  Started", _format_ts_ns(last_scan_started_at))
        table.add_row(
            "  Finished",
            "[yellow]RUNNING[/]"
            if last_scan_finished_at is None
            else _format_ts_ns(last_scan_finished_at),
        )
        if last_scan_root is not None:
            table.add_row("  Root", _format_parts_with_colors(last_scan_root))
        if last_scan_version is not None:
            table.add_row("  CLI version", f"[dim]{last_scan_version}[/]")
    if warnings:
        table.add_row("Warnings", f"[yellow]{'; '.join(warnings[:5])}[/]")
    console.print(table)
    return 0


@dataclass
class _PathNode:
    """Aggregate stats for one directory in the path tree."""

    file_count: int = 0
    size_bytes: int = 0
    message_count: int = 0
    duration_ns: int = 0
    topics: set[str] = field(default_factory=set)
    schemas: set[str] = field(default_factory=set)
    children: dict[str, _PathNode] = field(default_factory=dict)


def _format_seconds_short(secs: float) -> str:
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


def _format_node_stats(node: _PathNode) -> str:
    parts = [
        f"[yellow]{bytes_to_human(node.size_bytes)}[/]",
        f"[green]{_format_count(node.file_count)}f[/]",
        f"[cyan]{_format_count(node.message_count)}msg[/]",
    ]
    if node.duration_ns:
        parts.append(f"[magenta]{_format_seconds_short(node.duration_ns / 1e9)}[/]")
    if node.topics:
        parts.append(f"[blue]{len(node.topics)} topics[/]")
    if node.schemas:
        parts.append(f"[dim]{len(node.schemas)} schemas[/]")
    return "  ".join(parts)


def _build_path_tree(
    files: Sequence[tuple[str, int | None, int | None, int | None, int | None]],
    topics: Sequence[tuple[str, str | None]],
    schemas: Sequence[tuple[str, str | None]],
    root_prefix: str,
) -> _PathNode:
    """Group rows by their path prefix and accumulate stats up the chain.

    ``topics`` and ``schemas`` are aggregated one row per file, with the
    member values comma-joined. That avoids materializing one Python row
    per (file, topic) pair — for large indexes the un-aggregated fan-out
    is the dominant cost of the tree. Topic names start with ``/`` and
    schema hashes are hex digests, so ``,`` is a safe separator.
    """
    root = _PathNode()
    chain_cache: dict[str, list[_PathNode]] = {}

    def _chain_for(abs_path: str) -> list[_PathNode]:
        cached = chain_cache.get(abs_path)
        if cached is not None:
            return cached
        try:
            rel = os.path.relpath(abs_path, root_prefix) if root_prefix else abs_path
        except ValueError:
            rel = abs_path
        parts = [p for p in Path(rel).parts if p not in ("", os.sep, ".")]
        # Last component is the filename — only its ancestor directories
        # get nodes.
        chain = [root]
        node = root
        for part in parts[:-1]:
            node = node.children.setdefault(part, _PathNode())
            chain.append(node)
        chain_cache[abs_path] = chain
        return chain

    for abs_path, size, msg_count, ts_start, ts_end in files:
        chain = _chain_for(abs_path)
        size_v = size or 0
        msg_v = msg_count or 0
        dur_v = _safe_duration_ns(ts_start, ts_end) or 0
        for node in chain:
            node.file_count += 1
            node.size_bytes += size_v
            node.message_count += msg_v
            node.duration_ns += dur_v

    for abs_path, joined in topics:
        if not joined:
            continue
        members = joined.split(",")
        for node in _chain_for(abs_path):
            node.topics.update(members)

    for abs_path, joined in schemas:
        if not joined:
            continue
        members = joined.split(",")
        for node in _chain_for(abs_path):
            node.schemas.update(members)

    return root


_TREE_SORT_KEYS = {
    "size": lambda kv: (-kv[1].size_bytes, kv[0]),
    "files": lambda kv: (-kv[1].file_count, kv[0]),
    "messages": lambda kv: (-kv[1].message_count, kv[0]),
    "duration": lambda kv: (-kv[1].duration_ns, kv[0]),
    "name": lambda kv: (0.0, kv[0]),
}


def _fold_single_child_chain(name: str, node: _PathNode) -> tuple[str, _PathNode]:
    """Collapse ``a/ -> b/ -> c/`` chains where each level has exactly one child."""
    while len(node.children) == 1:
        only_name, only_child = next(iter(node.children.items()))
        name = f"{name}/{only_name}"
        node = only_child
    return name, node


def _render_path_tree(
    root: _PathNode,
    root_label: str,
    *,
    max_depth: int,
    min_files: int,
    sort_by: str,
) -> Tree:
    sort_key = _TREE_SORT_KEYS[sort_by]
    folded_root_label, folded_root = _fold_single_child_chain(root_label, root)
    tree = Tree(f"[bold blue]{folded_root_label}[/]  [dim]→[/]  {_format_node_stats(folded_root)}")

    def _add(parent: Tree, node: _PathNode, depth: int) -> None:
        if depth >= max_depth:
            if node.children:
                parent.add(f"[dim]… {len(node.children):,} subdirs collapsed[/]")
            return
        for name, child in sorted(node.children.items(), key=sort_key):
            if child.file_count < min_files:
                continue
            display_name, display_child = _fold_single_child_chain(name, child)
            sub = parent.add(
                f"[bold]{display_name}/[/]  [dim]→[/]  {_format_node_stats(display_child)}"
            )
            _add(sub, display_child, depth + 1)

    _add(tree, folded_root, 0)
    return tree


@index_app.command(name="tree")
def tree_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to restrict the tree to."),
    ] = None,
    *,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
    max_depth: Annotated[
        int,
        Parameter(
            help=(
                "Limit how many directory levels to render. "
                "Aggregates still cover everything below."
            ),
        ),
    ] = 4,
    min_files: Annotated[
        int,
        Parameter(help="Hide directories containing fewer than this many .mcap files."),
    ] = 1,
    sort_by: Annotated[
        Literal["size", "files", "messages", "duration", "name"],
        Parameter(help="Sort children of each node by this metric (descending)."),
    ] = "size",
) -> int:
    """Show a directory-tree breakdown of indexed data.

    Each node aggregates everything below it: total size on disk, number of
    indexed files, total message count, total duration, and the number of
    distinct topics / schemas that appear under the prefix.
    """
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    path_filter_params = _optional_path_filter_params(folder)
    time_filter_params = (SANE_EPOCH_NS, SANE_EPOCH_NS)

    try:
        with open_db(db_path, read_only=True) as conn:
            files = conn.execute(
                """SELECT cf.abs_path, cf.size_bytes, c.message_count,
                          CASE WHEN c.message_start_time_ns >= ?
                               THEN c.message_start_time_ns ELSE c.sane_message_start_time_ns END
                               AS eff_start,
                          CASE WHEN c.message_start_time_ns >= ?
                               THEN c.message_end_time_ns ELSE c.sane_message_end_time_ns END
                               AS eff_end
                    FROM current_file cf
                    JOIN content c ON c.id = cf.content_id
                    WHERE (? IS NULL OR cf.abs_path = ?
                           OR substr(cf.abs_path, 1, ?) = ?)""",
                (*time_filter_params, *path_filter_params),
            ).fetchall()
            # One row per file. The per-content aggregate (~56k content rows on a
            # representative corpus) collapses what would otherwise be a
            # row-per-(file, topic) fan-out (~1.5M rows) and is the dominant cost
            # of the tree. Members are comma-joined; see ``_build_path_tree``.
            topics = conn.execute(
                """SELECT cf.abs_path, agg.names
                    FROM current_file cf
                    JOIN (
                        SELECT cc.content_id, GROUP_CONCAT(DISTINCT t.name) AS names
                        FROM content_channel cc
                        JOIN channel_signature sig ON sig.id = cc.channel_signature_id
                        JOIN topic t         ON t.id          = sig.topic_id
                        GROUP BY cc.content_id
                    ) agg ON agg.content_id = cf.content_id
                    WHERE (? IS NULL OR cf.abs_path = ?
                           OR substr(cf.abs_path, 1, ?) = ?)""",
                path_filter_params,
            ).fetchall()
            schema_rows = conn.execute(
                """SELECT cf.abs_path, agg.hashes
                    FROM current_file cf
                    JOIN (
                        SELECT cs.content_id, GROUP_CONCAT(DISTINCT s.schema_hash) AS hashes
                        FROM content_schema cs
                        JOIN schema s ON s.id = cs.schema_id
                        GROUP BY cs.content_id
                    ) agg ON agg.content_id = cf.content_id
                    WHERE (? IS NULL OR cf.abs_path = ?
                           OR substr(cf.abs_path, 1, ?) = ?)""",
                path_filter_params,
            ).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    if not files:
        console.print("[dim]No indexed files match.[/]")
        return 0

    if folder is not None:
        root_prefix = str(folder.expanduser().resolve())
    else:
        try:
            root_prefix = os.path.commonpath([row[0] for row in files])
        except ValueError:
            root_prefix = ""

    root_node = _build_path_tree(files, topics, schema_rows, root_prefix)
    rendered = _render_path_tree(
        root_node,
        root_label=root_prefix or "(all)",
        max_depth=max_depth,
        min_files=min_files,
        sort_by=sort_by,
    )
    console.print(rendered)
    return 0


@index_app.command(name="query")
def query_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to restrict results to."),
    ] = None,
    *,
    sort_by: Annotated[
        Literal["path", "duration", "messages", "size", "start"],
        Parameter(
            name=["--sort-by"],
            help="Sort results (descending except for ``path``).",
        ),
    ] = "path",
    topic: Annotated[
        str | None,
        Parameter(name=["--topic"], help="Match files containing this topic."),
    ] = None,
    schema: Annotated[
        str | None,
        Parameter(name=["--schema"], help="Match files containing this schema name."),
    ] = None,
    fingerprint: Annotated[
        str | None,
        Parameter(name=["--fingerprint"], help="Match by summary fingerprint (e.g. 's1:…')."),
    ] = None,
    at: Annotated[
        str | None,
        Parameter(
            name=["--at"],
            help="Match files whose time range contains this instant (ns or RFC3339).",
        ),
    ] = None,
    since: Annotated[
        str | None,
        Parameter(
            name=["--since"],
            help="Match files overlapping the window starting at this instant.",
        ),
    ] = None,
    until: Annotated[
        str | None,
        Parameter(
            name=["--until"],
            help="Match files overlapping the window ending at this instant.",
        ),
    ] = None,
    limit: Annotated[int, Parameter(name=["--limit"], help="Max rows to print.")] = 50,
    format: Annotated[
        OutputFormat,
        Parameter(name=["--format"], help="Output as Rich table, JSON, or paths-only."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Look up files in the sidecar DB."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    window_start: int | None = None
    window_end: int | None = None
    if at is not None:
        instant = _parse_time_or_exit(at, "at")
        window_start = instant
        window_end = instant
    if since is not None:
        window_start = _parse_time_or_exit(since, "since")
    if until is not None:
        window_end = _parse_time_or_exit(until, "until")

    # Mirror ``_safe_duration_ns`` in SQL: reject sub-epoch starts and
    # implausibly-long spans so the sort key matches the rendered cell. The
    # CASE collapses bogus rows to ``0`` so they sink to the bottom on
    # ``--sort-by duration DESC``.
    _safe_dur_sql = (
        f"CASE WHEN {_EFF_START_SQL} >= {SANE_EPOCH_NS}"
        f"      AND ({_EFF_END_SQL} - {_EFF_START_SQL}) BETWEEN 0 AND {_MAX_PLAUSIBLE_DURATION_NS}"
        f"     THEN ({_EFF_END_SQL} - {_EFF_START_SQL}) ELSE 0 END"
    )
    _safe_start_sql = (
        f"CASE WHEN {_EFF_START_SQL} >= {SANE_EPOCH_NS} THEN {_EFF_START_SQL} ELSE NULL END"
    )
    order_by = {
        "path": "cf.abs_path",
        "duration": f"{_safe_dur_sql} DESC",
        "messages": "messages DESC"
        if (topic is not None or schema is not None)
        else "c.message_count DESC",
        "size": "cf.size_bytes DESC",
        "start": f"{_safe_start_sql} DESC",
    }[sort_by]

    folder_clause = ""
    folder_params: tuple[str | int, ...] = ()
    if folder is not None:
        folder_where, folder_params = _path_prefix_where(folder)
        # _path_prefix_where returns ``WHERE ...`` — re-tag as inline conjunct.
        folder_clause = folder_where.removeprefix("WHERE ").replace("abs_path", "cf.abs_path")

    channel_filtered = topic is not None or schema is not None
    params: list[str | int] = []
    if channel_filtered:
        sql = (
            "SELECT cf.abs_path, "  # noqa: S608
            "COALESCE(SUM(cc.message_count), 0) AS messages, "
            "COUNT(DISTINCT cc.mcap_channel_id) AS channels, "
            f"{_EFF_START_SQL} AS eff_start, "
            f"{_EFF_END_SQL}   AS eff_end, "
            "cf.size_bytes, "
            "c.summary_fingerprint "
            "FROM current_file cf "
            "JOIN content c           ON c.id        = cf.content_id "
            "JOIN content_channel cc  ON cc.content_id       = cf.content_id "
            "JOIN channel_signature sig     ON sig.id  = cc.channel_signature_id "
        )
        where: list[str] = []
        if topic is not None:
            sql += "JOIN topic t ON t.id = sig.topic_id "
            where.append("t.name = ?")
            params.append(topic)
        if schema is not None:
            sql += "JOIN schema s ON s.id = sig.schema_id "
            where.append("s.name = ?")
            params.append(schema)
        if fingerprint is not None:
            where.append("c.summary_fingerprint = ?")
            params.append(fingerprint)
        if window_end is not None:
            where.append("c.message_start_time_ns <= ?")
            params.append(window_end)
        if window_start is not None:
            where.append("c.message_end_time_ns >= ?")
            params.append(window_start)
        if folder_clause:
            where.append(folder_clause)
            params.extend(folder_params)
        sql += "WHERE " + " AND ".join(where) + " "
        sql += f"GROUP BY cf.abs_path ORDER BY {order_by} LIMIT ?"
    else:
        sql = (
            "SELECT cf.abs_path, c.message_count, c.channel_count, "  # noqa: S608
            f"{_EFF_START_SQL} AS eff_start, "
            f"{_EFF_END_SQL}   AS eff_end, "
            "cf.size_bytes, "
            "c.summary_fingerprint "
            "FROM current_file cf "
            "JOIN content c ON c.id = cf.content_id "
        )
        where = []
        if fingerprint is not None:
            where.append("c.summary_fingerprint = ?")
            params.append(fingerprint)
        if window_end is not None:
            where.append("c.message_start_time_ns <= ?")
            params.append(window_end)
        if window_start is not None:
            where.append("c.message_end_time_ns >= ?")
            params.append(window_start)
        if folder_clause:
            where.append(folder_clause)
            params.extend(folder_params)
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += f"ORDER BY {order_by} LIMIT ?"
    params.append(limit)

    try:
        with open_db(db_path, read_only=True) as conn:
            sql_rows = conn.execute(sql, params).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    rows: list[dict[str, object]] = [
        {
            "short_id": _short_id_from_fingerprint(fp),
            "summary_fingerprint": fp,
            "path": path,
            "messages": msgs,
            "channels": channels,
            "start_time_ns": start,
            "end_time_ns": end,
            "duration_ns": _safe_duration_ns(start, end),
            "size_bytes": size,
        }
        for path, msgs, channels, start, end, size, fp in sql_rows
    ]

    if format == "table":
        filter_desc: list[str] = []
        if topic is not None:
            filter_desc.append(f"topic={topic}")
        if schema is not None:
            filter_desc.append(f"schema={schema}")
        if fingerprint is not None:
            filter_desc.append(f"fingerprint={fingerprint}")
        if at is not None:
            filter_desc.append(f"at={at}")
        if since is not None:
            filter_desc.append(f"since={since}")
        if until is not None:
            filter_desc.append(f"until={until}")
        if filter_desc:
            console.print(f"[dim]Filter:[/] {' '.join(filter_desc)}")

    if not rows:
        if format == "table":
            console.print("[yellow]No matches[/]")
        elif format == "json":
            _stdout_line("[]")
        return 0

    if _emit_non_table(format, rows):
        return 0

    msg_header = "Matched msgs" if channel_filtered else "Messages"
    ch_header = "Matched ch." if channel_filtered else "Channels"
    table = Table()
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Path", overflow="fold")
    table.add_column(msg_header, justify="right", style="green")
    table.add_column(ch_header, justify="right", style="green")
    table.add_column("Start (UTC)")
    table.add_column("End (UTC)")
    table.add_column("Duration", justify="right", style="cyan")
    for row in rows:
        msgs = row["messages"]
        channels = row["channels"]
        start = row["start_time_ns"]
        end = row["end_time_ns"]
        duration_ns = row["duration_ns"]
        duration_str = _format_duration_ns(0, duration_ns) if isinstance(duration_ns, int) else "-"
        short_id = row["short_id"] if isinstance(row["short_id"], str) and row["short_id"] else "-"
        table.add_row(
            short_id,
            _format_parts_with_colors(str(row["path"])),
            f"{msgs:,}" if isinstance(msgs, int) else "-",
            f"{channels:,}" if isinstance(channels, int) else "-",
            _format_ts_ns(start if isinstance(start, int) else None),
            _format_ts_ns(end if isinstance(end, int) else None),
            duration_str,
        )
    console.print(table)
    return 0


@index_app.command(name="topics")
def topics_cmd(
    prefix: Annotated[
        str | None,
        Parameter(help="Optional topic prefix filter (e.g. '/tf' or '/sensor')."),
    ] = None,
    *,
    sort_by: Annotated[
        Literal["files", "messages", "schemas", "name"],
        Parameter(
            name=["--sort-by"],
            help="Sort results (descending except for ``name``).",
        ),
    ] = "files",
    limit: Annotated[int, Parameter(name=["--limit"], help="Max rows to print.")] = 50,
    min_files: Annotated[
        int,
        Parameter(name=["--min-files"], help="Hide topics seen in fewer files than this."),
    ] = 1,
    format: Annotated[
        OutputFormat,
        Parameter(name=["--format"], help="Output as Rich table or JSON (paths-only is N/A)."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """List topics in the index with file and message counts."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    # Always tie-break by the other counts and then the topic name for
    # deterministic output.
    order_by = {
        "files": "files DESC, messages DESC, t.name",
        "messages": "messages DESC, files DESC, t.name",
        "schemas": "schemas DESC, files DESC, t.name",
        "name": "t.name",
    }[sort_by]

    sql = (
        "SELECT t.name AS topic, "
        "       COUNT(DISTINCT cf.abs_path)        AS files, "
        "       COALESCE(SUM(cc.message_count), 0) AS messages, "
        "       COUNT(DISTINCT sig.schema_id)   AS schemas, "
        "       MIN(s.name)                        AS schema_name "
        "FROM current_file cf "
        "JOIN content_channel cc ON cc.content_id      = cf.content_id "
        "JOIN channel_signature sig    ON sig.id = cc.channel_signature_id "
        "JOIN topic t            ON t.id         = sig.topic_id "
        "LEFT JOIN schema s      ON s.id     = sig.schema_id "
    )
    params: list[str | int] = []
    if prefix is not None:
        sql += "WHERE t.name LIKE ? ESCAPE '\\' "
        params.append(_like_prefix_param(prefix))
    sql += f"GROUP BY t.name HAVING files >= ? ORDER BY {order_by} LIMIT ?"
    params.extend([min_files, limit])

    try:
        with open_db(db_path, read_only=True) as conn:
            sql_rows = conn.execute(sql, params).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    rows: list[dict[str, object]] = [
        {
            "topic": topic,
            "files": files,
            "messages": messages,
            "schemas": schemas,
            "schema": schema_name,
        }
        for topic, files, messages, schemas, schema_name in sql_rows
    ]

    if not rows:
        if format == "table":
            console.print("[yellow]No topics[/]")
        elif format == "json":
            _stdout_line("[]")
        return 0

    if _emit_non_table(format, rows, path_key="topic"):
        return 0

    table = Table(title=f"Topics ({len(rows):,})")
    table.add_column("Topic", overflow="fold")
    table.add_column("Files", justify="right", style="green")
    table.add_column("Messages", justify="right", style="green")
    table.add_column("Schema", overflow="fold")
    for row in rows:
        files = row["files"]
        messages = row["messages"]
        schemas = row["schemas"]
        schema_name = row["schema"]
        if isinstance(schema_name, str):
            cell = _format_schema_with_link(schema_name)
            if isinstance(schemas, int) and schemas > 1:
                cell += f"  [dim](+{schemas - 1} more)[/]"
        else:
            cell = "-"
        table.add_row(
            _format_parts_with_colors(str(row["topic"])),
            f"{files:,}" if isinstance(files, int) else "-",
            _format_count(int(messages)) if isinstance(messages, int) else "-",
            cell,
        )
    console.print(table)
    return 0


@index_app.command(name="schemas")
def schemas_cmd(
    prefix: Annotated[
        str | None,
        Parameter(help="Optional schema-name prefix (e.g. 'sensor_msgs')."),
    ] = None,
    *,
    sort_by: Annotated[
        Literal["files", "messages", "topics", "name", "encoding"],
        Parameter(
            name=["--sort-by"],
            help="Sort results (descending for the counts; ascending for name / encoding).",
        ),
    ] = "files",
    limit: Annotated[int, Parameter(name=["--limit"], help="Max rows to print.")] = 50,
    min_files: Annotated[
        int,
        Parameter(name=["--min-files"], help="Hide schemas used by fewer files than this."),
    ] = 1,
    format: Annotated[
        OutputFormat,
        Parameter(name=["--format"], help="Output as Rich table or JSON (paths-only is N/A)."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """List schema names in the index with the number of files using each."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    order_by = {
        "files": "files DESC, s.name",
        "messages": "messages DESC, files DESC, s.name",
        "topics": "topics DESC, files DESC, s.name",
        "name": "s.name",
        "encoding": "s.encoding, s.name",
    }[sort_by]

    # JOIN through ``channel_signature`` so we can count topics + messages per
    # schema. This drops schemas that are declared in a Summary but never
    # referenced by any channel — that's both rare and arguably more useful,
    # since the new ``topics`` / ``messages`` columns are meaningless without
    # a channel anyway.
    sql = (
        "SELECT s.name, s.encoding, "
        "       COUNT(DISTINCT cf.abs_path)        AS files, "
        "       COUNT(DISTINCT sig.topic_id)       AS topics, "
        "       COALESCE(SUM(cc.message_count), 0) AS messages "
        "FROM current_file cf "
        "JOIN content_channel cc ON cc.content_id      = cf.content_id "
        "JOIN channel_signature sig    ON sig.id = cc.channel_signature_id "
        "JOIN schema s           ON s.id     = sig.schema_id "
    )
    params: list[str | int] = []
    if prefix is not None:
        sql += "WHERE s.name LIKE ? ESCAPE '\\' "
        params.append(_like_prefix_param(prefix))
    sql += f"GROUP BY s.id, s.name, s.encoding HAVING files >= ? ORDER BY {order_by} LIMIT ?"
    params.extend([min_files, limit])

    try:
        with open_db(db_path, read_only=True) as conn:
            sql_rows = conn.execute(sql, params).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    rows: list[dict[str, object]] = [
        {
            "name": name,
            "encoding": encoding,
            "files": files,
            "topics": topics,
            "messages": messages,
        }
        for name, encoding, files, topics, messages in sql_rows
    ]

    if not rows:
        if format == "table":
            console.print("[yellow]No schemas[/]")
        elif format == "json":
            _stdout_line("[]")
        return 0

    if _emit_non_table(format, rows, path_key="name"):
        return 0

    table = Table(title=f"Schemas ({len(rows):,})")
    table.add_column("Name", overflow="fold")
    table.add_column("Encoding", style="yellow")
    table.add_column("Files", justify="right", style="green")
    table.add_column("Topics", justify="right", style="green")
    table.add_column("Messages", justify="right", style="green")
    for row in rows:
        files = row["files"]
        topics = row["topics"]
        messages = row["messages"]
        name = row["name"]
        table.add_row(
            _format_schema_with_link(str(name)) if name else "-",
            str(row.get("encoding") or "-"),
            f"{files:,}" if isinstance(files, int) else "-",
            f"{topics:,}" if isinstance(topics, int) else "-",
            _format_count(int(messages)) if isinstance(messages, int) else "-",
        )
    console.print(table)
    return 0


@index_app.command(name="duplicates")
def duplicates_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to restrict the search."),
    ] = None,
    *,
    min_copies: Annotated[
        int,
        Parameter(name=["--min-copies"], help="Only show fingerprints with at least N paths."),
    ] = 2,
    min_bytes: Annotated[
        int,
        Parameter(
            name=["--min-bytes"],
            help="Only show contents whose size is at least this many bytes.",
        ),
    ] = 0,
    sort: Annotated[
        Literal["savings", "copies", "size"],
        Parameter(name=["--sort"], help="Order rows by reclaimable bytes, copy count, or size."),
    ] = "savings",
    limit: Annotated[int, Parameter(name=["--limit"], help="Max groups to print.")] = 50,
    format: Annotated[
        OutputFormat,
        Parameter(name=["--format"], help="Output as Rich table, JSON, or paths-only."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Group files by cheap byte probe and report likely reclaimable disk space."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    order_clause = {
        "savings": "reclaimable_bytes DESC",
        "copies": "copies DESC, reclaimable_bytes DESC",
        "size": "size_bytes DESC",
    }[sort]

    sql = (
        "SELECT cf.file_fingerprint, MIN(cf.size_bytes) AS size_bytes, "
        "       COUNT(*) AS copies, "
        "       MIN(cf.size_bytes) * (COUNT(*) - 1) AS reclaimable_bytes, "
        "       GROUP_CONCAT(cf.abs_path, CHAR(10)) AS paths "
        "FROM current_file cf "
    )
    where_parts = ["cf.content_id IS NOT NULL"]
    params: list[str | int] = []
    if folder is not None:
        predicate, prefix_params = _path_prefix_predicate(folder)
        where_parts.append(f"({predicate})")
        params.extend(prefix_params)
    sql += "WHERE " + " AND ".join(where_parts) + " "
    sql += (
        "GROUP BY cf.file_fingerprint "
        "HAVING copies >= ? AND size_bytes >= ? "
        f"ORDER BY {order_clause} LIMIT ?"
    )
    params.extend([min_copies, min_bytes, limit])

    try:
        with open_db(db_path, read_only=True) as conn:
            sql_rows = conn.execute(sql, params).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    groups: list[dict[str, object]] = [
        {
            "file_fingerprint": fp,
            "size_bytes": size,
            "copies": copies,
            "reclaimable_bytes": reclaimable,
            "paths": (paths_blob or "").split("\n"),
        }
        for fp, size, copies, reclaimable, paths_blob in sql_rows
    ]

    if not groups:
        if format == "table":
            console.print("[yellow]No duplicate groups[/]")
        elif format == "json":
            _stdout_line("[]")
        return 0

    if format == "json":
        _stdout_line(_json.dumps(groups, default=str))
        return 0
    if format == "paths-only":
        for group in groups:
            paths_list = group["paths"]
            if isinstance(paths_list, list):
                for path in paths_list:
                    _stdout_line(str(path))
        return 0

    total_reclaimable = sum(v for g in groups if isinstance(v := g["reclaimable_bytes"], int))
    table = Table(
        title=(
            f"Duplicate groups ({len(groups):,}) — "
            f"reclaimable ≈ {bytes_to_human(total_reclaimable)}"
        )
    )
    table.add_column("Fingerprint", style="dim")
    table.add_column("Size", justify="right", style="yellow")
    table.add_column("Copies", justify="right", style="green")
    table.add_column("Reclaimable", justify="right", style="yellow")
    table.add_column("Paths", overflow="fold")
    for group in groups:
        size = group["size_bytes"]
        copies = group["copies"]
        reclaimable = group["reclaimable_bytes"]
        paths_list = group["paths"]
        if isinstance(paths_list, list):
            paths_text = "\n".join(_format_parts_with_colors(str(p)) for p in paths_list)
        else:
            paths_text = "-"
        table.add_row(
            str(group["file_fingerprint"]),
            bytes_to_human(size) if isinstance(size, int) else "-",
            f"{copies:,}" if isinstance(copies, int) else "-",
            bytes_to_human(reclaimable) if isinstance(reclaimable, int) else "-",
            paths_text,
        )
    console.print(table)
    return 0


@index_app.command(name="sessions")
def sessions_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to filter sessions by their ``root_path``."),
    ] = None,
    *,
    since: Annotated[
        str | None,
        Parameter(
            name=["--since"],
            help="Only sessions started after this instant (ns or RFC3339).",
        ),
    ] = None,
    until: Annotated[
        str | None,
        Parameter(
            name=["--until"],
            help="Only sessions started before this instant (ns or RFC3339).",
        ),
    ] = None,
    running: Annotated[
        bool,
        Parameter(
            name=["--running"],
            help="Only sessions that have not finished (``finished_at IS NULL``).",
        ),
    ] = False,
    limit: Annotated[int, Parameter(name=["--limit"], help="Max sessions to print.")] = 25,
    format: Annotated[
        OutputFormat,
        Parameter(
            name=["--format"],
            help="Output as Rich table, JSON, or paths-only (root_path).",
        ),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """List ``scan_session`` rows — when, where, and how each scan went."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    where: list[str] = []
    params: list[str | int] = []
    if folder is not None:
        clause, prefix_params = _path_prefix_where(folder)
        where.append(clause.removeprefix("WHERE ").replace("abs_path", "fp.value"))
        params.extend(prefix_params)
    if since is not None:
        where.append("s.started_at_ns >= ?")
        params.append(_parse_time_or_exit(since, "since"))
    if until is not None:
        where.append("s.started_at_ns <= ?")
        params.append(_parse_time_or_exit(until, "until"))
    if running:
        where.append("s.finished_at_ns IS NULL")

    sql = (
        "SELECT s.id, s.started_at_ns, s.finished_at_ns, fp.value AS root_path, "
        "       s.pymcap_cli_version, "
        "       COALESCE(obs.n, 0)  AS observations, "
        "       COALESCE(newc.n, 0) AS new_content, "
        "       COALESCE(errs.n, 0) AS errors "
        "FROM scan_session s "
        "JOIN file_path fp ON fp.id = s.root_file_path_id "
        "LEFT JOIN ("
        "  SELECT scan_session_id, COUNT(*) AS n FROM file_observation GROUP BY scan_session_id"
        ") obs "
        "  ON obs.scan_session_id = s.id "
        "LEFT JOIN (SELECT first_seen_scan_session_id AS scan_session_id, COUNT(*) AS n "
        "           FROM content WHERE first_seen_scan_session_id IS NOT NULL "
        "           GROUP BY first_seen_scan_session_id) newc "
        "  ON newc.scan_session_id = s.id "
        "LEFT JOIN (SELECT scan_session_id, COUNT(*) AS n FROM scan_error "
        "           GROUP BY scan_session_id) errs "
        "  ON errs.scan_session_id = s.id "
    )
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += "ORDER BY s.id DESC LIMIT ?"
    params.append(limit)

    try:
        with open_db(db_path, read_only=True) as conn:
            rows_sql = conn.execute(sql, params).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    rows: list[dict[str, object]] = [
        {
            "id": sid,
            "started_at_ns": started,
            "finished_at_ns": finished,
            "duration_ns": (finished - started)
            if (finished and started and finished > started)
            else None,
            "root_path": root,
            "pymcap_cli_version": version,
            "observations": obs,
            "new_content": newc,
            "errors": errs,
        }
        for sid, started, finished, root, version, obs, newc, errs in rows_sql
    ]

    if not rows:
        if format == "table":
            console.print("[yellow]No sessions[/]")
        elif format == "json":
            _stdout_line("[]")
        return 0

    if _emit_non_table(format, rows, path_key="root_path"):
        return 0

    table = Table(title=f"Scan sessions ({len(rows):,})")
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Started (UTC)")
    table.add_column("Finished (UTC)")
    table.add_column("Duration", justify="right", style="cyan")
    table.add_column("Root", overflow="fold")
    table.add_column("Files", justify="right", style="green")
    table.add_column("New", justify="right", style="green")
    table.add_column("Errors", justify="right", style="red")
    for row in rows:
        finished = row["finished_at_ns"]
        if not isinstance(finished, int):
            duration_cell = "[yellow]RUNNING[/]"
        else:
            duration_cell = _format_duration_ns(
                row["started_at_ns"] if isinstance(row["started_at_ns"], int) else None,
                finished,
            )
        errs = row["errors"]
        table.add_row(
            str(row["id"]),
            _format_ts_ns(row["started_at_ns"] if isinstance(row["started_at_ns"], int) else None),
            _format_ts_ns(finished if isinstance(finished, int) else None),
            duration_cell,
            _format_parts_with_colors(str(row["root_path"])),
            f"{row['observations']:,}" if isinstance(row["observations"], int) else "-",
            f"{row['new_content']:,}" if isinstance(row["new_content"], int) else "-",
            f"[red]{errs:,}[/]"
            if isinstance(errs, int) and errs
            else (f"{errs:,}" if isinstance(errs, int) else "-"),
        )
    console.print(table)
    return 0


@index_app.command(name="errors")
def errors_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to restrict errors to."),
    ] = None,
    *,
    kind: Annotated[
        str | None,
        Parameter(
            name=["--kind"],
            help="Filter by error_kind (e.g. ``corrupt``, ``no_summary``, ``io``).",
        ),
    ] = None,
    session: Annotated[
        int | None,
        Parameter(name=["--session"], help="Filter by ``scan_session.id``."),
    ] = None,
    since: Annotated[
        str | None,
        Parameter(name=["--since"], help="Only errors observed after this instant."),
    ] = None,
    until: Annotated[
        str | None,
        Parameter(name=["--until"], help="Only errors observed before this instant."),
    ] = None,
    current: Annotated[
        bool,
        Parameter(
            name=["--current"],
            help="Only errors whose (path, size, mtime) still match the latest "
            "file_observation — i.e. the file is still broken in the same way.",
        ),
    ] = False,
    limit: Annotated[int, Parameter(name=["--limit"], help="Max rows to print.")] = 50,
    format: Annotated[
        OutputFormat,
        Parameter(
            name=["--format"],
            help="Output as Rich table, JSON, or paths-only.",
        ),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Browse ``scan_error`` rows: which files failed, when, and why."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    sql = (
        "SELECT se.id, se.observed_at_ns, fp.value AS abs_path, se.error_kind, "
        "       se.error_message, se.size_bytes, se.scan_session_id "
        "FROM scan_error se "
        "JOIN file_path fp ON fp.id = se.file_path_id "
    )
    where: list[str] = []
    params: list[str | int] = []
    if current:
        sql += (
            "JOIN current_file cf "
            "  ON cf.abs_path  = fp.value "
            " AND cf.size_bytes = se.size_bytes "
            " AND cf.mtime_ns   = se.mtime_ns "
        )
    if folder is not None:
        clause, prefix_params = _path_prefix_where(folder)
        where.append(clause.removeprefix("WHERE ").replace("abs_path", "fp.value"))
        params.extend(prefix_params)
    if kind is not None:
        where.append("se.error_kind = ?")
        params.append(kind)
    if session is not None:
        where.append("se.scan_session_id = ?")
        params.append(session)
    if since is not None:
        where.append("se.observed_at_ns >= ?")
        params.append(_parse_time_or_exit(since, "since"))
    if until is not None:
        where.append("se.observed_at_ns <= ?")
        params.append(_parse_time_or_exit(until, "until"))
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += "ORDER BY se.observed_at_ns DESC, se.id DESC LIMIT ?"
    params.append(limit)

    try:
        with open_db(db_path, read_only=True) as conn:
            rows_sql = conn.execute(sql, params).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    rows: list[dict[str, object]] = [
        {
            "id": eid,
            "observed_at_ns": observed_at,
            "path": path,
            "kind": kind_,
            "message": msg,
            "size_bytes": size,
            "session_id": session_id,
        }
        for eid, observed_at, path, kind_, msg, size, session_id in rows_sql
    ]

    if not rows:
        if format == "table":
            console.print("[yellow]No errors[/]")
        elif format == "json":
            _stdout_line("[]")
        return 0

    if _emit_non_table(format, rows, path_key="path"):
        return 0

    table = Table(title=f"Scan errors ({len(rows):,})")
    table.add_column("Observed (UTC)")
    table.add_column("Kind", style="red")
    table.add_column("Path", overflow="fold")
    table.add_column("Message", overflow="fold")
    table.add_column("Size", justify="right", style="yellow")
    table.add_column("Session", justify="right", style="dim")
    for row in rows:
        table.add_row(
            _format_ts_ns(
                row["observed_at_ns"] if isinstance(row["observed_at_ns"], int) else None
            ),
            _describe_error_kind(str(row["kind"])),
            _format_parts_with_colors(str(row["path"])),
            str(row["message"] or "-"),
            bytes_to_human(row["size_bytes"]) if isinstance(row["size_bytes"], int) else "-",
            str(row["session_id"]) if isinstance(row["session_id"], int) else "-",
        )
    console.print(table)
    return 0


# Bucket spec → SQLite ``strftime`` format. Keep these in sync with the
# ``Literal`` on ``timeline_cmd``'s ``--bucket`` argument.
_TIMELINE_BUCKET_FORMAT: dict[str, str] = {
    "day": "%Y-%m-%d",
    "week": "%Y-W%W",
    "month": "%Y-%m",
    "year": "%Y",
}


@index_app.command(name="timeline")
def timeline_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to restrict the timeline to."),
    ] = None,
    *,
    bucket: Annotated[
        Literal["day", "week", "month", "year"],
        Parameter(name=["--bucket"], help="Time bucket size."),
    ] = "day",
    since: Annotated[
        str | None,
        Parameter(name=["--since"], help="Only files whose recording started after this instant."),
    ] = None,
    until: Annotated[
        str | None,
        Parameter(name=["--until"], help="Only files whose recording started before this instant."),
    ] = None,
    limit: Annotated[int, Parameter(name=["--limit"], help="Max buckets to print.")] = 100,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Bucketed histogram of recording activity (files, messages, bytes).

    Buckets are based on each file's ``message_start_time``. Files with a
    sub-2000 start are skipped (same gate the rest of ``index`` uses for
    duration math), so a single bad-clocked file can't poison the chart.
    """
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    path_filter_params = _optional_path_filter_params(folder)
    since_ns = _parse_time_or_exit(since, "since") if since is not None else None
    until_ns = _parse_time_or_exit(until, "until") if until is not None else None

    bucket_fmt = _TIMELINE_BUCKET_FORMAT[bucket]
    sql = (
        "SELECT "
        "strftime(?, c.message_start_time_ns / 1000000000, 'unixepoch') "
        "AS bucket, "
        "       COUNT(DISTINCT cf.abs_path)        AS files, "
        "       COALESCE(SUM(c.message_count), 0) AS messages, "
        "       COALESCE(SUM(cf.size_bytes), 0)   AS size_bytes "
        "FROM current_file cf "
        "JOIN content c ON c.id = cf.content_id "
        "WHERE c.message_start_time_ns >= ? "
        "AND (? IS NULL OR cf.abs_path = ? OR substr(cf.abs_path, 1, ?) = ?) "
        "AND (? IS NULL OR c.message_start_time_ns >= ?) "
        "AND (? IS NULL OR c.message_start_time_ns <= ?) "
        "GROUP BY bucket ORDER BY bucket ASC LIMIT ?"
    )
    params: list[str | int | None] = [
        bucket_fmt,
        SANE_EPOCH_NS,
        *path_filter_params,
        since_ns,
        since_ns,
        until_ns,
        until_ns,
        limit,
    ]

    try:
        with open_db(db_path, read_only=True) as conn:
            rows = conn.execute(sql, params).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    if not rows:
        console.print("[yellow]No data in range[/]")
        return 0

    max_files = max(r[1] for r in rows) or 1
    bar_width = 30

    table = Table(title=f"Timeline by {bucket} ({len(rows):,} buckets)")
    table.add_column(bucket.capitalize(), style="bold blue")
    table.add_column("Files", justify="right", style="green")
    table.add_column("Activity", overflow="crop")
    table.add_column("Messages", justify="right", style="green")
    table.add_column("Size", justify="right", style="yellow")
    for label, files, messages, size in rows:
        bar = "█" * max(1, round(bar_width * files / max_files)) if files else ""
        table.add_row(
            str(label),
            f"{files:,}",
            f"[cyan]{bar}[/]",
            _format_count(int(messages)) if isinstance(messages, int) else "-",
            bytes_to_human(size) if isinstance(size, int) else "-",
        )
    console.print(table)
    return 0


@index_app.command(name="info")
def info_cmd(
    target: str,
    *,
    format: Annotated[
        Literal["table", "json"],
        Parameter(name=["--format"], help="Output as Rich tables or JSON."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Show everything the index knows about TARGET (path, fingerprint, or short id)."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    try:
        with open_db(db_path, read_only=True) as conn:
            summary_fp, abs_path, err = _resolve_target_to_summary_fp(conn, target)
            if err is not None or summary_fp is None:
                console.print(f"[red]Error:[/] {err}")
                return 1

            content = conn.execute(
                "SELECT c.size_bytes, lib.name, prof.name, "
                "       c.message_count, c.schema_count, c.channel_count, "
                "       c.attachment_count, c.metadata_count, c.chunk_count, "
                "       c.message_start_time_ns, c.message_end_time_ns, c.first_seen_at_ns, "
                "       c.compression, c.compressed_size_bytes, c.uncompressed_size_bytes "
                "FROM content c "
                "LEFT JOIN library lib  ON lib.id  = c.library_id "
                "LEFT JOIN profile prof ON prof.id = c.profile_id "
                "WHERE c.summary_fingerprint = ?",
                (summary_fp,),
            ).fetchone()
            topic_rows = conn.execute(
                "SELECT cc.mcap_channel_id, t.name AS topic, "
                "       sig.schema_id, sig.message_encoding, "
                "       cc.message_count, cc.uncompressed_size_bytes, "
                "       cc.message_start_time_ns, cc.message_end_time_ns, "
                "       cc.distribution_blob "
                "FROM content_channel cc "
                "JOIN content c       ON c.id        = cc.content_id "
                "JOIN channel_signature sig ON sig.id  = cc.channel_signature_id "
                "JOIN topic t         ON t.id          = sig.topic_id "
                "WHERE c.summary_fingerprint = ? "
                "ORDER BY cc.message_count DESC NULLS LAST, t.name",
                (summary_fp,),
            ).fetchall()
            schema_dim_rows = conn.execute(
                "SELECT DISTINCT s.id, s.name, s.encoding, s.size_bytes "
                "FROM content_channel cc "
                "JOIN content c       ON c.id        = cc.content_id "
                "JOIN channel_signature sig ON sig.id  = cc.channel_signature_id "
                "JOIN schema s        ON s.id      = sig.schema_id "
                "WHERE c.summary_fingerprint = ?",
                (summary_fp,),
            ).fetchall()
            observation_rows = conn.execute(
                "SELECT fp.value AS abs_path, obs.observed_at_ns, obs.scan_session_id, "
                "       obs.file_fingerprint, c.summary_fingerprint "
                "FROM file_observation obs "
                "JOIN file_path fp ON fp.id = obs.file_path_id "
                "LEFT JOIN content c ON c.id = obs.content_id "
                "WHERE c.summary_fingerprint = ? OR fp.value = ? "
                "ORDER BY obs.observed_at_ns DESC LIMIT 20",
                (summary_fp, abs_path or ""),
            ).fetchall()
            error_rows = conn.execute(
                "SELECT fp.value AS abs_path, se.observed_at_ns, se.error_kind, se.error_message "
                "FROM scan_error se "
                "JOIN file_path fp ON fp.id = se.file_path_id "
                "WHERE fp.value = ? "
                "ORDER BY se.observed_at_ns DESC LIMIT 10",
                (abs_path or "",),
            ).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    (
        size_bytes,
        library,
        profile,
        message_count,
        schema_count,
        channel_count,
        attachment_count,
        metadata_count,
        chunk_count,
        start_ns,
        end_ns,
        first_seen_at,
        compression,
        compressed_size_bytes,
        uncompressed_size_bytes,
    ) = content

    identity = {
        "summary_fingerprint": summary_fp,
        "short_id": _short_id_from_fingerprint(summary_fp),
        "path": abs_path,
        "size_bytes": size_bytes,
        "library": library,
        "profile": profile,
        "message_count": message_count,
        "schema_count": schema_count,
        "channel_count": channel_count,
        "attachment_count": attachment_count,
        "metadata_count": metadata_count,
        "chunk_count": chunk_count,
        "message_start_time_ns": start_ns,
        "message_end_time_ns": end_ns,
        "duration_ns": (duration_ns := _safe_duration_ns(start_ns, end_ns)),
        "compression": compression,
        "compressed_size_bytes": compressed_size_bytes,
        "uncompressed_size_bytes": uncompressed_size_bytes,
        "first_seen_at_ns": first_seen_at,
    }
    schema_name_by_pk_id: dict[int, str | None] = {
        pk_id: name for pk_id, name, _enc, _sz in schema_dim_rows
    }
    topics_payload: list[_IndexedTopicPayload] = [
        {
            "channel_id": ch_id,
            "topic": topic,
            "schema_pk_id": schema_pk_id,
            "schema": schema_name_by_pk_id.get(schema_pk_id) if schema_pk_id is not None else None,
            "encoding": encoding,
            "message_count": msg_count,
            "uncompressed_size_bytes": ch_bytes,
            "message_start_time_ns": ch_start,
            "message_end_time_ns": ch_end,
            "duration_ns": _safe_duration_ns(ch_start, ch_end),
            "distribution": unpack_distribution_blob(dist_blob),
        }
        for (
            ch_id,
            topic,
            schema_pk_id,
            encoding,
            msg_count,
            ch_bytes,
            ch_start,
            ch_end,
            dist_blob,
        ) in topic_rows
    ]
    observations_payload = [
        {
            "path": obs_path,
            "observed_at_ns": observed_at,
            "session_id": session_id,
            "file_fingerprint": file_fp,
            "summary_fingerprint": obs_summary_fp,
        }
        for obs_path, observed_at, session_id, file_fp, obs_summary_fp in observation_rows
    ]
    errors_payload = [
        {
            "path": err_path,
            "observed_at_ns": observed_at,
            "kind": kind,
            "message": message,
        }
        for err_path, observed_at, kind, message in error_rows
    ]

    if format == "json":
        _stdout_line(
            _json.dumps(
                {
                    "identity": identity,
                    "topics": topics_payload,
                    "observations": observations_payload,
                    "errors": errors_payload,
                },
                default=str,
            )
        )
        return 0

    identity_table = Table.grid(padding=(0, 1))
    identity_table.add_column(style="bold blue")
    identity_table.add_column()
    identity_table.add_row("Summary fingerprint:", f"[dim]{summary_fp}[/]")
    short_id = _short_id_from_fingerprint(summary_fp)
    if short_id:
        identity_table.add_row("Short ID:", f"[bold green]{short_id}[/]")
    if abs_path is not None:
        identity_table.add_row("Path:", _format_parts_with_colors(abs_path))
    if size_bytes is not None:
        size_cell = f"[green]{bytes_to_human(size_bytes)}[/] [dim]({size_bytes:,} B)[/]"
        if isinstance(duration_ns, int) and duration_ns > 0:
            bps = size_bytes / (duration_ns / 1_000_000_000)
            size_cell += (
                f" [red]{bytes_to_human(bps)}/s[/] [orange1]{bytes_to_human(bps * 3600)}/h[/]"
            )
        identity_table.add_row("Size:", size_cell)
    else:
        identity_table.add_row("Size:", "-")
    identity_table.add_row("Library:", f"[yellow]{library or '-'}[/]")
    identity_table.add_row("Profile:", f"[yellow]{profile or '-'}[/]")
    if isinstance(message_count, int):
        identity_table.add_row("Messages:", f"[green]{_format_count(message_count)}[/]")
    else:
        identity_table.add_row("Messages:", "-")
    identity_table.add_row(
        "Schemas / Channels / Chunks:",
        f"[green]{schema_count or 0:,}[/] / "
        f"[green]{channel_count or 0:,}[/] / "
        f"[cyan]{chunk_count or 0:,}[/]",
    )
    identity_table.add_row(
        "Attachments / Metadata:",
        f"[yellow]{attachment_count or 0:,}[/] / [cyan]{metadata_count or 0:,}[/]",
    )
    identity_table.add_row("Start:", _format_ts_ns(start_ns))
    identity_table.add_row("End:", _format_ts_ns(end_ns))
    identity_table.add_row(
        "Duration:",
        f"[cyan]{_format_duration_ns(0, duration_ns) if duration_ns is not None else '-'}[/]",
    )
    identity_table.add_row(
        "Compression:",
        _format_compression_cell(
            compression,
            compressed_size_bytes,
            uncompressed_size_bytes,
        ),
    )
    console.print("[bold cyan]Identity[/]")
    console.print(identity_table)

    if topics_payload:
        console.print(f"[bold cyan]Topics ({len(topics_payload):,})[/]")
        # ``responsive=False`` so the distribution sparkline (the main reason
        # we store ``content_channel.distribution_blob``) renders regardless
        # of terminal width.
        console.print(
            display_channels_table(
                _topics_to_channel_table_data(topics_payload, schema_dim_rows, duration_ns),
                console,
                responsive=False,
                index_duration=True,
            )
        )

    if observations_payload:
        obs_table = Table(title=f"Observations ({len(observations_payload):,})")
        obs_table.add_column("Path", overflow="fold")
        obs_table.add_column("Observed at (UTC)", style="cyan")
        obs_table.add_column("Session", justify="right", style="green")
        obs_table.add_column("File fp", style="dim")
        obs_table.add_column("Summary fp", style="dim")
        for entry in observations_payload:
            observed_at = entry["observed_at_ns"]
            obs_table.add_row(
                _format_parts_with_colors(str(entry["path"])),
                _format_ts_ns(observed_at if isinstance(observed_at, int) else None),
                str(entry["session_id"]),
                str(entry["file_fingerprint"] or "-"),
                str(entry["summary_fingerprint"] or "-"),
            )
        console.print(obs_table)

    if errors_payload:
        err_table = Table(title=f"Errors ({len(errors_payload):,})")
        err_table.add_column("Path", overflow="fold")
        err_table.add_column("Observed at (UTC)", style="cyan")
        err_table.add_column("Kind", style="red")
        err_table.add_column("Message", overflow="fold", style="dim")
        for entry in errors_payload:
            observed_at = entry["observed_at_ns"]
            err_table.add_row(
                _format_parts_with_colors(str(entry["path"])),
                _format_ts_ns(observed_at if isinstance(observed_at, int) else None),
                _describe_error_kind(str(entry["kind"])),
                str(entry["message"] or "-"),
            )
        console.print(err_table)

    return 0
