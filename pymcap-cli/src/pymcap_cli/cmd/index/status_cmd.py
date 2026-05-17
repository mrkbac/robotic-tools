"""``pymcap-cli index status`` — coverage stats for the sidecar DB."""

import sqlite3
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd.index._helpers import (
    _connect_status_db,
    _describe_error_kind,
    _format_duration_ns,
    _format_optional_compact_count,
    _format_optional_count,
    _format_ts_ns,
    _format_user_version,
    _path_prefix_predicate,
    _resolve_db,
    _status_fetchall,
    _status_fetchone,
    _status_int,
    console,
)
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.utils import bytes_to_human


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

    current_file_where = ""
    current_file_params: tuple[str, ...] = ()
    scan_error_where = ""
    scan_error_params: tuple[str, ...] = ()
    if folder is not None:
        predicate, prefix_params = _path_prefix_predicate(folder)
        current_file_where = f"WHERE {predicate.replace('abs_path', 'cf.abs_path')}"
        scan_error_where = f"WHERE {predicate.replace('abs_path', 'fp.value')}"
        current_file_params = prefix_params
        scan_error_params = prefix_params

    warnings: list[str] = []
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

        summary_sql = (
            "SELECT COUNT(DISTINCT cf.abs_path), "  # noqa: S608
            "       COUNT(DISTINCT CASE WHEN cf.content_id IS NOT NULL THEN cf.abs_path END), "
            "       COUNT(DISTINCT cf.content_id), "
            "       COALESCE(SUM(cf.size_bytes), 0), "
            "       COALESCE(SUM(c.message_count), 0) "
            "FROM current_file cf "
            "LEFT JOIN content c ON c.id = cf.content_id "
            f"{current_file_where}"
        )
        summary_row = _status_fetchone(
            conn,
            "Summary metrics",
            summary_sql,
            current_file_params,
            warnings,
        )
        if summary_row is None:
            files = None
            with_content = None
            contents = None
            total_bytes = None
            total_messages = None
        else:
            files = int(summary_row[0] or 0)
            with_content = int(summary_row[1] or 0)
            contents = int(summary_row[2] or 0)
            total_bytes = int(summary_row[3] or 0)
            total_messages = int(summary_row[4] or 0)

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
            scan_error_count_sql = (
                "SELECT COUNT(*) FROM scan_error se "  # noqa: S608
                "JOIN file_path fp ON fp.id = se.file_path_id "
                f"{scan_error_where}"
            )
            errors = _status_int(
                conn,
                "Scan errors recorded",
                scan_error_count_sql,
                scan_error_params,
                warnings,
            )
            scan_error_breakdown_sql = (
                "SELECT se.error_kind, COUNT(*) FROM scan_error se "  # noqa: S608
                "JOIN file_path fp ON fp.id = se.file_path_id "
                f"{scan_error_where} "
                "GROUP BY se.error_kind ORDER BY se.error_kind"
            )
            error_breakdown = [
                (str(kind), int(count or 0))
                for kind, count in _status_fetchall(
                    conn,
                    "Scan error breakdown",
                    scan_error_breakdown_sql,
                    scan_error_params,
                    warnings,
                )
            ]

        shape_sql = (
            "SELECT COUNT(DISTINCT sig.topic_id), "  # noqa: S608
            "       COUNT(DISTINCT sig.schema_id), "
            "       COUNT(*) "
            "FROM current_file cf "
            "JOIN content_channel cc ON cc.content_id = cf.content_id "
            "JOIN channel_signature sig ON sig.id = cc.channel_signature_id "
            f"{current_file_where}"
        )
        shape_row = _status_fetchone(
            conn,
            "Dataset shape",
            shape_sql,
            current_file_params,
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
