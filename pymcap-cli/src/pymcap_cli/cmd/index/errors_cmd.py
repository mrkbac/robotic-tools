"""``pymcap-cli index errors`` — browse scan_error rows."""

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd.index._helpers import (
    OutputFormat,
    _describe_error_kind,
    _emit_non_table,
    _format_ts_ns,
    _parse_time_or_exit,
    _path_prefix_where,
    _print_db_needs_migration,
    _resolve_db,
    _stdout_line,
    console,
)
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db
from pymcap_cli.utils import bytes_to_human


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
