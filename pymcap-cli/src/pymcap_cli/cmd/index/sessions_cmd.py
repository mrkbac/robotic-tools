"""``pymcap-cli index sessions`` — list scan_session rows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd.index._helpers import (
    OutputFormat,
    _emit_non_table,
    _format_duration_ns,
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
