"""``pymcap-cli index timeline`` — bucketed activity histogram."""

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd.index._helpers import (
    _format_count,
    _parse_time_or_exit,
    _path_prefix_predicate,
    _print_db_needs_migration,
    _resolve_db,
    console,
)
from pymcap_cli.index import SANE_EPOCH_NS
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db
from pymcap_cli.utils import bytes_to_human

# Bucket spec → SQLite ``strftime`` format. Keep these in sync with the
# ``Literal`` on ``timeline_cmd``'s ``--bucket`` argument.
_TIMELINE_BUCKET_FORMAT: dict[str, str] = {
    "day": "%Y-%m-%d",
    "week": "%Y-W%W",
    "month": "%Y-%m",
    "year": "%Y",
}


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
    """Bucketed histogram of recording activity (files, messages, bytes)."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    path_clause = ""
    path_params: tuple[str, ...] = ()
    if folder is not None:
        predicate, path_params = _path_prefix_predicate(folder)
        path_clause = f"AND ({predicate.replace('abs_path', 'cf.abs_path')}) "

    since_ns = _parse_time_or_exit(since, "since") if since is not None else None
    until_ns = _parse_time_or_exit(until, "until") if until is not None else None

    bucket_fmt = _TIMELINE_BUCKET_FORMAT[bucket]
    sql = (
        "SELECT "  # noqa: S608
        "strftime(?, c.message_start_time_ns / 1000000000, 'unixepoch') "
        "AS bucket, "
        "       COUNT(DISTINCT cf.abs_path)        AS files, "
        "       COALESCE(SUM(c.message_count), 0) AS messages, "
        "       COALESCE(SUM(cf.size_bytes), 0)   AS size_bytes "
        "FROM current_file cf "
        "JOIN content c ON c.id = cf.content_id "
        "WHERE c.message_start_time_ns >= ? "
        f"{path_clause}"
        "AND (? IS NULL OR c.message_start_time_ns >= ?) "
        "AND (? IS NULL OR c.message_start_time_ns <= ?) "
        "GROUP BY bucket ORDER BY bucket ASC LIMIT ?"
    )
    params: list[str | int | None] = [
        bucket_fmt,
        SANE_EPOCH_NS,
        *path_params,
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
