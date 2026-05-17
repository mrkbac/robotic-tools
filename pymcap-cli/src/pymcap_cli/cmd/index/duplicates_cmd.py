"""``pymcap-cli index duplicates`` — find duplicated content rows."""

import json as _json
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd.index._helpers import (
    OutputFormat,
    _path_prefix_predicate,
    _print_db_needs_migration,
    _resolve_db,
    _stdout_line,
    console,
)
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db
from pymcap_cli.utils import bytes_to_human


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
