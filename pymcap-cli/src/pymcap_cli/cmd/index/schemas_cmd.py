"""``pymcap-cli index schemas`` — schema-level rollup."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd.index._helpers import (
    OutputFormat,
    _emit_non_table,
    _format_count,
    _like_prefix_param,
    _print_db_needs_migration,
    _resolve_db,
    _stdout_line,
    console,
)
from pymcap_cli.display.display_utils import _format_schema_with_link
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db


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
