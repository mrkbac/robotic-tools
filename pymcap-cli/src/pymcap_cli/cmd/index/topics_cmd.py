"""``pymcap-cli index topics`` — topic-level rollup."""

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
from pymcap_cli.display.display_utils import _format_parts_with_colors, _format_schema_with_link
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db


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

    order_by = {
        "files": "files DESC, messages DESC, t.name",
        "messages": "messages DESC, files DESC, t.name",
        "schemas": "schemas DESC, files DESC, t.name",
        "name": "t.name",
    }[sort_by]

    sql = (
        "SELECT t.name AS topic, "
        "       SUM(cfc.file_count)                AS files, "
        "       COALESCE(SUM(COALESCE(cc.message_count, 0) * cfc.file_count), 0) "
        "                                             AS messages, "
        "       COUNT(DISTINCT sig.schema_id)   AS schemas, "
        "       MIN(s.name)                        AS schema_name "
        "FROM topic t "
        "CROSS JOIN channel_signature sig INDEXED BY channel_signature_topic_id "
        "CROSS JOIN content_channel cc INDEXED BY content_channel_sig_content_msg "
        "CROSS JOIN content_current_file_count cfc "
        "LEFT JOIN schema s      ON s.id     = sig.schema_id "
        "WHERE sig.topic_id = t.id "
        "  AND cc.channel_signature_id = sig.id "
        "  AND cfc.content_id = cc.content_id "
    )
    params: list[str | int] = []
    if prefix is not None:
        sql += "AND t.name LIKE ? ESCAPE '\\' "
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
