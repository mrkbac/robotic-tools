"""``pymcap-cli index schemas`` — schema-level rollup."""

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
        "WITH content_schema_files AS MATERIALIZED ("
        "  SELECT DISTINCT cs.content_id, cs.schema_id, cfc.file_count "
        "  FROM content_current_file_count cfc "
        "  JOIN content_schema cs ON cs.content_id = cfc.content_id"
        "), schema_messages AS MATERIALIZED ("
        "  SELECT cc.content_id, sig.schema_id, "
        "         SUM(COALESCE(cc.message_count, 0)) AS messages "
        "  FROM content_current_file_count cfc "
        "  JOIN content_channel cc ON cc.content_id = cfc.content_id "
        "  JOIN channel_signature sig ON sig.id = cc.channel_signature_id "
        "  WHERE sig.schema_id IS NOT NULL "
        "  GROUP BY cc.content_id, sig.schema_id"
        "), schema_topic_counts AS MATERIALIZED ("
        "  SELECT schema_id, COUNT(DISTINCT topic_id) AS topics "
        "  FROM ("
        "    SELECT DISTINCT sig.schema_id, sig.topic_id "
        "    FROM content_current_file_count cfc "
        "    JOIN content_channel cc ON cc.content_id = cfc.content_id "
        "    JOIN channel_signature sig ON sig.id = cc.channel_signature_id "
        "    WHERE sig.schema_id IS NOT NULL"
        "  ) "
        "  GROUP BY schema_id"
        ") "
        "SELECT s.name, s.encoding, "
        "       SUM(content_schema_files.file_count) AS files, "
        "       COALESCE(schema_topic_counts.topics, 0) AS topics, "
        "       COALESCE(SUM(COALESCE(schema_messages.messages, 0) "
        "                    * content_schema_files.file_count), 0) "
        "                                             AS messages "
        "FROM content_schema_files "
        "JOIN schema s ON s.id = content_schema_files.schema_id "
        "LEFT JOIN schema_messages "
        "  ON schema_messages.content_id = content_schema_files.content_id "
        " AND schema_messages.schema_id = content_schema_files.schema_id "
        "LEFT JOIN schema_topic_counts "
        "  ON schema_topic_counts.schema_id = content_schema_files.schema_id "
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
