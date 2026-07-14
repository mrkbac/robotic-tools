"""``pymcap-cli index schemas`` — schema-level rollup."""

from typing import Annotated

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd._cli_options import (
    IndexDbOption,
    IndexLimitOption,
    IndexMinFilesOption,
    IndexSchemaSortOption,
    IndexTableJsonFormatOption,
)
from pymcap_cli.cmd.index._helpers import (
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
    sort_by: IndexSchemaSortOption = "files",
    limit: IndexLimitOption = 50,
    min_files: IndexMinFilesOption = 1,
    format: IndexTableJsonFormatOption = "table",
    db: IndexDbOption = None,
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

    params: list[str | int] = []
    if prefix is not None:
        sql = (
            "WITH selected_schema AS MATERIALIZED ("
            "  SELECT id, name, encoding FROM schema WHERE name LIKE ? ESCAPE '\\'"
            "), schema_content AS MATERIALIZED ("
            "  SELECT DISTINCT cs.content_id, cs.schema_id "
            "  FROM selected_schema s "
            "  JOIN content_schema cs INDEXED BY content_schema_schema_id "
            "    ON cs.schema_id = s.id "
            "  JOIN content_current_file_count cfc ON cfc.content_id = cs.content_id"
            "), schema_files AS MATERIALIZED ("
            "  SELECT schema_content.schema_id, SUM(cfc.file_count) AS files "
            "  FROM schema_content "
            "  JOIN content_current_file_count cfc "
            "    ON cfc.content_id = schema_content.content_id "
            "  GROUP BY schema_content.schema_id"
            "), schema_channel_rollup AS MATERIALIZED ("
            "  SELECT sig.schema_id, sig.topic_id, "
            "         COALESCE(SUM(COALESCE(cc.message_count, 0) * cfc.file_count), 0) "
            "           AS messages "
            "  FROM selected_schema s "
            "  JOIN channel_signature sig INDEXED BY channel_signature_schema_id "
            "    ON sig.schema_id = s.id "
            "  JOIN content_channel cc INDEXED BY content_channel_sig_content_msg "
            "    ON cc.channel_signature_id = sig.id "
            "  JOIN content_current_file_count cfc ON cfc.content_id = cc.content_id "
            "  GROUP BY sig.schema_id, sig.topic_id"
            ") "
            "SELECT s.name, s.encoding, "
            "       schema_files.files AS files, "
            "       COUNT(schema_channel_rollup.topic_id) AS topics, "
            "       COALESCE(SUM(schema_channel_rollup.messages), 0) AS messages "
            "FROM schema_files "
            "JOIN selected_schema s ON s.id = schema_files.schema_id "
            "LEFT JOIN schema_channel_rollup "
            "  ON schema_channel_rollup.schema_id = schema_files.schema_id "
        )
        params.append(_like_prefix_param(prefix))
    else:
        sql = (
            "WITH schema_files AS MATERIALIZED ("
            "  SELECT schema_id, SUM(file_count) AS files "
            "  FROM ("
            "    SELECT cs.schema_id, cs.content_id, MAX(cfc.file_count) AS file_count "
            "    FROM content_schema cs INDEXED BY content_schema_schema_content "
            "    JOIN content_current_file_count cfc ON cfc.content_id = cs.content_id "
            "    GROUP BY cs.schema_id, cs.content_id"
            "  ) "
            "  GROUP BY schema_id"
            "), signature_messages AS MATERIALIZED ("
            "  SELECT cc.channel_signature_id, "
            "         COALESCE(SUM(COALESCE(cc.message_count, 0) * cfc.file_count), 0) "
            "           AS messages "
            "  FROM content_channel cc INDEXED BY content_channel_sig_content_msg "
            "  JOIN content_current_file_count cfc ON cfc.content_id = cc.content_id "
            "  GROUP BY cc.channel_signature_id"
            "), schema_channel_rollup AS MATERIALIZED ("
            "  SELECT sig.schema_id, sig.topic_id, SUM(signature_messages.messages) AS messages "
            "  FROM signature_messages "
            "  JOIN channel_signature sig ON sig.id = signature_messages.channel_signature_id "
            "  WHERE sig.schema_id IS NOT NULL "
            "  GROUP BY sig.schema_id, sig.topic_id"
            ") "
            "SELECT s.name, s.encoding, "
            "       schema_files.files AS files, "
            "       COUNT(schema_channel_rollup.topic_id) AS topics, "
            "       COALESCE(SUM(schema_channel_rollup.messages), 0) AS messages "
            "FROM schema_files "
            "JOIN schema s ON s.id = schema_files.schema_id "
            "LEFT JOIN schema_channel_rollup "
            "  ON schema_channel_rollup.schema_id = schema_files.schema_id "
        )
    sql += (
        "GROUP BY s.id, s.name, s.encoding, schema_files.files "
        f"HAVING files >= ? ORDER BY {order_by} LIMIT ?"
    )
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
