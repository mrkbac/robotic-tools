"""``pymcap-cli index query`` — look up indexed files."""

from typing import Annotated, Literal

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd._cli_options import (
    AtTimeOption,
    IndexDbOption,
    IndexFolderOption,
    IndexLimitOption,
    IndexSinceOption,
    IndexTableJsonPathsFormatOption,
    IndexUntilOption,
)
from pymcap_cli.cmd.index._helpers import (
    _EFF_END_SQL,
    _EFF_START_SQL,
    _MAX_PLAUSIBLE_DURATION_NS,
    _emit_non_table,
    _format_duration_ns,
    _format_ts_ns,
    _parse_time_or_exit,
    _path_prefix_where,
    _print_db_needs_migration,
    _resolve_db,
    _safe_duration_ns,
    _short_id_from_fingerprint,
    _stdout_line,
    console,
)
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.index import SANE_EPOCH_NS
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db


def query_cmd(
    folder: IndexFolderOption = None,
    *,
    sort_by: Annotated[
        Literal[
            "discovered",
            "observed",
            "modified",
            "path",
            "start",
            "end",
            "size",
            "messages",
            "duration",
        ],
        Parameter(
            name=["--sort-by", "-s"],
            help="Sort results (descending except for ``path``).",
        ),
    ] = "discovered",
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
    at: AtTimeOption = None,
    since: IndexSinceOption = None,
    until: IndexUntilOption = None,
    limit: IndexLimitOption = 50,
    format: IndexTableJsonPathsFormatOption = "table",
    db: IndexDbOption = None,
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

    _safe_dur_sql = (
        f"CASE WHEN {_EFF_START_SQL} >= {SANE_EPOCH_NS}"
        f"      AND ({_EFF_END_SQL} - {_EFF_START_SQL}) BETWEEN 0 AND {_MAX_PLAUSIBLE_DURATION_NS}"
        f"     THEN ({_EFF_END_SQL} - {_EFF_START_SQL}) ELSE 0 END"
    )
    _safe_start_sql = (
        f"CASE WHEN {_EFF_START_SQL} >= {SANE_EPOCH_NS} THEN {_EFF_START_SQL} ELSE NULL END"
    )
    _safe_end_sql = f"CASE WHEN {_EFF_END_SQL} >= {SANE_EPOCH_NS} THEN {_EFF_END_SQL} ELSE NULL END"
    order_by = {
        "discovered": "c.first_seen_at_ns DESC, cf.abs_path",
        "observed": "cf.observed_at_ns DESC, cf.abs_path",
        "modified": "cf.mtime_ns DESC, cf.abs_path",
        "path": "cf.abs_path",
        "duration": f"{_safe_dur_sql} DESC, cf.abs_path",
        "messages": "messages DESC, cf.abs_path"
        if (topic is not None or schema is not None)
        else "c.message_count DESC, cf.abs_path",
        "size": "cf.size_bytes DESC, cf.abs_path",
        "start": f"{_safe_start_sql} DESC, cf.abs_path",
        "end": f"{_safe_end_sql} DESC, cf.abs_path",
    }[sort_by]

    folder_clause = ""
    folder_params: tuple[str | int, ...] = ()
    if folder is not None:
        folder_where, folder_params = _path_prefix_where(folder)
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
