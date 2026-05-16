"""`pymcap-cli index` — sidecar catalog of MCAP summaries."""

from __future__ import annotations

import importlib.metadata
import json as _json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from collections.abc import Sequence
from dataclasses import dataclass, field

from cyclopts import App, Parameter
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

from pymcap_cli.display.display_utils import (
    _format_parts_with_colors,
    _format_schema_with_link,
    _text_to_color,
)
from pymcap_cli.index.db import default_db_path, open_db
from pymcap_cli.index.scanner import ScanStats, scan
from pymcap_cli.utils import bytes_to_human, parse_time_arg

OutputFormat = Literal["table", "json", "paths-only"]

logger = logging.getLogger(__name__)
console = Console()

index_app = App(
    name="index",
    help="Maintain a sidecar SQLite catalog of MCAP summaries for fast recovery.",
)


_ERROR_KIND_LABEL: dict[str, str] = {
    "io": "io — file could not be read",
    "corrupt": "corrupt — not a valid MCAP",
    "no_summary": "no_summary — no usable summary; rerun scan with --rebuild-missing",
}


# Anything before 2000-01-01 UTC almost certainly comes from an uninitialised
# clock (e.g. ROS time before NTP sync) and would inflate any duration
# aggregate by decades. We use it to gate duration math without changing what
# is stored in the catalog.
_SANE_EPOCH_NS = 946_684_800 * 1_000_000_000  # 2000-01-01T00:00:00Z

# Pick whichever value (raw vs precomputed sane fallback) is post-epoch. The
# ``sane_message_*`` columns are populated by the scanner from
# ``ChunkIndex`` records, with the same threshold applied (see
# ``scanner._build_content_row``).
_EFF_START_SQL = (
    f"CASE WHEN c.message_start_time >= {_SANE_EPOCH_NS} "
    "THEN c.message_start_time ELSE c.sane_message_start_time END"
)
_EFF_END_SQL = (
    f"CASE WHEN c.message_start_time >= {_SANE_EPOCH_NS} "
    "THEN c.message_end_time ELSE c.sane_message_end_time END"
)


def _safe_duration_ns(start_ns: int | None, end_ns: int | None) -> int | None:
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return None
    if start_ns < _SANE_EPOCH_NS:
        return None
    return end_ns - start_ns


def _pymcap_cli_version() -> str:
    try:
        return importlib.metadata.version("pymcap-cli")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_db(db: Path | None) -> Path:
    return (db or default_db_path()).expanduser()


def _path_prefix_where(path: Path) -> tuple[str, tuple[str | int, ...]]:
    """Build a prefix-match WHERE clause that is safe against LIKE wildcards.

    Uses ``substr(abs_path, 1, ?) = ?`` so paths containing ``_`` or ``%`` do
    not accidentally match unrelated siblings.
    """
    resolved = str(path.expanduser().resolve())
    child_prefix = resolved if resolved.endswith(os.sep) else f"{resolved}{os.sep}"
    return (
        "WHERE (abs_path = ? OR substr(abs_path, 1, ?) = ?)",
        (resolved, len(child_prefix), child_prefix),
    )


def _format_count(n: int) -> str:
    """Compact human-readable count (`19`, `258.0K`, `14.7M`, `1.2B`)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _format_ts_ns(time_ns: int | None) -> str:
    """Render UTC timestamp as ``<colored date> <cyan time>``.

    Date gets a deterministic color so different days visually pop; the time
    portion is uniform cyan to keep the row readable.
    """
    if time_ns is None or time_ns == 0:
        return "-"
    dt = datetime.fromtimestamp(time_ns / 1e9, tz=timezone.utc)
    date = dt.strftime("%Y-%m-%d")
    time = dt.strftime("%H:%M:%S")
    return f"[{_text_to_color(date)}]{date}[/] [cyan]{time}[/]"


def _format_duration_ns(start_ns: int | None, end_ns: int | None) -> str:
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return "-"
    secs = (end_ns - start_ns) / 1e9
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"
    if secs < 86400:
        h, rem = divmod(int(secs), 3600)
        return f"{h}h {rem // 60}m"
    d, rem = divmod(int(secs), 86400)
    return f"{d}d {rem // 3600}h"


def _describe_error_kind(kind: str) -> str:
    return _ERROR_KIND_LABEL.get(kind, kind)


def _path_prefix_predicate(path: Path) -> tuple[str, tuple[str | int, ...]]:
    """Same shape as ``_path_prefix_where`` but without the leading ``WHERE``."""
    where, params = _path_prefix_where(path)
    return where.removeprefix("WHERE "), params


def _like_prefix_param(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def _stdout_line(text: str) -> None:
    sys.stdout.write(f"{text}\n")


def _emit_non_table(
    fmt: OutputFormat,
    rows: list[dict[str, object]],
    path_key: str = "path",
) -> bool:
    """If fmt is json or paths-only, emit to stdout and return True.

    Returns False for ``table`` so the caller can build a Rich table itself.
    Non-table output bypasses the Rich console so shell pipes get clean text.
    """
    if fmt == "json":
        _stdout_line(_json.dumps(rows, default=str))
        return True
    if fmt == "paths-only":
        for row in rows:
            value = row.get(path_key)
            if value is not None:
                _stdout_line(str(value))
        return True
    return False


def _parse_time_or_exit(time_str: str, flag: str) -> int:
    try:
        return parse_time_arg(time_str)
    except ValueError as exc:
        console.print(f"[red]Error:[/] --{flag}: {exc}")
        raise SystemExit(2) from exc


@index_app.command(name="scan")
def scan_cmd(
    folder: Path,
    *,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
    jobs: Annotated[
        int,
        Parameter(name=["-j", "--jobs"], help="Parallel I/O workers."),
    ] = 8,
    retry_errors: Annotated[
        bool,
        Parameter(
            name=["--retry-errors"],
            help="Retry files previously recorded in scan_error.",
        ),
    ] = False,
    rebuild_missing: Annotated[
        bool,
        Parameter(
            help=(
                "Rebuild summaries in memory for files without usable summary data. "
                "This can read entire MCAP files."
            ),
        ),
    ] = False,
    root_only: Annotated[
        bool,
        Parameter(name=["--root-only"], help="Do not recurse into subfolders."),
    ] = False,
) -> int:
    """Scan FOLDER recursively and index every .mcap into the sidecar DB.

    Unchanged files (matching path + size + mtime) are skipped without any
    file I/O. Files that moved or were duplicated can reuse an existing
    summary snapshot via the cheap byte probe.
    """
    folder = folder.expanduser().resolve()
    if not folder.exists():
        console.print(f"[red]Error:[/] {folder} does not exist")
        return 1

    db_path = _resolve_db(db)
    console.print(f"[dim]DB:[/dim] {db_path}")
    console.print(f"[dim]Root:[/dim] {folder}")

    with Progress(
        TextColumn("[bold blue]Scanning"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn(
            "[dim]walked={task.fields[walked]} dirs[/] "
            "[green]indexed={task.fields[indexed]}[/] "
            "[cyan]reused={task.fields[reused]}[/] "
            "[dim]skip={task.fields[skip]}[/] "
            "[red]err={task.fields[errored]}[/]"
        ),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress_bar:
        task_id = progress_bar.add_task(
            "scan", total=None, walked=0, indexed=0, reused=0, skip=0, errored=0
        )
        seen = 0

        def _progress(stats: ScanStats) -> None:
            nonlocal seen
            total = stats.discovered + stats.deleted
            if progress_bar.tasks[task_id].total != total:
                progress_bar.update(task_id, total=total)
            done = (
                stats.deleted
                + stats.stat_skipped
                + stats.error_skipped
                + stats.indexed
                + stats.fingerprint_reused
                + stats.errored
            )
            progress_bar.update(
                task_id,
                advance=done - seen,
                walked=stats.dirs_walked,
                indexed=stats.indexed,
                reused=stats.fingerprint_reused,
                skip=stats.stat_skipped + stats.error_skipped,
                errored=stats.errored,
            )
            seen = done

        with open_db(db_path) as conn:
            stats = scan(
                folder,
                conn,
                pymcap_cli_version=_pymcap_cli_version(),
                jobs=jobs,
                recurse=not root_only,
                retry_errors=retry_errors,
                rebuild_missing=rebuild_missing,
                progress=_progress,
            )

    table = Table(title="Scan summary", show_header=False)
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Discovered", f"{stats.discovered:,}")
    table.add_row("Stat-skipped (unchanged)", f"{stats.stat_skipped:,}")
    table.add_row("Error-skipped (previously failed)", f"{stats.error_skipped:,}")
    table.add_row("Fingerprint-reused (moved/dup)", f"{stats.fingerprint_reused:,}")
    table.add_row("Indexed (new content)", f"{stats.indexed:,}")
    table.add_row("Deleted/stale paths", f"{stats.deleted:,}")
    table.add_row("Errored", f"{stats.errored:,}")
    for kind, count in sorted(stats.errored_by_kind.items()):
        table.add_row(f"  [dim]└ {_describe_error_kind(kind)}[/]", f"{count:,}")
    console.print(table)
    return 0


@index_app.command(name="status")
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

    where = ""
    params: tuple[str | int, ...] = ()
    if folder is not None:
        where, params = _path_prefix_where(folder)

    with open_db(db_path, read_only=True) as conn:
        files = conn.execute(
            f"SELECT COUNT(DISTINCT abs_path) FROM current_file {where}",  # noqa: S608
            params,
        ).fetchone()[0]
        with_content = conn.execute(
            f"SELECT COUNT(DISTINCT abs_path) FROM current_file {where} "  # noqa: S608
            f"{'AND' if where else 'WHERE'} summary_fingerprint IS NOT NULL",
            params,
        ).fetchone()[0]
        contents_sql = f"SELECT COUNT(*) FROM content WHERE summary_fingerprint IN (SELECT summary_fingerprint FROM current_file {where})"  # noqa: S608, E501
        contents = conn.execute(contents_sql, params).fetchone()[0]
        errors = conn.execute(
            f"SELECT COUNT(*) FROM scan_error {where}",  # noqa: S608
            params,
        ).fetchone()[0]
        totals_sql = (
            "SELECT COALESCE(SUM(c.message_count),0), COALESCE(SUM(cf.size_bytes),0) "
            "FROM current_file cf "
            "JOIN content c ON c.content_id = cf.content_id"
        )
        if where:
            totals_sql += " WHERE (abs_path = ? OR substr(abs_path, 1, ?) = ?)"
        total_messages, total_bytes = conn.execute(totals_sql, params).fetchone()
        error_breakdown = conn.execute(
            f"SELECT error_kind, COUNT(*) FROM scan_error {where} GROUP BY error_kind ORDER BY error_kind",  # noqa: S608, E501
            params,
        ).fetchall()

    coverage = f"{with_content:,} / {files:,}" if files else "0 / 0"
    if files:
        coverage += f"  [dim]({with_content * 100 // files}%)[/]"

    table = Table(title="Index status", show_header=False)
    table.add_column(style="bold blue")
    table.add_column()
    table.add_row("DB", f"[green]{db_path}[/]")
    if folder is not None:
        table.add_row("Filter", f"[green]{folder}[/]")
    table.add_row("Files tracked", f"[green]{files:,}[/]")
    table.add_row("Files with summary", f"[green]{coverage}[/]")
    table.add_row("Distinct content rows", f"[green]{contents:,}[/]")
    table.add_row(
        "Scan errors recorded",
        f"[red]{errors:,}[/]" if errors else f"[green]{errors:,}[/]",
    )
    for kind, count in error_breakdown:
        table.add_row(f"  [dim]└ {_describe_error_kind(kind)}[/]", f"[dim]{count:,}[/]")
    table.add_row("Total messages", f"[green]{_format_count(int(total_messages))}[/]")
    table.add_row(
        "Total bytes",
        f"[yellow]{bytes_to_human(total_bytes)}[/]  [dim]({total_bytes:,} B)[/]",
    )
    console.print(table)
    return 0


@dataclass
class _PathNode:
    """Aggregate stats for one directory in the path tree."""

    file_count: int = 0
    size_bytes: int = 0
    message_count: int = 0
    duration_ns: int = 0
    topics: set[str] = field(default_factory=set)
    schemas: set[str] = field(default_factory=set)
    children: dict[str, "_PathNode"] = field(default_factory=dict)


def _format_seconds_short(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        m, s = divmod(int(secs), 60)
        return f"{m}m{s}s"
    if secs < 86400:
        h, rem = divmod(int(secs), 3600)
        return f"{h}h{rem // 60}m"
    d, rem = divmod(int(secs), 86400)
    return f"{d}d{rem // 3600}h"


def _format_node_stats(node: _PathNode) -> str:
    parts = [
        f"[yellow]{bytes_to_human(node.size_bytes)}[/]",
        f"[green]{_format_count(node.file_count)}f[/]",
        f"[cyan]{_format_count(node.message_count)}msg[/]",
    ]
    if node.duration_ns:
        parts.append(f"[magenta]{_format_seconds_short(node.duration_ns / 1e9)}[/]")
    if node.topics:
        parts.append(f"[blue]{len(node.topics)} topics[/]")
    if node.schemas:
        parts.append(f"[dim]{len(node.schemas)} schemas[/]")
    return "  ".join(parts)


def _build_path_tree(
    files: Sequence[tuple[str, int | None, int | None, int | None, int | None]],
    topics: Sequence[tuple[str, str]],
    schemas: Sequence[tuple[str, str]],
    root_prefix: str,
) -> _PathNode:
    """Group rows by their path prefix and accumulate stats up the chain."""
    root = _PathNode()
    chain_cache: dict[str, list[_PathNode]] = {}

    def _chain_for(abs_path: str) -> list[_PathNode]:
        cached = chain_cache.get(abs_path)
        if cached is not None:
            return cached
        try:
            rel = os.path.relpath(abs_path, root_prefix) if root_prefix else abs_path
        except ValueError:
            rel = abs_path
        parts = [p for p in rel.split(os.sep) if p and p != "."]
        # Last component is the filename — only its ancestor directories
        # get nodes.
        chain = [root]
        node = root
        for part in parts[:-1]:
            node = node.children.setdefault(part, _PathNode())
            chain.append(node)
        chain_cache[abs_path] = chain
        return chain

    for abs_path, size, msg_count, ts_start, ts_end in files:
        chain = _chain_for(abs_path)
        size_v = size or 0
        msg_v = msg_count or 0
        dur_v = _safe_duration_ns(ts_start, ts_end) or 0
        for node in chain:
            node.file_count += 1
            node.size_bytes += size_v
            node.message_count += msg_v
            node.duration_ns += dur_v

    for abs_path, topic in topics:
        for node in _chain_for(abs_path):
            node.topics.add(topic)

    for abs_path, schema_hash in schemas:
        for node in _chain_for(abs_path):
            node.schemas.add(schema_hash)

    return root


_TREE_SORT_KEYS = {
    "size": lambda kv: (-kv[1].size_bytes, kv[0]),
    "files": lambda kv: (-kv[1].file_count, kv[0]),
    "messages": lambda kv: (-kv[1].message_count, kv[0]),
    "duration": lambda kv: (-kv[1].duration_ns, kv[0]),
    "name": lambda kv: (0.0, kv[0]),
}


def _fold_single_child_chain(name: str, node: _PathNode) -> tuple[str, _PathNode]:
    """Collapse ``a/ -> b/ -> c/`` chains where each level has exactly one child."""
    while len(node.children) == 1:
        only_name, only_child = next(iter(node.children.items()))
        name = f"{name}/{only_name}"
        node = only_child
    return name, node


def _render_path_tree(
    root: _PathNode,
    root_label: str,
    *,
    max_depth: int,
    min_files: int,
    sort_by: str,
) -> Tree:
    sort_key = _TREE_SORT_KEYS[sort_by]
    folded_root_label, folded_root = _fold_single_child_chain(root_label, root)
    tree = Tree(
        f"[bold blue]{folded_root_label}[/]  [dim]→[/]  {_format_node_stats(folded_root)}"
    )

    def _add(parent: Tree, node: _PathNode, depth: int) -> None:
        if depth >= max_depth:
            if node.children:
                parent.add(f"[dim]… {len(node.children):,} subdirs collapsed[/]")
            return
        for name, child in sorted(node.children.items(), key=sort_key):
            if child.file_count < min_files:
                continue
            display_name, display_child = _fold_single_child_chain(name, child)
            sub = parent.add(
                f"[bold]{display_name}/[/]  [dim]→[/]  {_format_node_stats(display_child)}"
            )
            _add(sub, display_child, depth + 1)

    _add(tree, folded_root, 0)
    return tree


@index_app.command(name="tree")
def tree_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to restrict the tree to."),
    ] = None,
    *,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
    max_depth: Annotated[
        int,
        Parameter(
            help="Limit how many directory levels to render. Aggregates still cover everything below.",
        ),
    ] = 4,
    min_files: Annotated[
        int,
        Parameter(help="Hide directories containing fewer than this many .mcap files."),
    ] = 1,
    sort_by: Annotated[
        Literal["size", "files", "messages", "duration", "name"],
        Parameter(help="Sort children of each node by this metric (descending)."),
    ] = "size",
) -> int:
    """Show a directory-tree breakdown of indexed data.

    Each node aggregates everything below it: total size on disk, number of
    indexed files, total message count, total duration, and the number of
    distinct topics / schemas that appear under the prefix.
    """
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    where = ""
    params: tuple[str | int, ...] = ()
    if folder is not None:
        where, params = _path_prefix_where(folder)

    with open_db(db_path, read_only=True) as conn:
        files = conn.execute(
            f"""SELECT cf.abs_path, cf.size_bytes, c.message_count,
                       {_EFF_START_SQL} AS eff_start,
                       {_EFF_END_SQL}   AS eff_end
                FROM current_file cf
                JOIN content c ON c.content_id = cf.content_id
                {where}""",  # noqa: S608
            params,
        ).fetchall()
        topics = conn.execute(
            f"""SELECT cf.abs_path, t.name
                FROM current_file cf
                JOIN content_channel cc ON cc.content_id     = cf.content_id
                JOIN channel_sig sig    ON sig.channel_sig_id = cc.channel_sig_id
                JOIN topic t            ON t.topic_id         = sig.topic_id
                {where}""",  # noqa: S608
            params,
        ).fetchall()
        schema_rows = conn.execute(
            f"""SELECT cf.abs_path, s.schema_hash
                FROM current_file cf
                JOIN content_schema cs ON cs.content_id = cf.content_id
                JOIN schema s          ON s.schema_pk_id = cs.schema_pk_id
                {where}""",  # noqa: S608
            params,
        ).fetchall()

    if not files:
        console.print("[dim]No indexed files match.[/]")
        return 0

    if folder is not None:
        root_prefix = str(folder.expanduser().resolve())
    else:
        try:
            root_prefix = os.path.commonpath([row[0] for row in files])
        except ValueError:
            root_prefix = ""

    root_node = _build_path_tree(files, topics, schema_rows, root_prefix)
    rendered = _render_path_tree(
        root_node,
        root_label=root_prefix or "(all)",
        max_depth=max_depth,
        min_files=min_files,
        sort_by=sort_by,
    )
    console.print(rendered)
    return 0


@index_app.command(name="query")
def query_cmd(
    folder: Annotated[
        Path | None,
        Parameter(help="Optional path prefix to restrict results to."),
    ] = None,
    *,
    sort_by: Annotated[
        Literal["path", "duration", "messages", "size", "start"],
        Parameter(
            name=["--sort-by"],
            help="Sort results (descending except for ``path``).",
        ),
    ] = "path",
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
    at: Annotated[
        str | None,
        Parameter(
            name=["--at"],
            help="Match files whose time range contains this instant (ns or RFC3339).",
        ),
    ] = None,
    since: Annotated[
        str | None,
        Parameter(
            name=["--since"],
            help="Match files overlapping the window starting at this instant.",
        ),
    ] = None,
    until: Annotated[
        str | None,
        Parameter(
            name=["--until"],
            help="Match files overlapping the window ending at this instant.",
        ),
    ] = None,
    limit: Annotated[int, Parameter(name=["--limit"], help="Max rows to print.")] = 50,
    format: Annotated[
        OutputFormat,
        Parameter(name=["--format"], help="Output as Rich table, JSON, or paths-only."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
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

    # Files with bogus start times (uninitialised clock) would sort first if
    # we just diffed end-start. Use the chunk-based fallback when the
    # file-level start is sub-epoch.
    _safe_dur_sql = f"({_EFF_END_SQL} - {_EFF_START_SQL})"
    _safe_start_sql = (
        f"CASE WHEN {_EFF_START_SQL} >= {_SANE_EPOCH_NS} "
        f"THEN {_EFF_START_SQL} ELSE NULL END"
    )
    order_by = {
        "path": "cf.abs_path",
        "duration": f"{_safe_dur_sql} DESC",
        "messages": "messages DESC" if (topic is not None or schema is not None) else "c.message_count DESC",
        "size": "cf.size_bytes DESC",
        "start": f"{_safe_start_sql} DESC",
    }[sort_by]

    folder_clause = ""
    folder_params: tuple[str | int, ...] = ()
    if folder is not None:
        folder_where, folder_params = _path_prefix_where(folder)
        # _path_prefix_where returns ``WHERE ...`` — re-tag as inline conjunct.
        folder_clause = folder_where.removeprefix("WHERE ").replace(
            "abs_path", "cf.abs_path"
        )

    channel_filtered = topic is not None or schema is not None
    params: list[str | int] = []
    if channel_filtered:
        sql = (
            "SELECT cf.abs_path, "
            "COALESCE(SUM(cc.message_count), 0) AS messages, "
            "COUNT(DISTINCT cc.channel_id) AS channels, "
            f"{_EFF_START_SQL} AS eff_start, "
            f"{_EFF_END_SQL}   AS eff_end, "
            "cf.size_bytes "
            "FROM current_file cf "
            "JOIN content c           ON c.content_id        = cf.content_id "
            "JOIN content_channel cc  ON cc.content_id       = cf.content_id "
            "JOIN channel_sig sig     ON sig.channel_sig_id  = cc.channel_sig_id "
        )
        where: list[str] = []
        if topic is not None:
            sql += "JOIN topic t ON t.topic_id = sig.topic_id "
            where.append("t.name = ?")
            params.append(topic)
        if schema is not None:
            sql += "JOIN schema s ON s.schema_pk_id = sig.schema_pk_id "
            where.append("s.name = ?")
            params.append(schema)
        if fingerprint is not None:
            where.append("c.summary_fingerprint = ?")
            params.append(fingerprint)
        if window_end is not None:
            where.append("c.message_start_time <= ?")
            params.append(window_end)
        if window_start is not None:
            where.append("c.message_end_time >= ?")
            params.append(window_start)
        if folder_clause:
            where.append(folder_clause)
            params.extend(folder_params)
        sql += "WHERE " + " AND ".join(where) + " "
        sql += f"GROUP BY cf.abs_path ORDER BY {order_by} LIMIT ?"
    else:
        sql = (
            "SELECT cf.abs_path, c.message_count, c.channel_count, "
            f"{_EFF_START_SQL} AS eff_start, "
            f"{_EFF_END_SQL}   AS eff_end, "
            "cf.size_bytes "
            "FROM current_file cf "
            "JOIN content c ON c.content_id = cf.content_id "
        )
        where = []
        if fingerprint is not None:
            where.append("c.summary_fingerprint = ?")
            params.append(fingerprint)
        if window_end is not None:
            where.append("c.message_start_time <= ?")
            params.append(window_end)
        if window_start is not None:
            where.append("c.message_end_time >= ?")
            params.append(window_start)
        if folder_clause:
            where.append(folder_clause)
            params.extend(folder_params)
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += f"ORDER BY {order_by} LIMIT ?"
    params.append(limit)

    with open_db(db_path, read_only=True) as conn:
        sql_rows = conn.execute(sql, params).fetchall()

    rows: list[dict[str, object]] = [
        {
            "path": path,
            "messages": msgs,
            "channels": channels,
            "start_time_ns": start,
            "end_time_ns": end,
            "duration_ns": _safe_duration_ns(start, end),
            "size_bytes": size,
        }
        for path, msgs, channels, start, end, size in sql_rows
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
        duration_str = (
            _format_duration_ns(0, duration_ns)
            if isinstance(duration_ns, int)
            else "-"
        )
        table.add_row(
            _format_parts_with_colors(str(row["path"])),
            f"{msgs:,}" if isinstance(msgs, int) else "-",
            f"{channels:,}" if isinstance(channels, int) else "-",
            _format_ts_ns(start if isinstance(start, int) else None),
            _format_ts_ns(end if isinstance(end, int) else None),
            duration_str,
        )
    console.print(table)
    return 0


@index_app.command(name="topics")
def topics_cmd(
    prefix: Annotated[
        str | None,
        Parameter(help="Optional topic prefix filter (e.g. '/tf' or '/sensor')."),
    ] = None,
    *,
    sort_by: Annotated[
        Literal["files", "messages", "name"],
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

    # Always tie-break by the other counts and then the topic name for
    # deterministic output.
    order_by = {
        "files":    "files DESC, messages DESC, t.name",
        "messages": "messages DESC, files DESC, t.name",
        "name":     "t.name",
    }[sort_by]

    sql = (
        "SELECT t.name AS topic, "
        "       COUNT(DISTINCT cf.abs_path) AS files, "
        "       COALESCE(SUM(cc.message_count), 0) AS messages "
        "FROM current_file cf "
        "JOIN content_channel cc ON cc.content_id      = cf.content_id "
        "JOIN channel_sig sig    ON sig.channel_sig_id = cc.channel_sig_id "
        "JOIN topic t            ON t.topic_id         = sig.topic_id "
    )
    params: list[str | int] = []
    if prefix is not None:
        sql += "WHERE t.name LIKE ? ESCAPE '\\' "
        params.append(_like_prefix_param(prefix))
    sql += f"GROUP BY t.name HAVING files >= ? ORDER BY {order_by} LIMIT ?"
    params.extend([min_files, limit])

    with open_db(db_path, read_only=True) as conn:
        sql_rows = conn.execute(sql, params).fetchall()

    rows: list[dict[str, object]] = [
        {"topic": topic, "files": files, "messages": messages}
        for topic, files, messages in sql_rows
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
    for row in rows:
        files = row["files"]
        messages = row["messages"]
        table.add_row(
            _format_parts_with_colors(str(row["topic"])),
            f"{files:,}" if isinstance(files, int) else "-",
            _format_count(int(messages)) if isinstance(messages, int) else "-",
        )
    console.print(table)
    return 0


@index_app.command(name="schemas")
def schemas_cmd(
    prefix: Annotated[
        str | None,
        Parameter(help="Optional schema-name prefix (e.g. 'sensor_msgs')."),
    ] = None,
    *,
    sort_by: Annotated[
        Literal["files", "name", "encoding"],
        Parameter(
            name=["--sort-by"],
            help="Sort results (descending only for ``files``).",
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
        "files":    "files DESC, s.name",
        "name":     "s.name",
        "encoding": "s.encoding, s.name",
    }[sort_by]

    sql = (
        "SELECT s.name, s.encoding, "
        "       COUNT(DISTINCT cf.abs_path) AS files "
        "FROM current_file cf "
        "JOIN content_schema cs ON cs.content_id  = cf.content_id "
        "JOIN schema s          ON s.schema_pk_id = cs.schema_pk_id "
    )
    params: list[str | int] = []
    if prefix is not None:
        sql += "WHERE s.name LIKE ? ESCAPE '\\' "
        params.append(_like_prefix_param(prefix))
    sql += f"GROUP BY s.name, s.encoding HAVING files >= ? ORDER BY {order_by} LIMIT ?"
    params.extend([min_files, limit])

    with open_db(db_path, read_only=True) as conn:
        sql_rows = conn.execute(sql, params).fetchall()

    rows: list[dict[str, object]] = [
        {"name": name, "encoding": encoding, "files": files} for name, encoding, files in sql_rows
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
    for row in rows:
        files = row["files"]
        name = row["name"]
        table.add_row(
            _format_schema_with_link(str(name)) if name else "-",
            str(row.get("encoding") or "-"),
            f"{files:,}" if isinstance(files, int) else "-",
        )
    console.print(table)
    return 0


@index_app.command(name="duplicates")
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

    with open_db(db_path, read_only=True) as conn:
        sql_rows = conn.execute(sql, params).fetchall()

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


@index_app.command(name="info")
def info_cmd(
    target: str,
    *,
    format: Annotated[
        Literal["table", "json"],
        Parameter(name=["--format"], help="Output as Rich tables or JSON."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Show everything the index knows about TARGET (path or summary fingerprint)."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    abs_path: str | None = None
    summary_fp: str | None = None
    candidate = Path(target).expanduser()
    if candidate.exists() or os.sep in target:
        abs_path = str(candidate.resolve())
    else:
        summary_fp = target

    with open_db(db_path, read_only=True) as conn:
        if abs_path is not None:
            row = conn.execute(
                "SELECT summary_fingerprint FROM current_file WHERE abs_path = ?",
                (abs_path,),
            ).fetchone()
            if row is None or row[0] is None:
                console.print(f"[red]Error:[/] no indexed content for {abs_path}")
                return 1
            summary_fp = row[0]
        else:
            row = conn.execute(
                "SELECT 1 FROM content WHERE summary_fingerprint = ?", (summary_fp,)
            ).fetchone()
            if row is None:
                console.print(f"[red]Error:[/] unknown summary fingerprint {summary_fp}")
                return 1

        content = conn.execute(
            "SELECT size_bytes, library, profile, message_count, schema_count, channel_count, "
            "       attachment_count, metadata_count, chunk_count, "
            "       message_start_time, message_end_time, first_seen_at "
            "FROM content WHERE summary_fingerprint = ?",
            (summary_fp,),
        ).fetchone()
        topic_rows = conn.execute(
            "SELECT t.name AS topic, s.name AS schema_name, "
            "       sig.message_encoding, cc.message_count "
            "FROM content_channel cc "
            "JOIN content c       ON c.content_id        = cc.content_id "
            "JOIN channel_sig sig ON sig.channel_sig_id  = cc.channel_sig_id "
            "JOIN topic t         ON t.topic_id          = sig.topic_id "
            "LEFT JOIN schema s   ON s.schema_pk_id      = sig.schema_pk_id "
            "WHERE c.summary_fingerprint = ? "
            "ORDER BY cc.message_count DESC NULLS LAST, t.name",
            (summary_fp,),
        ).fetchall()
        observation_rows = conn.execute(
            "SELECT fo.abs_path, fo.observed_at, fo.session_id, "
            "       fo.file_fingerprint, c.summary_fingerprint "
            "FROM file_observation fo "
            "LEFT JOIN content c ON c.content_id = fo.content_id "
            "WHERE c.summary_fingerprint = ? OR fo.abs_path = ? "
            "ORDER BY fo.observed_at DESC LIMIT 20",
            (summary_fp, abs_path or ""),
        ).fetchall()
        error_rows = conn.execute(
            "SELECT abs_path, observed_at, error_kind, error_message "
            "FROM scan_error WHERE abs_path = ? ORDER BY observed_at DESC LIMIT 10",
            (abs_path or "",),
        ).fetchall()

    (
        size_bytes,
        library,
        profile,
        message_count,
        schema_count,
        channel_count,
        attachment_count,
        metadata_count,
        chunk_count,
        start_ns,
        end_ns,
        first_seen_at,
    ) = content

    identity = {
        "summary_fingerprint": summary_fp,
        "path": abs_path,
        "size_bytes": size_bytes,
        "library": library,
        "profile": profile,
        "message_count": message_count,
        "schema_count": schema_count,
        "channel_count": channel_count,
        "attachment_count": attachment_count,
        "metadata_count": metadata_count,
        "chunk_count": chunk_count,
        "message_start_time_ns": start_ns,
        "message_end_time_ns": end_ns,
        "duration_ns": (end_ns - start_ns) if start_ns and end_ns and end_ns > start_ns else None,
        "first_seen_at_ns": first_seen_at,
    }
    topics_payload = [
        {
            "topic": topic,
            "schema": schema_name,
            "encoding": encoding,
            "message_count": msg_count,
        }
        for topic, schema_name, encoding, msg_count in topic_rows
    ]
    observations_payload = [
        {
            "path": obs_path,
            "observed_at_ns": observed_at,
            "session_id": session_id,
            "file_fingerprint": file_fp,
            "summary_fingerprint": obs_summary_fp,
        }
        for obs_path, observed_at, session_id, file_fp, obs_summary_fp in observation_rows
    ]
    errors_payload = [
        {
            "path": err_path,
            "observed_at_ns": observed_at,
            "kind": kind,
            "message": message,
        }
        for err_path, observed_at, kind, message in error_rows
    ]

    if format == "json":
        _stdout_line(
            _json.dumps(
                {
                    "identity": identity,
                    "topics": topics_payload,
                    "observations": observations_payload,
                    "errors": errors_payload,
                },
                default=str,
            )
        )
        return 0

    identity_table = Table.grid(padding=(0, 1))
    identity_table.add_column(style="bold blue")
    identity_table.add_column()
    identity_table.add_row("Summary fingerprint:", f"[dim]{summary_fp}[/]")
    if abs_path is not None:
        identity_table.add_row("Path:", _format_parts_with_colors(abs_path))
    if size_bytes is not None:
        identity_table.add_row(
            "Size:",
            f"[green]{bytes_to_human(size_bytes)}[/] [dim]({size_bytes:,} B)[/]",
        )
    else:
        identity_table.add_row("Size:", "-")
    identity_table.add_row("Library:", f"[yellow]{library or '-'}[/]")
    identity_table.add_row("Profile:", f"[yellow]{profile or '-'}[/]")
    if isinstance(message_count, int):
        identity_table.add_row("Messages:", f"[green]{_format_count(message_count)}[/]")
    else:
        identity_table.add_row("Messages:", "-")
    identity_table.add_row(
        "Schemas / Channels / Chunks:",
        f"[green]{schema_count or 0:,}[/] / "
        f"[green]{channel_count or 0:,}[/] / "
        f"[cyan]{chunk_count or 0:,}[/]",
    )
    identity_table.add_row(
        "Attachments / Metadata:",
        f"[yellow]{attachment_count or 0:,}[/] / [cyan]{metadata_count or 0:,}[/]",
    )
    identity_table.add_row("Start:", _format_ts_ns(start_ns))
    identity_table.add_row("End:", _format_ts_ns(end_ns))
    identity_table.add_row("Duration:", f"[cyan]{_format_duration_ns(start_ns, end_ns)}[/]")
    console.print("[bold cyan]Identity[/]")
    console.print(identity_table)

    if topics_payload:
        topic_table = Table(title=f"Topics ({len(topics_payload):,})")
        topic_table.add_column("Topic", overflow="fold")
        topic_table.add_column("Schema", overflow="fold")
        topic_table.add_column("Encoding", style="yellow")
        topic_table.add_column("Messages", justify="right", style="green")
        for entry in topics_payload:
            msgs = entry["message_count"]
            schema_name = entry["schema"]
            topic_table.add_row(
                _format_parts_with_colors(str(entry["topic"])),
                _format_schema_with_link(str(schema_name)) if schema_name else "-",
                str(entry["encoding"] or "-"),
                _format_count(int(msgs)) if isinstance(msgs, int) else "-",
            )
        console.print(topic_table)

    if observations_payload:
        obs_table = Table(title=f"Observations ({len(observations_payload):,})")
        obs_table.add_column("Path", overflow="fold")
        obs_table.add_column("Observed at (UTC)", style="cyan")
        obs_table.add_column("Session", justify="right", style="green")
        obs_table.add_column("File fp", style="dim")
        obs_table.add_column("Summary fp", style="dim")
        for entry in observations_payload:
            observed_at = entry["observed_at_ns"]
            obs_table.add_row(
                _format_parts_with_colors(str(entry["path"])),
                _format_ts_ns(observed_at if isinstance(observed_at, int) else None),
                str(entry["session_id"]),
                str(entry["file_fingerprint"] or "-"),
                str(entry["summary_fingerprint"] or "-"),
            )
        console.print(obs_table)

    if errors_payload:
        err_table = Table(title=f"Errors ({len(errors_payload):,})")
        err_table.add_column("Path", overflow="fold")
        err_table.add_column("Observed at (UTC)", style="cyan")
        err_table.add_column("Kind", style="red")
        err_table.add_column("Message", overflow="fold", style="dim")
        for entry in errors_payload:
            observed_at = entry["observed_at_ns"]
            err_table.add_row(
                _format_parts_with_colors(str(entry["path"])),
                _format_ts_ns(observed_at if isinstance(observed_at, int) else None),
                _describe_error_kind(str(entry["kind"])),
                str(entry["message"] or "-"),
            )
        console.print(err_table)

    return 0
