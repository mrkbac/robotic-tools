"""``pymcap-cli index tree`` — directory-tree breakdown of indexed data."""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from rich.tree import Tree

from pymcap_cli.cmd.index._helpers import (
    _format_count,
    _optional_path_filter_params,
    _print_db_needs_migration,
    _resolve_db,
    _safe_duration_ns,
    console,
)
from pymcap_cli.index import SANE_EPOCH_NS
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db
from pymcap_cli.utils import bytes_to_human


@dataclass
class _PathNode:
    """Aggregate stats for one directory in the path tree."""

    file_count: int = 0
    size_bytes: int = 0
    message_count: int = 0
    duration_ns: int = 0
    topics: set[str] = field(default_factory=set)
    schemas: set[str] = field(default_factory=set)
    children: "dict[str, _PathNode]" = field(default_factory=dict)


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
    topics: Sequence[tuple[str, str | None]],
    schemas: Sequence[tuple[str, str | None]],
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
        parts = [p for p in Path(rel).parts if p not in ("", os.sep, ".")]
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

    for abs_path, joined in topics:
        if not joined:
            continue
        members = joined.split(",")
        for node in _chain_for(abs_path):
            node.topics.update(members)

    for abs_path, joined in schemas:
        if not joined:
            continue
        members = joined.split(",")
        for node in _chain_for(abs_path):
            node.schemas.update(members)

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
    tree = Tree(f"[bold blue]{folded_root_label}[/]  [dim]→[/]  {_format_node_stats(folded_root)}")

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
            help=(
                "Limit how many directory levels to render. "
                "Aggregates still cover everything below."
            ),
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
    """Show a directory-tree breakdown of indexed data."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    path_filter_params = _optional_path_filter_params(folder)
    time_filter_params = (SANE_EPOCH_NS, SANE_EPOCH_NS)

    try:
        with open_db(db_path, read_only=True) as conn:
            files = conn.execute(
                """SELECT cf.abs_path, cf.size_bytes, c.message_count,
                          CASE WHEN c.message_start_time_ns >= ?
                               THEN c.message_start_time_ns ELSE c.sane_message_start_time_ns END
                               AS eff_start,
                          CASE WHEN c.message_start_time_ns >= ?
                               THEN c.message_end_time_ns ELSE c.sane_message_end_time_ns END
                               AS eff_end
                    FROM current_file cf
                    JOIN content c ON c.id = cf.content_id
                    WHERE (? IS NULL OR cf.abs_path = ?
                           OR substr(cf.abs_path, 1, ?) = ?)""",
                (*time_filter_params, *path_filter_params),
            ).fetchall()
            topics = conn.execute(
                """SELECT cf.abs_path, agg.names
                    FROM current_file cf
                    JOIN (
                        SELECT cc.content_id, GROUP_CONCAT(DISTINCT t.name) AS names
                        FROM content_channel cc
                        JOIN channel_signature sig ON sig.id = cc.channel_signature_id
                        JOIN topic t         ON t.id          = sig.topic_id
                        GROUP BY cc.content_id
                    ) agg ON agg.content_id = cf.content_id
                    WHERE (? IS NULL OR cf.abs_path = ?
                           OR substr(cf.abs_path, 1, ?) = ?)""",
                path_filter_params,
            ).fetchall()
            schema_rows = conn.execute(
                """SELECT cf.abs_path, agg.hashes
                    FROM current_file cf
                    JOIN (
                        SELECT cs.content_id, GROUP_CONCAT(DISTINCT s.schema_hash) AS hashes
                        FROM content_schema cs
                        JOIN schema s ON s.id = cs.schema_id
                        GROUP BY cs.content_id
                    ) agg ON agg.content_id = cf.content_id
                    WHERE (? IS NULL OR cf.abs_path = ?
                           OR substr(cf.abs_path, 1, ?) = ?)""",
                path_filter_params,
            ).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

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
