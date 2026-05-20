"""``pymcap-cli index tree`` — directory-tree breakdown of indexed data."""

import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from rich.text import Text
from rich.tree import Tree

from pymcap_cli.cmd.index._helpers import (
    _format_count,
    _path_prefix_predicate,
    _print_db_needs_migration,
    _resolve_db,
    _safe_duration_ns,
    console,
)
from pymcap_cli.index import SANE_EPOCH_NS
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db
from pymcap_cli.utils import bytes_to_human

_TreeRow = tuple[
    str,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
]


@dataclass(slots=True)
class _PathNode:
    """Aggregate stats for one directory in the path tree.

    ``channel_mask`` is a Python int whose set bits identify the distinct
    ``channel_signature.id`` values appearing in any file under the node;
    ORing it as we walk up the path chain unions the sets, so
    ``channel_mask.bit_count()`` is the distinct-channel count. The
    distinct-schema count is derived lazily at render time by mapping each
    set channel bit through :data:`_csig_to_schema_bit`; we don't carry a
    second per-node schema mask because deriving it for the handful of
    rendered nodes is ~10x cheaper than fetching ``content_schema`` rows
    for every content at query time.
    """

    file_count: int = 0
    size_bytes: int = 0
    message_count: int = 0
    duration_ns: int = 0
    channel_mask: int = 0
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


_ARROW = Text("  →  ", style="dim")
_GAP = Text("  ")


def _node_stats_text(node: _PathNode, csig_to_schema_bit: Sequence[int]) -> Text:
    """Pre-styled per-node stats line, bypassing Rich's markup parser.

    Markup tag parsing dominates the render path on large catalogues; building
    a :class:`rich.text.Text` directly with style spans avoids that work and
    cuts ``console.print`` time roughly in half.
    """
    text = Text()
    text.append(bytes_to_human(node.size_bytes), style="yellow")
    text.append(_GAP)
    text.append(f"{_format_count(node.file_count)}f", style="green")
    text.append(_GAP)
    text.append(f"{_format_count(node.message_count)}msg", style="cyan")
    text.append(_GAP)
    text.append(f"{_format_count(node.channel_mask.bit_count())}ch", style="bright_cyan")
    text.append(_GAP)
    text.append(
        f"{_format_count(_node_schema_count(node, csig_to_schema_bit))}sch",
        style="blue",
    )
    if node.duration_ns:
        text.append(_GAP)
        text.append(_format_seconds_short(node.duration_ns / 1e9), style="magenta")
    return text


def _node_label(name: str, node: _PathNode, csig_to_schema_bit: Sequence[int]) -> Text:
    label = Text(f"{name}/", style="bold")
    label.append_text(_ARROW)
    label.append_text(_node_stats_text(node, csig_to_schema_bit))
    return label


def _root_label(name: str, node: _PathNode, csig_to_schema_bit: Sequence[int]) -> Text:
    label = Text(name, style="bold blue")
    label.append_text(_ARROW)
    label.append_text(_node_stats_text(node, csig_to_schema_bit))
    return label


def _node_schema_count(node: _PathNode, csig_to_schema_bit: Sequence[int]) -> int:
    """Distinct schemas referenced by any channel under ``node``.

    Walks the set bits of ``node.channel_mask`` and ORs in each channel's
    ``1 << schema_id`` (zero for channels without a schema). Limited to
    rendered nodes — ~thousands max — so the per-node cost (~hundreds of µs
    for fully-populated catalogues) dominates over the alternative of
    aggregating ``content_schema`` for every content in SQL.
    """
    mask = node.channel_mask
    schema_mask = 0
    limit = len(csig_to_schema_bit)
    while mask:
        lsb = (mask & -mask).bit_length() - 1
        if lsb < limit:
            schema_mask |= csig_to_schema_bit[lsb]
        mask &= mask - 1
    return schema_mask.bit_count()


class _BitMask:
    """SQLite aggregate that ORs ``1 << value`` into a Python int per group.

    Finalize returns the result as little-endian ``bytes`` so the C boundary
    only carries a single BLOB per group; the caller converts back to ``int``
    via :func:`int.from_bytes`. Going through bytes is measurably faster than
    GROUP_CONCAT + CSV parse on million-row source tables because the
    fan-out happens in C and the Python decode step is a single call per row.
    """

    __slots__ = ("mask",)

    def __init__(self) -> None:
        self.mask = 0

    def step(self, value: int | None) -> None:
        if value is not None:
            self.mask |= 1 << value

    def finalize(self) -> bytes:
        return self.mask.to_bytes((self.mask.bit_length() + 7) // 8, "little")


def _load_channel_masks(conn: sqlite3.Connection) -> dict[int, int]:
    """Return ``content_id`` → channel-signature bitmask.

    Each value is a Python ``int`` whose set bits enumerate the distinct
    ``channel_signature_id`` referenced by that content. ORing two masks
    unions the underlying sets; ``mask.bit_count()`` is the distinct-channel
    count for the union. The custom aggregate keeps row counts close to the
    number of contents (~tens of thousands) instead of the raw join size
    (~millions), which is what keeps the whole command under the 1s budget
    on large catalogues.

    Always scans ``content_channel`` in full — even with a folder filter — to
    avoid re-evaluating the ``current_file`` view inside the aggregation.
    On large catalogues the view's correlated subquery is ~2.5x slower than
    the unfiltered scan when the folder matches most files; extra mask entries
    for unmatched contents cost a few MB of dict memory but are never looked
    up because :func:`_build_path_tree` only references the content_ids that
    appear in the main row set.
    """
    conn.create_aggregate("_bit_mask", 1, _BitMask)  # ty: ignore[invalid-argument-type]
    return {
        content_id: int.from_bytes(buf, "little")
        for content_id, buf in conn.execute(
            "SELECT content_id, _bit_mask(channel_signature_id) "
            "FROM content_channel GROUP BY content_id"
        )
    }


def _load_csig_to_schema_bit(conn: sqlite3.Connection) -> list[int]:
    """Return a list indexed by ``channel_signature.id`` of ``1 << schema_id``.

    Zero means the channel has no schema. We index by signature id so the
    bit-walking loop in :func:`_node_schema_count` can do an ``O(1)`` lookup
    without a dict. The signature table has only a few thousand rows, so the
    resulting list is small and the query is essentially free.

    Distinct schemas are approximated through this channel→schema map rather
    than queried from ``content_schema`` because deriving them at render time
    only for the handful of rendered nodes is dramatically cheaper than
    aggregating per-content schema masks for every catalogue entry. The
    approximation misses orphan schemas referenced by zero channels
    (typically <0.1% of rows); for a tree summary that's a worthwhile trade.
    """
    rows = conn.execute(
        "SELECT id, schema_id FROM channel_signature WHERE schema_id IS NOT NULL"
    ).fetchall()
    max_id = max((row[0] for row in rows), default=-1)
    lookup = [0] * (max_id + 1)
    for csig_id, schema_id in rows:
        lookup[csig_id] = 1 << schema_id
    return lookup


def _build_path_tree(
    rows: Sequence[_TreeRow],
    root_prefix: str,
    *,
    channel_masks: dict[int, int] | None = None,
) -> _PathNode:
    """Group rows by their path prefix and accumulate stats up the chain.

    Each ``_TreeRow`` is ``(abs_path, size_bytes, message_count, content_id,
    eff_start_ns, eff_end_ns)``. ``channel_masks`` maps a ``content_id`` to a
    Python int whose set bits enumerate distinct channel signatures referenced
    by that content; the bitmasks OR together along the directory chain so
    ``bit_count()`` yields per-node distinct counts. Schema counts are derived
    on demand from the channel mask at render time.
    """
    root = _PathNode()
    chain_cache: dict[str, list[_PathNode]] = {}
    root_prefix_with_sep = f"{root_prefix.rstrip(os.sep)}{os.sep}" if root_prefix else ""
    channel_masks = channel_masks or {}

    def _chain_for(abs_path: str) -> list[_PathNode]:
        cached = chain_cache.get(abs_path)
        if cached is not None:
            return cached
        if root_prefix_with_sep and abs_path.startswith(root_prefix_with_sep):
            rel = abs_path[len(root_prefix_with_sep) :]
        else:
            try:
                rel = os.path.relpath(abs_path, root_prefix) if root_prefix else abs_path
            except ValueError:
                rel = abs_path
        parts = [p for p in rel.split(os.sep) if p not in ("", ".")]  # noqa: PTH206
        chain = [root]
        node = root
        for part in parts[:-1]:
            existing = node.children.get(part)
            if existing is None:
                existing = _PathNode()
                node.children[part] = existing
            node = existing
            chain.append(node)
        chain_cache[abs_path] = chain
        return chain

    for abs_path, size, msg_count, content_id, ts_start, ts_end in rows:
        chain = _chain_for(abs_path)
        size_v = size or 0
        msg_v = msg_count or 0
        dur_v = _safe_duration_ns(ts_start, ts_end) or 0
        ch_mask = channel_masks.get(content_id, 0) if content_id is not None else 0
        for node in chain:
            node.file_count += 1
            node.size_bytes += size_v
            node.message_count += msg_v
            node.duration_ns += dur_v
            node.channel_mask |= ch_mask

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
    max_children: int,
    min_files: int,
    sort_by: str,
    csig_to_schema_bit: Sequence[int],
) -> Tree:
    sort_key = _TREE_SORT_KEYS[sort_by]
    folded_root_label, folded_root = _fold_single_child_chain(root_label, root)
    tree = Tree(_root_label(folded_root_label, folded_root, csig_to_schema_bit))

    def _add(parent: Tree, node: _PathNode, depth: int) -> None:
        if depth >= max_depth:
            if node.children:
                parent.add(Text(f"… {len(node.children):,} subdirs collapsed", style="dim"))
            return
        children = [
            (name, child)
            for name, child in sorted(node.children.items(), key=sort_key)
            if child.file_count >= min_files
        ]
        if max_children > 0:
            visible_children = children[:max_children]
            hidden_count = len(children) - len(visible_children)
        else:
            visible_children = children
            hidden_count = 0
        for name, child in visible_children:
            display_name, display_child = _fold_single_child_chain(name, child)
            sub = parent.add(_node_label(display_name, display_child, csig_to_schema_bit))
            _add(sub, display_child, depth + 1)
        if hidden_count:
            parent.add(Text(f"… {hidden_count:,} more subdirs hidden", style="dim"))

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
    max_children: Annotated[
        int,
        Parameter(
            name=["--max-children"],
            help="Maximum sibling directories to render per node. Use 0 for unlimited.",
        ),
    ] = 50,
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

    path_clause = ""
    path_params: tuple[str, ...] = ()
    if folder is not None:
        predicate, path_params = _path_prefix_predicate(folder)
        path_clause = f"WHERE {predicate.replace('abs_path', 'cf.abs_path')}"

    time_filter_params = (SANE_EPOCH_NS, SANE_EPOCH_NS)

    try:
        with open_db(db_path, read_only=True) as conn:
            rows = conn.execute(
                f"""SELECT cf.abs_path, cf.size_bytes, c.message_count, cf.content_id,
                           CASE WHEN c.message_start_time_ns >= ?
                                THEN c.message_start_time_ns
                                ELSE c.sane_message_start_time_ns END AS eff_start,
                           CASE WHEN c.message_start_time_ns >= ?
                                THEN c.message_end_time_ns
                                ELSE c.sane_message_end_time_ns END AS eff_end
                      FROM current_file cf
                      JOIN content c ON c.id = cf.content_id
                      {path_clause}""",  # noqa: S608
                (*time_filter_params, *path_params),
            ).fetchall()
            channel_masks = _load_channel_masks(conn)
            csig_to_schema_bit = _load_csig_to_schema_bit(conn)
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    if not rows:
        console.print("[dim]No indexed files match.[/]")
        return 0

    root_prefix = str(folder.expanduser().resolve()) if folder is not None else ""

    root_node = _build_path_tree(
        rows,
        root_prefix,
        channel_masks=channel_masks,
    )
    rendered = _render_path_tree(
        root_node,
        root_label=root_prefix or "(all)",
        max_depth=max_depth,
        max_children=max_children,
        min_files=min_files,
        sort_by=sort_by,
        csig_to_schema_bit=csig_to_schema_bit,
    )
    console.print(rendered, soft_wrap=True)
    return 0
