"""In-place recovery command that rebuilds summary/footer."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Confirm
from small_mcap import MAGIC, get_summary
from small_mcap.exceptions import McapError
from small_mcap.records import Footer, Opcode, Summary, SummaryOffset
from small_mcap.writer import (
    _calculate_summary_crc,
    _calculate_summary_offset_start,
    _write_summary_section,
)

from pymcap_cli.utils import rebuild_info

console = Console()


def _summaries_equal(a: Summary, b: Summary) -> bool:
    """Deep-ish comparison of summary contents."""
    return (
        a.statistics == b.statistics
        and a.schemas == b.schemas
        and a.channels == b.channels
        and a.chunk_indexes == b.chunk_indexes
        and a.attachment_indexes == b.attachment_indexes
        and a.metadata_indexes == b.metadata_indexes
    )


def _describe_summary_diff(existing: Summary, rebuilt: Summary) -> list[str]:
    """Generate human-readable differences between two summaries."""
    diffs: list[str] = []

    def compare_dict(name: str, before: dict[Any, Any], after: dict[Any, Any]) -> None:
        if before == after:
            return
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        changed = sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
        parts = []
        if missing:
            parts.append(f"removed ids {missing}")
        if added:
            parts.append(f"added ids {added}")
        if changed:
            parts.append(f"changed ids {changed}")
        if not parts:
            parts.append("contents differ")
        diffs.append(f"{name}: " + "; ".join(parts))

    def compare_list(name: str, before: list[Any], after: list[Any]) -> None:
        if before == after:
            return
        if len(before) != len(after):
            diffs.append(f"{name}: count {len(before)} -> {len(after)}")
            return
        for idx, (b_item, a_item) in enumerate(zip(before, after, strict=True)):
            if b_item != a_item:
                diffs.append(f"{name}: differs at index {idx} {b_item} -> {a_item}")
                break

    if existing.statistics != rebuilt.statistics:
        diffs.append(f"statistics: {existing.statistics} -> {rebuilt.statistics}")

    compare_dict("schemas", existing.schemas, rebuilt.schemas)
    compare_dict("channels", existing.channels, rebuilt.channels)
    compare_list("chunk_indexes", existing.chunk_indexes, rebuilt.chunk_indexes)
    compare_list("attachment_indexes", existing.attachment_indexes, rebuilt.attachment_indexes)
    compare_list("metadata_indexes", existing.metadata_indexes, rebuilt.metadata_indexes)

    return diffs


def _build_summary_bytes(summary: Summary, summary_start: int) -> tuple[bytes, list[SummaryOffset]]:
    """Build summary bytes and offsets from an existing Summary."""
    buffer = io.BytesIO()
    offsets: list[SummaryOffset] = []

    sections = [
        (Opcode.SCHEMA, summary.schemas.values()),
        (Opcode.CHANNEL, summary.channels.values()),
        (Opcode.STATISTICS, [summary.statistics] if summary.statistics else []),
        (Opcode.CHUNK_INDEX, summary.chunk_indexes),
        (Opcode.ATTACHMENT_INDEX, summary.attachment_indexes),
        (Opcode.METADATA_INDEX, summary.metadata_indexes),
    ]

    for opcode, items in sections:
        _write_summary_section(buffer, offsets, opcode, items, summary_start)

    for offset in offsets:
        offset.write_record_to(buffer)

    return buffer.getvalue(), offsets


def recover_inplace(file: str, *, exact_sizes: bool = False, force: bool = False) -> int:
    """Rebuild an MCAP's summary/footer in place using recovered info."""
    path = Path(file)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        return 1

    file_size = path.stat().st_size

    with path.open("rb") as read_stream:
        has_valid_summary = False
        existing_summary = None
        try:
            existing_summary = get_summary(read_stream)
            has_valid_summary = True
        except McapError:
            has_valid_summary = False
        read_stream.seek(0)

        info = rebuild_info(read_stream, file_size, exact_sizes=exact_sizes, console=console)

    if has_valid_summary and existing_summary is not None:
        rebuilt = info.summary
        if _summaries_equal(existing_summary, rebuilt):
            console.print("[yellow]File already appears to have a valid summary/footer[/yellow]")
        else:
            console.print(
                "[yellow]Existing summary differs from rebuilt version (will overwrite):[/yellow]"
            )
            for line in _describe_summary_diff(existing_summary, rebuilt):
                console.print(f"  {line}")

    if not force and not Confirm.ask(
        f"[yellow]This will overwrite[/yellow] {path} [yellow]in place. Continue?[/yellow]",
        default=False,
    ):
        console.print("[yellow]Aborted[/yellow]")
        return 1

    summary_start = info.next_offset
    summary_data, summary_offsets = _build_summary_bytes(info.summary, summary_start)

    summary_offset_start = _calculate_summary_offset_start(
        summary_start, summary_data, summary_offsets, use_summary_offsets=True
    )
    summary_crc = _calculate_summary_crc(
        summary_data,
        summary_start,
        summary_offsets,
        use_summary_offsets=True,
        enable_crcs=True,
    )

    with path.open("r+b") as f:
        # Truncate any existing summary/footer and write the recovered one
        f.seek(summary_start)
        f.truncate()
        f.write(summary_data)

        footer = Footer(
            summary_start=summary_start if summary_data else 0,
            summary_offset_start=summary_offset_start,
            summary_crc=summary_crc,
        )
        footer.write_record_to(f)
        f.write(MAGIC)

    console.print("[green]✓ In-place recovery completed[/green]")
    return 0
