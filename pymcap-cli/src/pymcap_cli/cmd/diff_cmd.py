"""Diff command - compare MCAP files using message indexes."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, TypeVar

from cyclopts import Parameter
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree
from small_mcap import RebuildInfo, Schema, Statistics, Summary

from pymcap_cli.core.mcap_compare import (
    IndexedCompareKind,
    IndexedComparison,
    MessageIndexIdentityReadResult,
    TimestampRangeSummary,
    collect_message_timestamps,
    compare_indexed_identities,
    message_index_identity_from_info,
    read_compare_file,
    split_timestamps_into_segments,
)
from pymcap_cli.rihs01 import compute_rihs01
from pymcap_cli.utils import NS_TO_MS, NS_TO_SEC, bytes_to_human, format_ts_short

logger = logging.getLogger(__name__)
console = Console()


def time_str(time_ns: int) -> str:
    if time_ns == 0:
        return "N/A"
    dt = datetime.fromtimestamp(time_ns / NS_TO_SEC)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def duration_human(duration_ns: int) -> str:
    duration = timedelta(milliseconds=duration_ns / NS_TO_MS)
    total_seconds = duration.total_seconds()
    if total_seconds < 1:
        return f"{total_seconds * 1000:.0f} ms"
    if total_seconds < 60:
        return f"{total_seconds:.0f} s"
    if total_seconds < 3600:
        return f"{total_seconds / 60:.0f} min"
    return f"{total_seconds / 3600:.1f} hr"


def _file_label(path: str) -> str:
    return Path(path).name


@dataclass(frozen=True, slots=True)
class FileSummary:
    label: str
    path: str
    size_bytes: int
    summary: Summary
    statistics: Statistics

    @property
    def duration_ns(self) -> int:
        return self.statistics.message_end_time - self.statistics.message_start_time


@dataclass(slots=True)
class ChannelDiff:
    topic: str
    counts: dict[str, int] = field(default_factory=dict)
    timestamps: dict[str, set[int]] = field(default_factory=dict)
    common_count: int = 0

    @property
    def is_identical(self) -> bool:
        return all(len(ts) == 0 for ts in self.timestamps.values())

    def unique_in(self, label: str) -> int:
        return len(self.timestamps.get(label, set()))


@dataclass(frozen=True, slots=True)
class LabeledIndexedComparison:
    left_label: str
    right_label: str
    comparison: IndexedComparison


def _schema_fingerprint(schema: Schema) -> str:
    if schema.encoding == "ros2msg":
        try:
            return compute_rihs01(schema.name, schema.data)
        except Exception:  # noqa: BLE001, S110
            pass
    return hashlib.sha256(schema.data).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class SchemaDiff:
    name: str
    per_label: dict[str, str | None] = field(default_factory=dict)
    encodings: dict[str, str | None] = field(default_factory=dict)

    @property
    def is_identical(self) -> bool:
        present = {v for v in self.per_label.values() if v is not None}
        return len(present) <= 1


@dataclass(frozen=True, slots=True)
class ChannelSchemaMismatch:
    topic: str
    schema_names: dict[str, str | None] = field(default_factory=dict)
    schema_encodings: dict[str, str | None] = field(default_factory=dict)
    message_encodings: dict[str, str | None] = field(default_factory=dict)


def _compare_schemas(
    all_summaries: dict[str, FileSummary],
) -> dict[str, SchemaDiff]:
    all_names: set[str] = set()
    schemas_by_label: dict[str, dict[str, Schema]] = {}
    for label, fs in all_summaries.items():
        by_name = {s.name: s for s in fs.summary.schemas.values()}
        all_names |= by_name.keys()
        schemas_by_label[label] = by_name

    diffs: dict[str, SchemaDiff] = {}
    for name in sorted(all_names):
        per_label: dict[str, str | None] = {}
        encodings: dict[str, str | None] = {}
        for label in all_summaries:
            schema = schemas_by_label[label].get(name)
            if schema is not None:
                per_label[label] = _schema_fingerprint(schema)
                encodings[label] = schema.encoding
            else:
                per_label[label] = None
                encodings[label] = None
        diffs[name] = SchemaDiff(name=name, per_label=per_label, encodings=encodings)
    return diffs


def _check_channel_schema_mismatches(
    all_summaries: dict[str, FileSummary],
) -> tuple[list[ChannelSchemaMismatch], list[str]]:
    topic_info: dict[str, dict[str, tuple[str, str, str, str] | None]] = {}
    warnings: list[str] = []

    for label, fs in all_summaries.items():
        seen_topics: dict[str, int] = {}
        for channel in fs.summary.channels.values():
            count = seen_topics.get(channel.topic, 0)
            if count > 0:
                warnings.append(
                    f"{label}: topic {channel.topic!r} has {count + 1} channels, "
                    f"using schema from channel_id={channel.id}"
                )
            seen_topics[channel.topic] = count + 1

            schema = fs.summary.schemas.get(channel.schema_id) if channel.schema_id != 0 else None
            if schema is not None:
                fp = _schema_fingerprint(schema)
                entry = (schema.name, schema.encoding, fp)
            else:
                entry = None
            msg_enc = channel.message_encoding
            topic_info.setdefault(channel.topic, {})[label] = (
                (entry[0], entry[1], entry[2], msg_enc) if entry is not None else None
            )

    mismatches: list[ChannelSchemaMismatch] = []
    for topic in sorted(topic_info):
        schema_names: dict[str, str | None] = {}
        schema_encodings: dict[str, str | None] = {}
        message_encodings: dict[str, str | None] = {}
        fingerprints: set[str] = set()
        msg_encs: set[str] = set()

        for label in all_summaries:
            info = topic_info[topic].get(label)
            if info is not None:
                schema_names[label] = info[0]
                schema_encodings[label] = info[1]
                fingerprints.add(info[2])
                message_encodings[label] = info[3]
                msg_encs.add(info[3])
            else:
                schema_names[label] = None
                schema_encodings[label] = None
                message_encodings[label] = None

        if len(fingerprints) > 1 or len(msg_encs) > 1:
            mismatches.append(
                ChannelSchemaMismatch(
                    topic=topic,
                    schema_names=schema_names,
                    schema_encodings=schema_encodings,
                    message_encodings=message_encodings,
                )
            )

    return mismatches, warnings


def _unique_label(path: str, seen: dict[str, int]) -> str:
    base = _file_label(path)
    count = seen.get(base, 0) + 1
    seen[base] = count
    if count == 1:
        return base
    return f"{base}#{count}"


def _extract_summary(path: str, label: str, info: RebuildInfo, file_size: int) -> FileSummary:
    summary = info.summary
    stats = summary.statistics
    assert stats is not None
    return FileSummary(
        label=label,
        path=path,
        size_bytes=file_size,
        summary=summary,
        statistics=stats,
    )


def _process_file(
    path: str, label: str
) -> tuple[FileSummary, dict[int, set[int]], MessageIndexIdentityReadResult]:
    result = read_compare_file(path, rebuild_missing=True)
    indexed = MessageIndexIdentityReadResult(
        path=path,
        size_bytes=result.size_bytes,
        identity=message_index_identity_from_info(result.info),
        read_mode=result.read_mode,
    )
    return (
        _extract_summary(path, label, result.info, result.size_bytes),
        collect_message_timestamps(result.info),
        indexed,
    )


def _compare_channels(
    all_timestamps: dict[str, dict[int, set[int]]],
    all_summaries: dict[str, FileSummary],
) -> dict[str, ChannelDiff]:
    topic_timestamps: dict[str, dict[str, set[int]]] = {}
    for label, fs in all_summaries.items():
        ch_by_id = fs.summary.channels
        ts_by_ch = all_timestamps.get(label, {})
        per_topic: dict[str, set[int]] = {}
        for ch_id, timestamps in ts_by_ch.items():
            ch = ch_by_id.get(ch_id)
            topic = ch.topic if ch is not None else f"Channel_{ch_id}"
            if topic in per_topic:
                per_topic[topic] |= timestamps
            else:
                per_topic[topic] = set(timestamps)
        topic_timestamps[label] = per_topic

    all_topics: set[str] = set()
    for ts in topic_timestamps.values():
        all_topics |= ts.keys()

    labels = list(all_summaries.keys())
    diffs: dict[str, ChannelDiff] = {}
    for topic in all_topics:
        per_file_ts = {label: topic_timestamps[label].get(topic, set()) for label in labels}
        common = set.intersection(*per_file_ts.values()) if per_file_ts else set()
        diffs[topic] = ChannelDiff(
            topic=topic,
            counts={label: len(ts) for label, ts in per_file_ts.items()},
            timestamps={
                label: ts - set.union(*(s for k, s in per_file_ts.items() if k != label), set())
                for label, ts in per_file_ts.items()
            },
            common_count=len(common),
        )
    return diffs


_T = TypeVar("_T")


def _format_values(
    summaries: list[FileSummary],
    selector: Callable[[FileSummary], _T],
    formatter: Callable[[_T], str] | None = None,
) -> list[str]:
    if formatter is None:
        formatter = str
    values = [selector(fs) for fs in summaries]
    all_equal = len(set(values)) == 1
    color = "green" if all_equal else "yellow"
    return [f"[{color}]{formatter(v)}[/]" for v in values]


def _format_number_diffs(
    summaries: list[FileSummary],
    selector: Callable[[FileSummary], int],
) -> list[str]:
    values = [selector(fs) for fs in summaries]
    ref = values[0]
    all_equal = len(set(values)) == 1
    result: list[str] = []
    for v in values:
        s = f"{v:,}"
        if all_equal:
            result.append(f"[green]{s}[/]")
        elif v == ref:
            result.append(f"[yellow]{s}[/]")
        else:
            diff = v - ref
            sign = "+" if diff > 0 else ""
            result.append(f"[yellow]{s} ({sign}{diff:,})[/]")
    return result


def _build_summary_table(summaries: list[FileSummary]) -> Table:
    table = Table(title="File Comparison")
    table.add_column("Property", style="bold cyan")
    for fs in summaries:
        table.add_column(fs.label, justify="right")

    table.add_row("Size", *_format_values(summaries, lambda fs: fs.size_bytes, bytes_to_human))
    table.add_row(
        "Messages", *_format_number_diffs(summaries, lambda fs: fs.statistics.message_count)
    )
    table.add_row("Duration", *_format_values(summaries, lambda fs: fs.duration_ns, duration_human))
    table.add_row(
        "Start Time",
        *_format_values(summaries, lambda fs: fs.statistics.message_start_time, time_str),
    )
    table.add_row(
        "End Time",
        *_format_values(summaries, lambda fs: fs.statistics.message_end_time, time_str),
    )
    table.add_row("Chunks", *_format_number_diffs(summaries, lambda fs: fs.statistics.chunk_count))
    table.add_row(
        "Channels", *_format_number_diffs(summaries, lambda fs: fs.statistics.channel_count)
    )
    table.add_row(
        "Attachments",
        *_format_number_diffs(summaries, lambda fs: fs.statistics.attachment_count),
    )
    table.add_row(
        "Metadata", *_format_number_diffs(summaries, lambda fs: fs.statistics.metadata_count)
    )

    return table


def _build_channel_diff_table(
    diffs: dict[str, ChannelDiff],
    labels: list[str],
    *,
    skip_identical: bool = False,
) -> Table | None:
    if not diffs:
        return None

    table = Table(title="Message Index Diff (by Channel)")
    table.add_column("Topic", style="bold cyan")
    for label in labels:
        table.add_column(label, justify="right")
    table.add_column("Common", justify="right")

    first_label = labels[0]
    if len(labels) == 2:
        table.add_column("Added", justify="right")
        table.add_column("Removed", justify="right")

    sorted_diffs = sorted(diffs.values(), key=lambda d: d.topic)
    identical_count = 0

    for diff in sorted_diffs:
        if diff.is_identical:
            identical_count += 1
            if skip_identical:
                continue

        cells: list[str] = []
        ref_count = diff.counts.get(first_label, 0)
        all_counts_equal = len(set(diff.counts.values())) <= 1

        for label in labels:
            count = diff.counts.get(label, 0)
            s = f"{count:,}" if count > 0 else "[dim]0[/]"
            if count != ref_count:
                s = f"[yellow]{s}[/]"
            elif all_counts_equal and count > 0:
                s = f"[green]{s}[/]"
            cells.append(s)

        common = f"{diff.common_count:,}" if diff.common_count > 0 else "[dim]0[/]"
        cells.append(common)

        if len(labels) == 2:
            other_label = labels[1]
            added = diff.unique_in(other_label)
            removed = diff.unique_in(first_label)
            cells.append(f"[green]+{added:,}[/]" if added > 0 else "[dim]0[/]")
            cells.append(f"[red]-{removed:,}[/]" if removed > 0 else "[dim]0[/]")

        table.add_row(diff.topic, *cells)

    if skip_identical and identical_count > 0:
        table.caption = f"[dim]{identical_count} identical channels hidden[/]"

    return table


def _format_timestamp_ranges(
    timestamps: set[int], max_ranges: int = 3, *, total: int | None = None
) -> str:
    if not timestamps:
        return "[dim]-[/]"

    if total is not None and len(timestamps) == total:
        return f"all ({len(timestamps):,} msgs)"

    segments = split_timestamps_into_segments(sorted(timestamps))
    parts: list[str] = []

    for seg in segments[:max_ranges]:
        if len(seg) == 1:
            parts.append(format_ts_short(seg[0]))
        else:
            parts.append(
                f"{format_ts_short(seg[0])} - {format_ts_short(seg[-1])} ({len(seg):,} msgs)"
            )

    total_msgs = len(timestamps)
    remaining = total_msgs - sum(len(s) for s in segments[:max_ranges])
    if remaining > 0:
        remaining_segs = len(segments) - max_ranges
        parts.append(f"[dim]+{remaining:,} msgs in {remaining_segs} more ranges[/]")

    return ", ".join(parts)


def _build_sample_diffs_table(
    diffs: dict[str, ChannelDiff],
    labels: list[str],
    max_ranges: int = 3,
) -> Table | None:
    has_diffs = any(not d.is_identical for d in diffs.values())
    if not has_diffs:
        return None

    table = Table(title="Differing Timestamps")
    table.add_column("Topic", style="bold cyan")
    for label in labels:
        table.add_column(f"Only in {label}", justify="right")

    for diff in sorted(diffs.values(), key=lambda d: d.topic):
        if diff.is_identical:
            continue

        cells = [
            _format_timestamp_ranges(
                diff.timestamps.get(label, set()),
                max_ranges=max_ranges,
                total=diff.counts.get(label),
            )
            for label in labels
        ]
        table.add_row(diff.topic, *cells)

    return table


def _build_schema_diff_table(
    schema_diffs: dict[str, SchemaDiff],
    labels: list[str],
) -> Table | None:
    non_identical = [d for d in schema_diffs.values() if not d.is_identical]
    if not non_identical:
        return None

    table = Table(title="Schema Differences")
    table.add_column("Schema", style="bold cyan")
    for label in labels:
        table.add_column(label, justify="right")

    for diff in non_identical:
        cells: list[str] = []
        for label in labels:
            if diff.per_label.get(label) is None:
                cells.append("[dim]missing[/]")
            else:
                cell = f"[yellow]{diff.per_label[label]}[/]"
                enc = diff.encodings.get(label)
                if enc:
                    cell += f"\n[dim]{enc}[/]"
                cells.append(cell)
        table.add_row(diff.name, *cells)

    identical_count = len(schema_diffs) - len(non_identical)
    if identical_count > 0:
        table.caption = f"[dim]{identical_count} identical schemas hidden[/]"

    return table


def _build_channel_schema_mismatch_table(
    mismatches: list[ChannelSchemaMismatch],
    labels: list[str],
) -> Table | None:
    if not mismatches:
        return None

    table = Table(title="Channel Mismatches")
    table.add_column("Topic", style="bold cyan")
    for label in labels:
        table.add_column(label, justify="right")

    for mm in mismatches:
        cells: list[str] = []
        for label in labels:
            name = mm.schema_names.get(label)
            if name is None:
                cells.append("[dim]missing[/]")
                continue
            parts = [f"[red]{name}[/]"]
            sch_enc = mm.schema_encodings.get(label)
            msg_enc = mm.message_encodings.get(label)
            dim_parts = []
            if sch_enc:
                dim_parts.append(f"schema: {sch_enc}")
            if msg_enc:
                dim_parts.append(f"encoding: {msg_enc}")
            if dim_parts:
                parts.append(f"[dim]{', '.join(dim_parts)}[/]")
            cells.append("\n".join(parts))
        table.add_row(mm.topic, *cells)

    return table


def _format_count(value: int, singular: str, plural: str | None = None) -> str:
    unit = singular if value == 1 else plural or f"{singular}s"
    return f"{value:,} {unit}"


def _format_optional_time_window(start_time: int | None, end_time: int | None) -> str:
    if start_time is None or end_time is None:
        return "N/A"
    if start_time == end_time:
        return format_ts_short(start_time)
    return f"{format_ts_short(start_time)} - {format_ts_short(end_time)}"


def _format_range_summary(summary: TimestampRangeSummary) -> str:
    if not summary.ranges and summary.hidden_messages == 0:
        return "-"

    parts: list[str] = []
    for segment in summary.ranges:
        count_label = _format_count(segment.message_count, "msg")
        if segment.start_time == segment.end_time:
            parts.append(f"{format_ts_short(segment.start_time)} ({count_label})")
        else:
            parts.append(
                f"{format_ts_short(segment.start_time)} - "
                f"{format_ts_short(segment.end_time)} ({count_label})"
            )

    if summary.hidden_messages > 0:
        parts.append(f"+{_format_count(summary.hidden_messages, 'msg')}")

    return ", ".join(parts)


def _verdict_text(
    comparison: IndexedComparison,
    *,
    left_label: str,
    right_label: str,
) -> str:
    if comparison.kind is IndexedCompareKind.EXACT:
        return "[green]exact indexed duplicate[/green]"
    if comparison.kind is IndexedCompareKind.FULL_OVERLAP:
        return "[green]full message overlap[/green] [dim](footer/statistics differ)[/dim]"
    if comparison.kind is IndexedCompareKind.LEFT_SUBSET:
        return f"[yellow]{escape(left_label)} is contained in {escape(right_label)}[/yellow]"
    if comparison.kind is IndexedCompareKind.RIGHT_SUBSET:
        return f"[yellow]{escape(right_label)} is contained in {escape(left_label)}[/yellow]"
    if comparison.kind is IndexedCompareKind.EDGE_OVERLAP:
        return "[yellow]edge overlap[/yellow]"
    if comparison.kind is IndexedCompareKind.PARTIAL_OVERLAP:
        return "[yellow]partial overlap[/yellow]"
    return "[red]no indexed message overlap[/red]"


def _build_smart_diff_tree(
    comparisons: list[LabeledIndexedComparison],
    *,
    max_topic_rows: int = 3,
) -> Tree:
    root = Tree("[bold cyan]Smart Diff Verdict[/bold cyan]")

    for item in comparisons:
        comparison = item.comparison
        verdict = _verdict_text(
            comparison,
            left_label=item.left_label,
            right_label=item.right_label,
        )
        node = root.add(
            f"[bold]{escape(item.left_label)} <-> {escape(item.right_label)}[/bold]: "
            f"{verdict} "
            f"[dim]({_format_count(comparison.shared_messages, 'shared msg', 'shared msgs')}, "
            f"{_format_count(comparison.shared_channels, 'shared channel')}; "
            f"left-only {_format_count(comparison.left_extra_messages, 'msg')}, "
            f"right-only {_format_count(comparison.right_extra_messages, 'msg')})[/dim]"
        )

        for topic in comparison.topics[:max_topic_rows]:
            overlap_window = _format_optional_time_window(
                topic.shared_start_time, topic.shared_end_time
            )
            topic_node = node.add(
                f"[cyan]{escape(topic.topic)}[/cyan] "
                f"[dim]({_format_count(topic.shared_messages, 'shared msg', 'shared msgs')}, "
                f"left-only {_format_count(topic.left_only_messages, 'msg')}, "
                f"right-only {_format_count(topic.right_only_messages, 'msg')}, "
                f"overlap {overlap_window})[/dim]"
            )
            if topic.left_only_messages or topic.right_only_messages:
                topic_node.add(
                    f"[dim]left-only: {_format_range_summary(topic.left_only_ranges)}; "
                    f"right-only: {_format_range_summary(topic.right_only_ranges)}[/dim]"
                )

        hidden_topic_count = len(comparison.topics) - max_topic_rows
        if hidden_topic_count > 0:
            node.add(f"[dim]... {hidden_topic_count:,} more topic(s)[/dim]")

    return root


def _compare_to_first(
    indexed_results: list[MessageIndexIdentityReadResult],
    labels: list[str],
    *,
    max_range_preview: int,
) -> list[LabeledIndexedComparison]:
    first = indexed_results[0]
    first_label = labels[0]
    return [
        LabeledIndexedComparison(
            left_label=first_label,
            right_label=labels[index],
            comparison=compare_indexed_identities(
                first,
                indexed_result,
                max_range_preview=max_range_preview,
            ),
        )
        for index, indexed_result in enumerate(indexed_results[1:], start=1)
    ]


def diff_cmd(
    files: Annotated[
        list[str],
        Parameter(
            name=["files"],
            help="Paths to MCAP files to compare (local files or HTTP/HTTPS URLs)",
        ),
    ],
    *,
    skip_identical: Annotated[
        bool,
        Parameter(
            name=["--skip-identical"],
            help="Hide channels with identical message timestamps",
        ),
    ] = False,
    max_ranges: Annotated[
        int,
        Parameter(
            name=["--max-ranges"],
            help="Maximum number of timestamp ranges to show per channel",
        ),
    ] = 3,
) -> int:
    """Compare MCAP files using summary and message index timestamps.

    Fast comparison reads indexed files through their footer/summary first,
    then falls back to rebuilding metadata from the data section when the
    footer summary is unavailable.

    Parameters
    ----------
    files
        Paths to MCAP files to compare (2 or more)
    skip_identical
        Hide channels where all message timestamps match exactly
    max_ranges
        Maximum timestamp ranges to display per channel (default: 3)

    Examples
    --------
    ```
    # Compare two files
    pymcap-cli diff recording1.mcap recording2.mcap

    # Compare three files
    pymcap-cli diff a.mcap b.mcap c.mcap

    # Show only channels with differences
    pymcap-cli diff file1.mcap file2.mcap --skip-identical
    ```
    """
    if len(files) < 2:
        logger.error("At least two files must be specified")
        return 1

    summaries: list[FileSummary] = []
    all_timestamps: dict[str, dict[int, set[int]]] = {}
    all_summaries: dict[str, FileSummary] = {}
    indexed_results: list[MessageIndexIdentityReadResult] = []
    seen_labels: dict[str, int] = {}

    for path in files:
        label = _unique_label(path, seen_labels)
        try:
            fs, ts, indexed = _process_file(path, label)
        except Exception:
            logger.exception(f"Error reading {path}")
            return 1
        summaries.append(fs)
        all_timestamps[label] = ts
        all_summaries[label] = fs
        indexed_results.append(indexed)

    labels = [fs.label for fs in summaries]
    first_label = labels[0]
    smart_comparisons = _compare_to_first(
        indexed_results,
        labels,
        max_range_preview=max_ranges,
    )

    channel_diffs = _compare_channels(all_timestamps, all_summaries)
    schema_diffs = _compare_schemas(all_summaries)
    channel_schema_mismatches, topic_warnings = _check_channel_schema_mismatches(all_summaries)

    total_common = sum(d.common_count for d in channel_diffs.values())
    total_added = sum(d.unique_in(lbl) for d in channel_diffs.values() for lbl in labels[1:])
    total_removed = sum(d.unique_in(first_label) for d in channel_diffs.values())
    has_diffs = total_added > 0 or total_removed > 0
    has_schema_diffs = any(not d.is_identical for d in schema_diffs.values())
    has_mismatches = bool(channel_schema_mismatches)

    console.print()
    console.print(_build_smart_diff_tree(smart_comparisons))

    console.print()
    console.print(_build_summary_table(summaries))

    if topic_warnings:
        console.print()
        for w in topic_warnings:
            console.print(f"[yellow]⚠ {w}[/]")

    channel_table = _build_channel_diff_table(channel_diffs, labels, skip_identical=skip_identical)
    if channel_table:
        console.print()
        console.print(channel_table)

    if has_diffs:
        sample_table = _build_sample_diffs_table(channel_diffs, labels, max_ranges=max_ranges)
        if sample_table:
            console.print()
            console.print(sample_table)

    if has_schema_diffs:
        schema_table = _build_schema_diff_table(schema_diffs, labels)
        if schema_table:
            console.print()
            console.print(schema_table)

    if has_mismatches:
        mismatch_table = _build_channel_schema_mismatch_table(channel_schema_mismatches, labels)
        if mismatch_table:
            console.print()
            console.print(mismatch_table)

    console.print()
    all_good = not has_diffs and not has_schema_diffs and not has_mismatches
    if all_good:
        console.print(
            f"[green]✓ All {total_common:,} messages have identical timestamps and schemas[/]"
        )
    else:
        if not has_diffs:
            console.print(f"[green]✓ All {total_common:,} messages have identical timestamps[/]")
        else:
            console.print(f"[green]✓ {total_common:,} messages match[/]")
            if total_added > 0:
                console.print(f"[yellow]⚠ {total_added:,} messages added in other files[/]")
            if total_removed > 0:
                console.print(f"[red]⚠ {total_removed:,} messages removed from {first_label}[/]")
        if has_schema_diffs:
            diff_count = sum(1 for d in schema_diffs.values() if not d.is_identical)
            console.print(f"[red]⚠ {diff_count} schema(s) differ across files[/]")
        if has_mismatches:
            console.print(
                f"[red]⚠ {len(channel_schema_mismatches)} "
                f"channel(s) use different schemas across files[/]"
            )

    return 0
