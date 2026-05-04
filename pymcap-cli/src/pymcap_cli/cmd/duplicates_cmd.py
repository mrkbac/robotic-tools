"""Find likely duplicate MCAP recordings with a two-stage compare."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from cyclopts import Parameter
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.tree import Tree

from pymcap_cli.core.mcap_compare import (
    IdentityReadResult,
    IndexedChannelIdentity,
    IndexReadProgress,
    MessageIndexIdentity,
    MessageIndexIdentityReadResult,
    SummaryChannelRange,
    discover_mcap_candidates,
    path_basename,
    read_identity_file,
    read_message_index_identity_file,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)
console = Console()

_NS_TO_SEC = 1_000_000_000
_MAX_SKIPPED_DETAILS = 10
_PROGRESS_PAIR_UPDATE_INTERVAL = 128
_PROGRESS_INDEX_UPDATE_INTERVAL = 64


def _create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[current]}"),
        console=ERR,
        transient=False,
    )


def _progress_file(path: str) -> str:
    return escape(path_basename(path))


def _progress_pair(left_path: str, right_path: str) -> str:
    return f"{_progress_file(left_path)} <-> {_progress_file(right_path)}"


def _maybe_update_progress(
    progress: Progress | None,
    task: TaskID | None,
    *,
    completed: int,
    is_last: bool,
    interval: int,
    current: str,
) -> None:
    if progress is None or task is None:
        return
    if not is_last and completed % interval != 0:
        return
    progress.update(task, completed=completed, current=current)


@dataclass(frozen=True, slots=True)
class SkippedFile:
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class PartialMatch:
    left: MessageIndexIdentityReadResult
    right: MessageIndexIdentityReadResult
    shared_channels: int
    shared_messages: int

    @property
    def left_extra_channels(self) -> int:
        return len(self.left.identity.indexed_channels) - self.shared_channels

    @property
    def right_extra_channels(self) -> int:
        return len(self.right.identity.indexed_channels) - self.shared_channels

    @property
    def left_extra_messages(self) -> int:
        return self.left.identity.message_count - self.shared_messages

    @property
    def right_extra_messages(self) -> int:
        return self.right.identity.message_count - self.shared_messages


@dataclass(frozen=True, slots=True)
class AnchoredPartialRelation:
    match: PartialMatch
    anchor: MessageIndexIdentityReadResult
    related: MessageIndexIdentityReadResult

    @property
    def message_delta(self) -> int:
        return self.related.identity.message_count - self.anchor.identity.message_count

    @property
    def channel_delta(self) -> int:
        return self.related.identity.channel_count - self.anchor.identity.channel_count

    @property
    def anchor_extra_messages(self) -> int:
        if self.anchor.path == self.match.left.path:
            return self.match.left_extra_messages
        return self.match.right_extra_messages

    @property
    def related_extra_messages(self) -> int:
        if self.related.path == self.match.left.path:
            return self.match.left_extra_messages
        return self.match.right_extra_messages


@dataclass(frozen=True, slots=True)
class IndexedAnalysis:
    duplicate_groups: list[list[MessageIndexIdentityReadResult]]
    partial_matches: list[PartialMatch]


def _read_summary_identity(path: str, *, rebuild_missing: bool) -> IdentityReadResult | SkippedFile:
    try:
        result = read_identity_file(path, rebuild_missing=rebuild_missing)
        if result is None:
            return SkippedFile(path=path, reason="no summary/statistics")
    except Exception as exc:  # noqa: BLE001
        return SkippedFile(path=path, reason=str(exc) or exc.__class__.__name__)
    else:
        return result


def _read_index_identity(
    path: str,
    *,
    rebuild_missing: bool,
    index_progress: IndexReadProgress | None = None,
) -> MessageIndexIdentityReadResult | SkippedFile:
    try:
        result = read_message_index_identity_file(
            path,
            rebuild_missing=rebuild_missing,
            index_progress=index_progress,
        )
        if result is None:
            return SkippedFile(path=path, reason="no complete message indexes")
    except Exception as exc:  # noqa: BLE001
        return SkippedFile(path=path, reason=str(exc) or exc.__class__.__name__)
    else:
        return result


def _format_time(time_ns: int) -> str:
    if time_ns == 0:
        return "N/A"
    return datetime.fromtimestamp(time_ns / _NS_TO_SEC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _time_range(identity: MessageIndexIdentity) -> str:
    return (
        f"{_format_time(identity.message_start_time)} - {_format_time(identity.message_end_time)}"
    )


def _file_label(path: str) -> str:
    return f"[green]{path_basename(path)}[/green]\n[dim]{path}[/dim]"


def _format_count(value: int, singular: str, plural: str | None = None) -> str:
    unit = singular if value == 1 else plural or f"{singular}s"
    return f"{value:,} {unit}"


def _identity_summary(identity: MessageIndexIdentity) -> str:
    return (
        f"{_format_count(identity.message_count, 'msg')}, "
        f"{_format_count(identity.channel_count, 'channel')}, "
        f"{_format_count(identity.schema_count, 'schema')}, "
        f"{_time_range(identity)}"
    )


def _format_delta(value: int) -> str:
    return f"{value:+,}" if value else "0"


def _build_groups_tree(groups: list[list[MessageIndexIdentityReadResult]]) -> Tree:
    root = Tree("[bold cyan]Duplicate MCAP Groups[/bold cyan]")

    for group_index, group in enumerate(groups, start=1):
        representative = group[0].identity
        group_node = root.add(
            f"[bold]Group {group_index}[/bold] "
            f"[dim]({len(group)} files, {_identity_summary(representative)})[/dim]"
        )
        for scanned_file in sorted(group, key=lambda item: item.path):
            group_node.add(_file_label(scanned_file.path))

    return root


def _partial_anchor_sort_key(scanned_file: MessageIndexIdentityReadResult) -> tuple[int, int, str]:
    return (
        scanned_file.identity.message_count,
        scanned_file.identity.channel_count,
        scanned_file.path,
    )


def _anchored_relation(match: PartialMatch) -> AnchoredPartialRelation:
    if _partial_anchor_sort_key(match.left) >= _partial_anchor_sort_key(match.right):
        return AnchoredPartialRelation(match=match, anchor=match.left, related=match.right)
    return AnchoredPartialRelation(match=match, anchor=match.right, related=match.left)


def _anchored_partial_relations(
    matches: list[PartialMatch],
) -> list[tuple[MessageIndexIdentityReadResult, list[AnchoredPartialRelation]]]:
    relations_by_anchor: dict[str, list[AnchoredPartialRelation]] = defaultdict(list)
    anchors_by_path: dict[str, MessageIndexIdentityReadResult] = {}

    for match in matches:
        relation = _anchored_relation(match)
        anchors_by_path[relation.anchor.path] = relation.anchor
        relations_by_anchor[relation.anchor.path].append(relation)

    anchored = [
        (
            anchors_by_path[anchor_path],
            sorted(
                relations,
                key=lambda relation: (
                    -relation.match.shared_messages,
                    -relation.match.shared_channels,
                    relation.related.path,
                ),
            ),
        )
        for anchor_path, relations in relations_by_anchor.items()
    ]
    return sorted(
        anchored,
        key=lambda item: (
            -len(item[1]),
            -sum(relation.match.shared_messages for relation in item[1]),
            -item[0].identity.message_count,
            item[0].path,
        ),
    )


def _build_partial_tree(matches: list[PartialMatch]) -> Tree:
    root = Tree("[bold cyan]Partial MCAP Matches[/bold cyan]")

    for index, (anchor, relations) in enumerate(_anchored_partial_relations(matches), start=1):
        anchor_node = root.add(
            f"[bold]Anchor {index}: {path_basename(anchor.path)}[/bold] "
            f"[dim]({len(relations)} relation(s), {_identity_summary(anchor.identity)})[/dim]"
        )
        anchor_node.add(f"[dim]{anchor.path}[/dim]")

        for relation in relations:
            related_node = anchor_node.add(
                f"[green]{path_basename(relation.related.path)}[/green] "
                f"[dim](msgs {_format_delta(relation.message_delta)}, "
                f"channels {_format_delta(relation.channel_delta)}; "
                f"{_format_count(relation.match.shared_messages, 'shared msg', 'shared msgs')}, "
                f"{_format_count(relation.match.shared_channels, 'shared channel')}; "
                f"anchor-only {_format_count(relation.anchor_extra_messages, 'msg')}, "
                f"file-only {_format_count(relation.related_extra_messages, 'msg')})[/dim]"
            )
            related_node.add(f"[dim]{relation.related.path}[/dim]")

    return root


def _print_skipped_summary(skipped: list[SkippedFile]) -> None:
    if not skipped:
        return

    reasons = Counter(file.reason for file in skipped)
    reason_summary = ", ".join(f"{count} {reason}" for reason, count in sorted(reasons.items()))
    console.print(f"[yellow]Skipped {len(skipped)} file(s): {reason_summary}[/yellow]")

    table = Table(title="Skipped Files")
    table.add_column("Path", style="yellow")
    table.add_column("Reason")
    for skipped_file in skipped[:_MAX_SKIPPED_DETAILS]:
        table.add_row(skipped_file.path, skipped_file.reason)
    if len(skipped) > _MAX_SKIPPED_DETAILS:
        table.caption = f"{len(skipped) - _MAX_SKIPPED_DETAILS} more skipped file(s) hidden"
    console.print(table)


def _sort_groups(
    groups: list[list[MessageIndexIdentityReadResult]],
) -> list[list[MessageIndexIdentityReadResult]]:
    return sorted(
        groups,
        key=lambda group: (-len(group), group[0].identity.message_count, group[0].path),
    )


def _summary_groups(scanned: list[IdentityReadResult]) -> dict[str, list[IdentityReadResult]]:
    groups: dict[str, list[IdentityReadResult]] = defaultdict(list)
    for scanned_file in scanned:
        groups[scanned_file.identity.digest].append(scanned_file)
    return groups


def _ranges_overlap(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> bool:
    if left_start is None or left_end is None or right_start is None or right_end is None:
        return False
    return max(left_start, right_start) <= min(left_end, right_end)


def _summary_ranges_by_semantic_digest(
    ranges: tuple[SummaryChannelRange, ...],
) -> dict[str, list[SummaryChannelRange]]:
    result: dict[str, list[SummaryChannelRange]] = defaultdict(list)
    for channel_range in ranges:
        result[channel_range.channel_semantic_digest].append(channel_range)
    return result


def _has_approx_channel_overlap(
    left_ranges: dict[str, list[SummaryChannelRange]],
    right_ranges: dict[str, list[SummaryChannelRange]],
) -> bool:
    for channel_digest in left_ranges.keys() & right_ranges.keys():
        for left_range in left_ranges[channel_digest]:
            for right_range in right_ranges[channel_digest]:
                if _ranges_overlap(
                    left_range.message_start_time,
                    left_range.message_end_time,
                    right_range.message_start_time,
                    right_range.message_end_time,
                ):
                    return True

    return False


def _candidate_paths_for_index_stage(
    scanned: list[IdentityReadResult],
    *,
    include_all: bool,
    progress: Progress | None = None,
    task: TaskID | None = None,
) -> set[str]:
    paths: set[str] = set()
    grouped_by_summary = _summary_groups(scanned)
    completed_pairs = 0
    pair_count = len(scanned) * (len(scanned) - 1) // 2

    for group in grouped_by_summary.values():
        if include_all or len(group) > 1:
            paths.update(summary_result.path for summary_result in group)

    ranges_by_file = [
        _summary_ranges_by_semantic_digest(scanned_file.identity.channel_ranges)
        for scanned_file in scanned
    ]

    for index, left in enumerate(scanned):
        left_ranges = ranges_by_file[index]
        for right_offset, right in enumerate(scanned[index + 1 :], start=index + 1):
            if _has_approx_channel_overlap(left_ranges, ranges_by_file[right_offset]):
                paths.add(left.path)
                paths.add(right.path)
            completed_pairs += 1
            _maybe_update_progress(
                progress,
                task,
                completed=completed_pairs,
                is_last=completed_pairs == pair_count,
                interval=_PROGRESS_PAIR_UPDATE_INTERVAL,
                current=_progress_pair(left.path, right.path),
            )

    if progress is not None and task is not None:
        progress.update(
            task,
            completed=pair_count,
            current=f"{len(paths):,} candidate file(s)",
        )

    return paths


def _read_candidate_index_identities(
    candidate_paths: set[str],
    *,
    rebuild_missing: bool,
    progress: Progress | None = None,
    task: TaskID | None = None,
) -> tuple[list[MessageIndexIdentityReadResult], list[SkippedFile]]:
    scanned: list[MessageIndexIdentityReadResult] = []
    skipped: list[SkippedFile] = []

    sorted_paths = sorted(candidate_paths)
    file_count = len(sorted_paths)
    for file_index, path in enumerate(sorted_paths):
        file_number = file_index + 1
        progress_file = _progress_file(path)
        if progress is not None and task is not None:
            progress.update(
                task,
                completed=file_index,
                current=f"{file_number:,}/{file_count:,} {progress_file}",
            )

        def index_progress(
            completed_indexes: int,
            total_indexes: int,
            progress_file_index: int = file_index,
            progress_file_number: int = file_number,
            progress_file_name: str = progress_file,
        ) -> None:
            if progress is None or task is None:
                return
            if (
                completed_indexes not in (0, total_indexes)
                and completed_indexes % _PROGRESS_INDEX_UPDATE_INTERVAL != 0
            ):
                return
            if total_indexes:
                completed = progress_file_index + (completed_indexes / total_indexes)
                current = (
                    f"{progress_file_number:,}/{file_count:,} {progress_file_name} "
                    f"({completed_indexes:,}/{total_indexes:,} index records)"
                )
            else:
                completed = progress_file_index
                current = f"{progress_file_number:,}/{file_count:,} {progress_file_name}"
            progress.update(task, completed=completed, current=current)

        result = _read_index_identity(
            path,
            rebuild_missing=rebuild_missing,
            index_progress=index_progress,
        )
        if isinstance(result, MessageIndexIdentityReadResult):
            scanned.append(result)
        else:
            skipped.append(result)
        if progress is not None and task is not None:
            progress.update(task, completed=file_index + 1)

    if progress is not None and task is not None:
        progress.update(
            task,
            completed=len(sorted_paths),
            current=f"{len(scanned):,} indexed file(s)",
        )

    return scanned, skipped


def _message_bearing_channel_count(scanned_file: MessageIndexIdentityReadResult) -> int:
    return sum(1 for channel in scanned_file.identity.indexed_channels if channel.message_count)


def _is_full_index_overlap(
    left: MessageIndexIdentityReadResult,
    right: MessageIndexIdentityReadResult,
    *,
    shared_channels: int,
    shared_messages: int,
    left_message_bearing_channels: int,
    right_message_bearing_channels: int,
) -> bool:
    return (
        shared_messages == left.identity.message_count
        and shared_messages == right.identity.message_count
        and shared_channels == left_message_bearing_channels
        and shared_channels == right_message_bearing_channels
    )


def _analyze_indexed_matches(
    scanned: list[MessageIndexIdentityReadResult],
    *,
    include_all: bool,
    progress: Progress | None = None,
    task: TaskID | None = None,
) -> IndexedAnalysis:
    parent = list(range(len(scanned)))
    matches: list[PartialMatch] = []
    completed_pairs = 0
    pair_count = len(scanned) * (len(scanned) - 1) // 2

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_digest: dict[str, int] = {}
    for index, scanned_file in enumerate(scanned):
        first_index = first_by_digest.get(scanned_file.identity.digest)
        if first_index is None:
            first_by_digest[scanned_file.identity.digest] = index
            continue
        union(first_index, index)

    counters_by_file = [
        _channel_counters_by_digest(scanned_file.identity.indexed_channels)
        for scanned_file in scanned
    ]
    message_bearing_by_file = [
        _message_bearing_channel_count(scanned_file) for scanned_file in scanned
    ]

    for left_index, left in enumerate(scanned):
        for right_index, right in enumerate(scanned[left_index + 1 :], start=left_index + 1):
            if left.identity.digest == right.identity.digest:
                completed_pairs += 1
                continue

            _maybe_update_progress(
                progress,
                task,
                completed=completed_pairs,
                is_last=completed_pairs + 1 == pair_count,
                interval=_PROGRESS_PAIR_UPDATE_INTERVAL,
                current=_progress_pair(left.path, right.path),
            )

            shared_channels, shared_messages = _indexed_overlap(
                counters_by_file[left_index], counters_by_file[right_index]
            )
            if _is_full_index_overlap(
                left,
                right,
                shared_channels=shared_channels,
                shared_messages=shared_messages,
                left_message_bearing_channels=message_bearing_by_file[left_index],
                right_message_bearing_channels=message_bearing_by_file[right_index],
            ):
                union(left_index, right_index)
            elif shared_channels > 0 and shared_messages > 0:
                matches.append(
                    PartialMatch(
                        left=left,
                        right=right,
                        shared_channels=shared_channels,
                        shared_messages=shared_messages,
                    )
                )
            completed_pairs += 1

    if progress is not None and task is not None:
        progress.update(task, completed=pair_count, current=f"{len(matches):,} partial match(es)")

    groups_by_root: dict[int, list[MessageIndexIdentityReadResult]] = defaultdict(list)
    for index, scanned_file in enumerate(scanned):
        groups_by_root[find(index)].append(scanned_file)

    duplicate_groups = [group for group in groups_by_root.values() if include_all or len(group) > 1]

    return IndexedAnalysis(
        duplicate_groups=_sort_groups(duplicate_groups),
        partial_matches=sorted(
            matches,
            key=lambda match: (
                -match.shared_messages,
                -match.shared_channels,
                match.left.path,
                match.right.path,
            ),
        ),
    )


def _channel_counters_by_digest(
    channels: tuple[IndexedChannelIdentity, ...],
) -> dict[str, list[Counter[int]]]:
    result: dict[str, list[Counter[int]]] = defaultdict(list)
    for channel in channels:
        result[channel.channel_semantic_digest].append(Counter(channel.timestamps))
    return result


def _indexed_overlap(
    left_counters: dict[str, list[Counter[int]]],
    right_counters: dict[str, list[Counter[int]]],
) -> tuple[int, int]:
    candidate_pairs: list[tuple[int, str, int, int]] = []

    for channel_digest in left_counters.keys() & right_counters.keys():
        left_group = left_counters[channel_digest]
        right_group = right_counters[channel_digest]
        for left_index, left_counter in enumerate(left_group):
            for right_index, right_counter in enumerate(right_group):
                shared_messages = sum((left_counter & right_counter).values())
                if shared_messages > 0:
                    candidate_pairs.append(
                        (shared_messages, channel_digest, left_index, right_index)
                    )

    used_left: set[tuple[str, int]] = set()
    used_right: set[tuple[str, int]] = set()
    shared_channels = 0
    shared_messages_total = 0
    for shared_messages, channel_digest, left_index, right_index in sorted(
        candidate_pairs, reverse=True
    ):
        left_key = (channel_digest, left_index)
        right_key = (channel_digest, right_index)
        if left_key in used_left or right_key in used_right:
            continue
        used_left.add(left_key)
        used_right.add(right_key)
        shared_channels += 1
        shared_messages_total += shared_messages

    return shared_channels, shared_messages_total


def duplicates(
    paths: Annotated[
        list[str],
        Parameter(
            name=["paths"],
            help="Files and directories to scan for MCAP duplicates.",
        ),
    ],
    *,
    include_all: Annotated[
        bool,
        Parameter(
            name=["--all"],
            help="Show singleton groups as well as duplicate groups.",
        ),
    ] = False,
    rebuild_missing: Annotated[
        bool,
        Parameter(
            name=["--rebuild-missing"],
            help="Scan files without usable summary data and rebuild summary in memory.",
        ),
    ] = False,
) -> int:
    """Find likely duplicate MCAP recordings using summary and message indexes."""
    if not paths:
        logger.error("At least one file or directory must be specified")
        return 1

    summary_scanned: list[IdentityReadResult] = []
    skipped: list[SkippedFile] = []

    with _create_progress() as progress:
        discovery_task = progress.add_task(
            "Discovering MCAP files",
            total=1,
            current=f"{len(paths):,} input path(s)",
        )
        discovery_error: str | None = None
        try:
            candidates = discover_mcap_candidates(paths)
        except (OSError, ValueError) as exc:
            discovery_error = str(exc)
            candidates = []
        progress.update(
            discovery_task,
            completed=1,
            current=f"{len(candidates):,} MCAP candidate(s)",
        )

        if discovery_error is not None:
            logger.error(discovery_error)
            return 1

        if not candidates:
            logger.error("No MCAP candidates found")
            return 1

        summary_task = progress.add_task(
            "Reading MCAP summaries",
            total=len(candidates),
            current="",
        )
        for candidate in candidates:
            progress.update(summary_task, current=_progress_file(candidate))
            result = _read_summary_identity(candidate, rebuild_missing=rebuild_missing)
            if isinstance(result, IdentityReadResult):
                summary_scanned.append(result)
            else:
                skipped.append(result)
            progress.update(summary_task, advance=1)
        progress.update(
            summary_task,
            completed=len(candidates),
            current=f"{len(summary_scanned):,} summary file(s)",
        )

        summary_pair_count = len(summary_scanned) * (len(summary_scanned) - 1) // 2
        candidate_task = progress.add_task(
            "Finding overlap candidates",
            total=summary_pair_count,
            current="",
        )
        candidate_paths = _candidate_paths_for_index_stage(
            summary_scanned,
            include_all=include_all,
            progress=progress,
            task=candidate_task,
        )

        index_task = progress.add_task(
            "Reading message indexes",
            total=len(candidate_paths),
            current="",
        )
        index_scanned, index_skipped = _read_candidate_index_identities(
            candidate_paths,
            rebuild_missing=rebuild_missing,
            progress=progress,
            task=index_task,
        )
        skipped.extend(index_skipped)

        index_pair_count = len(index_scanned) * (len(index_scanned) - 1) // 2
        analysis_task = progress.add_task(
            "Scoring indexed overlaps",
            total=index_pair_count,
            current="",
        )
        analysis = _analyze_indexed_matches(
            index_scanned,
            include_all=include_all,
            progress=progress,
            task=analysis_task,
        )

    groups = analysis.duplicate_groups
    partial_matches = analysis.partial_matches

    if groups:
        console.print(_build_groups_tree(groups))
    if partial_matches:
        console.print(_build_partial_tree(partial_matches))
    if not groups and not partial_matches:
        console.print("[yellow]No duplicate or partial MCAP matches found[/yellow]")

    duplicate_group_count = sum(1 for group in groups if len(group) > 1)
    console.print(
        "[dim]"
        f"Summary-scanned {len(summary_scanned)} MCAP file(s), "
        f"message-index-checked {len(index_scanned)} candidate file(s), "
        f"found {duplicate_group_count} duplicate group(s), "
        f"{len(partial_matches)} partial match(es)"
        "[/dim]"
    )
    _print_skipped_summary(skipped)
    return 0
