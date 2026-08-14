"""Split command - divide MCAP files into multiple output segments."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import uuid4

from cyclopts import Group, Parameter, validators
from rich.console import Console
from ros_parser.message_path import MessagePathError

from pymcap_cli.cmd._arg_constraints import (
    all_or_none,
    at_least_one,
    constraint_group,
    each_requires,
)
from pymcap_cli.cmd._cli_options import (
    MESSAGE_PATH_GROUP,
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    LatchFromMetadataOption,
    LatchOption,
    MessagePathVariablesOption,
    NoClobberOption,
    SplitAtOption,
)
from pymcap_cli.cmd._cli_options import SPLIT_GROUP as CLI_SPLIT_GROUP
from pymcap_cli.cmd._message_path_options import (
    create_message_path_variables,
    output_template_uses_field,
    parse_message_path_scalar,
)
from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    processing_had_errors,
    resolve_overwrite_policy,
)
from pymcap_cli.cmd._run_processor_multi import run_processor_multi
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    OverwriteCollisionPolicy,
)
from pymcap_cli.core.output_validation import validate_mcap_outputs
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.paired_event_window import (
    PairedEventWindowProcessor,
    PairedWindowPlan,
    discover_paired_windows,
)
from pymcap_cli.core.processors.size_split import SizeSplitProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from pymcap_cli.types.duration import duration_ns_token_converter, parse_duration_ns
from pymcap_cli.types.size import parse_size_bytes
from pymcap_cli.utils import bytes_to_human, confirm_output_overwrite, parse_time_arg

if TYPE_CHECKING:
    from pymcap_cli.core.processors.base import OutputRouter

logger = logging.getLogger(__name__)
console = Console()

# Parameter groups
SPLIT_GROUP = Group("Split Mode")
OUTPUT_GROUP = Group("Output Options")
EXPRESSION_GROUP = Group("Expression Options")
WINDOW_GROUP = Group("Paired Window Options")

# At least one split trigger must be given; the expression-only knobs need --expression.
_SPLIT_MODE_CONSTRAINT = constraint_group(at_least_one)
_EXPRESSION_ONLY_CONSTRAINT = constraint_group(
    each_requires(
        "--expression",
        "--var",
        "--hysteresis",
        "--hysteresis-count",
        "--keep-trailing-context",
        "--keep-trailing-count",
        "--skip-value",
    )
)
_WINDOW_PAIR_CONSTRAINT = constraint_group(all_or_none)
_WINDOW_ONLY_CONSTRAINT = constraint_group(
    each_requires(
        "--window-start",
        "--min-window",
        "--max-window",
        "--orphan-stop",
        "--nested-start",
        "--unclosed-window",
        "--invalid-window",
    )
)


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _source_identity(path: Path) -> _SourceIdentity:
    value = path.stat()
    return _SourceIdentity(
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _ensure_source_unchanged(
    path: Path,
    expected: _SourceIdentity,
    phase: str,
) -> None:
    if _source_identity(path) != expected:
        raise RuntimeError(f"source changed during paired-window {phase}")


def _ensure_paired_processing_succeeded(has_errors: bool) -> None:
    if has_errors:
        raise RuntimeError("paired-window processing reported errors")


def _paired_output_paths(template: str, plan: PairedWindowPlan) -> tuple[Path, ...]:
    paths: list[Path] = []
    resolved_paths: set[Path] = set()
    for index, window in enumerate(plan.windows):
        fields = {
            "index": index,
            "index1": index + 1,
            "key": window.key,
            "start_time": window.start_time,
            "start_time_iso": datetime.fromtimestamp(
                window.start_time / 1e9, tz=timezone.utc
            ).isoformat(),
            "end_time": window.end_time,
            "window_start": window.start_time,
            "window_end": window.end_time,
        }
        path = Path(template.format(**fields))
        resolved = path.resolve(strict=False)
        if resolved in resolved_paths:
            raise ValueError(
                f"multiple paired windows resolve to output path {str(path)!r}; "
                "add '{index}' to the output template"
            )
        paths.append(path)
        resolved_paths.add(resolved)
    return tuple(paths)


def _cleanup_outputs(paths: set[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _prepare_paired_outputs(
    paths: tuple[Path, ...],
    policy: OverwriteCollisionPolicy,
) -> None:
    for path in paths:
        if not path.exists():
            continue
        if policy is OverwriteCollisionPolicy.ERROR:
            raise FileExistsError(f"output already exists: {path}")
        if policy is OverwriteCollisionPolicy.ASK:
            confirm_output_overwrite(path, force=False)


def _publish_paired_outputs(staged: tuple[Path, ...], final: tuple[Path, ...]) -> None:
    if len(staged) != len(final):
        raise RuntimeError("paired-window staged output count mismatch")
    validation = validate_mcap_outputs([], staged)
    if not validation.is_valid:
        raise RuntimeError(f"paired-window {validation.error}")
    for path in staged:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    for staged_path, final_path in zip(staged, final, strict=True):
        staged_path.replace(final_path)
        descriptor = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _coerce_duration_ns(value: int | str | None) -> int | None:
    if isinstance(value, str):
        return parse_duration_ns(value)
    return value


def split(
    file: str,
    *,
    duration: Annotated[
        str | None,
        Parameter(
            name=["--duration"],
            group=[SPLIT_GROUP, _SPLIT_MODE_CONSTRAINT],
            help="Split every N time units (e.g. 60s, 1.5m, 1h); bare numbers are seconds",
        ),
    ] = None,
    split_at: Annotated[
        SplitAtOption, Parameter(group=[CLI_SPLIT_GROUP, _SPLIT_MODE_CONSTRAINT])
    ] = None,
    expression: Annotated[
        str | None,
        Parameter(
            name=["-E", "--expression"],
            group=[SPLIT_GROUP, _SPLIT_MODE_CONSTRAINT, _EXPRESSION_ONLY_CONSTRAINT],
            help=(
                "Split whenever a ros-parser message path changes value, e.g. "
                "'/gps/fix.status.status' (value-change trigger) or "
                "'/detections.objects[:]{confidence>0.8}' (predicate trigger: "
                "match/no-match transitions). Extractors must resolve to a primitive; "
                "predicates normalize to true/false. Segments are numbered — use "
                "'{index:03d}' in --output-template. Messages on other topics "
                "follow the current segment (sticky). Chunks with no "
                "target-topic messages fast-copy without decoding."
            ),
        ),
    ] = None,
    window_start: Annotated[
        str | None,
        Parameter(
            name=["--window-start"],
            group=[
                SPLIT_GROUP,
                _SPLIT_MODE_CONSTRAINT,
                _WINDOW_PAIR_CONSTRAINT,
                _WINDOW_ONLY_CONSTRAINT,
            ],
            help="Absolute MessagePath that opens a paired event window.",
        ),
    ] = None,
    window_end: Annotated[
        str | None,
        Parameter(
            name=["--window-end"],
            group=[SPLIT_GROUP, _SPLIT_MODE_CONSTRAINT, _WINDOW_PAIR_CONSTRAINT],
            help="Absolute MessagePath that closes a paired event window.",
        ),
    ] = None,
    min_window: Annotated[
        int | None,
        Parameter(
            name=["--min-window"],
            group=[WINDOW_GROUP, _WINDOW_ONLY_CONSTRAINT],
            converter=duration_ns_token_converter,
            validator=validators.Number(gt=0),
            help="Minimum accepted start-to-stop duration.",
        ),
    ] = None,
    max_window: Annotated[
        int | None,
        Parameter(
            name=["--max-window"],
            group=[WINDOW_GROUP, _WINDOW_ONLY_CONSTRAINT],
            converter=duration_ns_token_converter,
            validator=validators.Number(gt=0),
            help="Maximum accepted start-to-stop duration.",
        ),
    ] = None,
    orphan_stop: Annotated[
        Literal["error", "ignore", "drop"],
        Parameter(
            name=["--orphan-stop"],
            group=[WINDOW_GROUP, _WINDOW_ONLY_CONSTRAINT],
            help="Policy for a stop observed while no window is open.",
        ),
    ] = "error",
    nested_start: Annotated[
        Literal["error", "ignore", "drop"],
        Parameter(
            name=["--nested-start"],
            group=[WINDOW_GROUP, _WINDOW_ONLY_CONSTRAINT],
            help="Policy for a start observed while a window is already open.",
        ),
    ] = "error",
    unclosed_window: Annotated[
        Literal["error", "ignore", "drop"],
        Parameter(
            name=["--unclosed-window"],
            group=[WINDOW_GROUP, _WINDOW_ONLY_CONSTRAINT],
            help="Policy for a start that remains open at end of file.",
        ),
    ] = "error",
    invalid_window: Annotated[
        Literal["error", "drop"],
        Parameter(
            name=["--invalid-window"],
            group=[WINDOW_GROUP, _WINDOW_ONLY_CONSTRAINT],
            help="Policy for a window outside --min-window/--max-window.",
        ),
    ] = "error",
    var: Annotated[
        MessagePathVariablesOption,
        Parameter(group=[MESSAGE_PATH_GROUP, _EXPRESSION_ONLY_CONSTRAINT]),
    ] = None,
    max_size: Annotated[
        str | None,
        Parameter(
            name=["--max-size"],
            group=[SPLIT_GROUP, _SPLIT_MODE_CONSTRAINT],
            help=(
                "Split when accumulated message bytes exceed N (e.g. '1G', "
                "'500MB', '2GiB'). Segment count is dynamic. Output file "
                "size is approximate — depends on output compression."
            ),
        ),
    ] = None,
    output_template: Annotated[
        str,
        Parameter(
            name=["-t", "--output-template"],
            group=OUTPUT_GROUP,
            help=(
                "Python format template for output filenames. Variables: {index}, {index1}, "
                "{key}, {value}, {start_time}, {start_time_iso}, {end_time}, "
                "{window_start}, {window_end}. Standard format specs apply, e.g. "
                "'{value:+d}' and '{index:03d}'. {value} requires --expression; window fields "
                "require paired event windows."
            ),
        ),
    ] = "output_{index:03d}.mcap",
    hysteresis: Annotated[
        int | None,
        Parameter(
            name=["--hysteresis"],
            group=[EXPRESSION_GROUP, _EXPRESSION_ONLY_CONSTRAINT],
            converter=duration_ns_token_converter,
            validator=validators.Number(gt=0),
            help=(
                "Time hysteresis for --expression: a new value must persist "
                "for at least this duration before a segment cut fires "
                "(e.g. '500ms', '2s'). Combines with --hysteresis-count."
            ),
        ),
    ] = None,
    hysteresis_count: Annotated[
        int | None,
        Parameter(
            name=["--hysteresis-count"],
            group=[EXPRESSION_GROUP, _EXPRESSION_ONLY_CONSTRAINT],
            validator=validators.Number(gt=0),
            help=(
                "Count hysteresis for --expression: a new value must appear "
                "this many times before a segment cut fires. Combines with "
                "--hysteresis."
            ),
        ),
    ] = None,
    keep_trailing_context: Annotated[
        int | None,
        Parameter(
            name=["--keep-trailing-context"],
            group=[EXPRESSION_GROUP, _EXPRESSION_ONLY_CONSTRAINT],
            converter=duration_ns_token_converter,
            validator=validators.Number(gt=0),
            help=(
                "After a transition, also write target-topic messages from "
                "this duration (e.g. '500ms') into the previous segment "
                "for context. Combines with --keep-trailing-count."
            ),
        ),
    ] = None,
    keep_trailing_count: Annotated[
        int | None,
        Parameter(
            name=["--keep-trailing-count"],
            group=[EXPRESSION_GROUP, _EXPRESSION_ONLY_CONSTRAINT],
            validator=validators.Number(gt=0),
            help=(
                "After a transition, also write up to this many "
                "target-topic messages into the previous segment for "
                "context. Combines with --keep-trailing-context."
            ),
        ),
    ] = None,
    skip_value: Annotated[
        list[str] | None,
        Parameter(
            name=["--skip-value"],
            group=[EXPRESSION_GROUP, _EXPRESSION_ONLY_CONSTRAINT],
            help=(
                "Expression value to omit from the output (repeatable). Values use JSON "
                "scalars when possible; negative values use --skip-value=-1."
            ),
        ),
    ] = None,
    latch: LatchOption = None,
    latch_from_metadata: LatchFromMetadataOption = False,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
    """Split an MCAP file into multiple output segments.

    Supports duration, timestamp, expression, size, and paired-event splitting.

    Parameters
    ----------
    file
        Path to the MCAP file to split (local file or HTTP/HTTPS URL).
    duration
        Split interval, e.g. "60s", "1.5m", "1h"; bare numbers are seconds.
    split_at
        Timestamps at which to split (ns integer or RFC3339 format).
    expression
        ros-parser message path; each distinct value becomes a segment.
    max_size
        Approximate byte budget per segment (e.g. ``1G``, ``500MB``).
        Output file size is approximate — depends on output compression.
    output_template
        Python format string for output filenames. Available variables:
        {index}, {index1}, {key}, {value}, {start_time}, {start_time_iso}, {end_time}.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    force
        Force overwrite of output files without confirmation.
    no_clobber
        Fail instead of prompting if any split output path already exists.
    delete_source
        Delete the source after validating every output for readability and
        expected message preservation. URL inputs and any source whose path equals one
        of the outputs are skipped.

    Examples
    --------
    ```
    # Split every 60 seconds
    pymcap-cli split input.mcap --duration 60s

    # Split every 5 minutes with custom naming
    pymcap-cli split input.mcap --duration 5m --output-template "seg_{index:03d}.mcap"

    # Split at specific timestamps
    pymcap-cli split input.mcap --split-at 1000000000 --split-at 2000000000

    # Split at RFC3339 timestamps
    pymcap-cli split input.mcap --split-at "2024-01-15T10:00:00Z"

    # Split by a message-path expression (new segment on each value change)
    pymcap-cli split input.mcap --expression '/gps/fix.status.status' \\
        -t 'gps_{index:03d}.mcap'

    # Filter-triggered split: new segment when confidence>0.8 turns on/off
    pymcap-cli split input.mcap \\
        --expression '/detections.objects[:]{confidence>0.8}' \\
        -t 'hits_{index:03d}.mcap'

    # Split when each output reaches roughly 1 GB
    pymcap-cli split input.mcap --max-size 1G -t 'shard_{index:03d}.mcap'
    ```
    """
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)

    # Parse split-at timestamps. Relative anchors are not supported here.
    split_points: list[int] = []
    if split_at:
        for ts in split_at:
            try:
                split_points.append(parse_time_arg(ts))
            except ValueError:
                logger.exception(f"Error parsing timestamp '{ts}'")
                return 1

    # Build split processors
    processors: list[OutputRouter] = []
    paired_router: PairedEventWindowProcessor | None = None
    paired_identity: _SourceIdentity | None = None
    owned_paired_outputs: set[Path] = set()
    paired_final_paths: tuple[Path, ...] = ()
    paired_staged_paths: tuple[Path, ...] = ()
    processing_output_template = output_template
    if duration:
        try:
            duration_ns = parse_duration_ns(duration)
        except ValueError:
            logger.exception(f"Error parsing duration '{duration}'")
            return 1
        if duration_ns <= 0:
            logger.error("Duration must be positive.")
            return 1
        processors.append(DurationSplitProcessor(duration_ns))
        logger.info(f"Duration split: every {duration} ({duration_ns:,} ns)")

    if split_points:
        processors.append(TimestampSplitProcessor(split_points))
        logger.info(f"Timestamp split: {len(split_points)} point(s)")

    if max_size:
        try:
            max_size_bytes = parse_size_bytes(max_size)
        except ValueError:
            logger.exception(f"Error parsing max-size '{max_size}'")
            return 1
        processors.append(SizeSplitProcessor(max_size_bytes))
        logger.info(
            f"Size split: every {bytes_to_human(max_size_bytes)} ({max_size_bytes:,} bytes)"
        )

    if (window_start is None) != (window_end is None):
        logger.error("--window-start and --window-end must be provided together")
        return 1
    if window_start is not None and window_end is not None:
        if duration or split_points or expression or max_size:
            logger.error("paired event windows cannot be combined with other split modes")
            return 1
        if min_window is not None and max_window is not None and min_window > max_window:
            logger.error("--min-window must not exceed --max-window")
            return 1
        source_path = Path(file)
        if not source_path.is_file():
            logger.error("paired event windows require one local seekable MCAP file")
            return 1
        try:
            paired_identity = _source_identity(source_path)
            plan = discover_paired_windows(
                source_path,
                window_start,
                window_end,
                minimum_duration_ns=min_window,
                maximum_duration_ns=max_window,
                orphan_stop=orphan_stop,
                nested_start=nested_start,
                unclosed_window=unclosed_window,
                invalid_window=invalid_window,
            )
            _ensure_source_unchanged(source_path, paired_identity, "discovery")
            paired_router = PairedEventWindowProcessor(window_start, window_end, plan)
            processors.append(paired_router)
            paired_final_paths = _paired_output_paths(output_template, plan)
            _prepare_paired_outputs(paired_final_paths, overwrite_policy)
            suffix = f".pymcap-partial-{os.getpid()}-{uuid4().hex}"
            processing_output_template = f"{output_template}{suffix}"
            paired_staged_paths = _paired_output_paths(processing_output_template, plan)
            owned_paired_outputs = set(paired_staged_paths)
        except Exception:
            logger.exception("Error discovering paired event windows")
            return 1
        logger.info(f"Paired event windows: {len(plan.windows)} validated window(s)")

    try:
        template_uses_value = output_template_uses_field(output_template, "value")
    except ValueError:
        logger.exception(f"Invalid output template {output_template!r}")
        return 1

    if expression:
        # Hysteresis / trailing-context only apply to expression splits.
        try:
            hysteresis_ns = _coerce_duration_ns(hysteresis)
            trailing_ns = _coerce_duration_ns(keep_trailing_context)
        except ValueError:
            logger.exception("Error parsing hysteresis/trailing-context duration")
            return 1
        try:
            variables = create_message_path_variables(var)
            skip_values = tuple(
                parse_message_path_scalar(value, source="--skip-value")
                for value in skip_value or ()
            )
            processors.append(
                ExpressionSplitProcessor(
                    expression,
                    hysteresis_ns=hysteresis_ns,
                    hysteresis_count=hysteresis_count,
                    trailing_context_ns=trailing_ns,
                    trailing_context_count=keep_trailing_count,
                    variables=variables,
                    skip_values=skip_values,
                    require_value=bool(skip_values) or template_uses_value,
                )
            )
        except MessagePathError:
            logger.exception(f"Error parsing expression '{expression}'")
            return 1
        except ValueError:
            logger.exception("Invalid expression split option")
            return 1
        logger.info(f"Expression split: {expression}")
    elif template_uses_value:
        logger.error("{value} in --output-template requires --expression.")
        return 1

    # Display split mode
    modes = []
    if duration:
        modes.append("Duration")
    if split_points:
        modes.append("Timestamp")
    if expression:
        modes.append("Expression")
    if paired_router is not None:
        modes.append("Paired window")
    if max_size:
        modes.append("Size")
    logger.info(f"Mode: {' + '.join(modes)} split")

    input_options = InputOptions.from_args(
        latch_topics=latch,
        latch_from_metadata=latch_from_metadata,
    )

    try:
        result = run_processor_multi(
            files=[file],
            input_options=input_options,
            output_options=OutputOptions(
                routers=processors,
                output_template=processing_output_template,
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
        if paired_router is not None:
            _ensure_paired_processing_succeeded(processing_had_errors(result.stats))
            paired_router.validate_complete()
            assert paired_identity is not None
            _ensure_source_unchanged(Path(file), paired_identity, "processing")
            _publish_paired_outputs(paired_staged_paths, paired_final_paths)
            owned_paired_outputs.clear()
            assert result.processor.output_manager is not None
            segments = sorted(
                result.processor.output_manager.segments.values(),
                key=lambda segment: segment.index,
            )
            for segment, final_path in zip(segments, paired_final_paths, strict=True):
                segment.path = str(final_path)
    except Exception:
        _cleanup_outputs(owned_paired_outputs)
        logger.exception("Error during splitting")
        return 1

    # Report results
    logger.info("[green]✓ Splitting completed successfully![/green]")
    console.print(
        f"Processed {result.stats.messages_processed:,} messages, "
        f"wrote {result.stats.writer_statistics.message_count:,} messages"
    )

    # Per-segment stats
    if result.stats.writer_statistics:
        stats = result.stats.writer_statistics
        console.print(f"Time range: {stats.message_start_time:,} - {stats.message_end_time:,} ns")

    assert result.processor.output_manager is not None
    console.print(f"Created {len(result.processor.output_manager.segments)} output file(s)")
    for _, segment in sorted(
        result.processor.output_manager.segments.items(), key=lambda x: x[1].index
    ):
        assert segment.writer is not None
        seg_stats = segment.writer.statistics
        console.print(
            f"  [{segment.index}] {segment.path}: "
            f"{seg_stats.message_count:,} messages, "
            f"{bytes_to_human(Path(segment.path).stat().st_size)} "
            f"({seg_stats.chunk_count} chunks)"
        )

    if delete_source:
        if processing_had_errors(result.stats):
            logger.error("Processing reported errors — source file preserved.")
            return 1
        outputs = [
            Path(segment.path) for segment in result.processor.output_manager.segments.values()
        ]
        return finalize_delete_source(
            sources=[file],
            outputs=outputs,
            lossy_topic_patterns=[".*"] if expression is not None or paired_router else [],
        )

    return 0
