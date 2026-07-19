"""Split command - divide MCAP files into multiple output segments."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Group, Parameter, validators
from rich.console import Console
from ros_parser.message_path import MessagePathError

from pymcap_cli.cmd._arg_constraints import at_least_one, constraint_group, each_requires
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
from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.size_split import SizeSplitProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from pymcap_cli.types.duration import duration_ns_token_converter, parse_duration_ns
from pymcap_cli.types.size import parse_size_bytes
from pymcap_cli.utils import bytes_to_human, parse_time_arg

if TYPE_CHECKING:
    from pymcap_cli.core.processors.base import OutputRouter

logger = logging.getLogger(__name__)
console = Console()

# Parameter groups
SPLIT_GROUP = Group("Split Mode")
OUTPUT_GROUP = Group("Output Options")
EXPRESSION_GROUP = Group("Expression Options")

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
                "{key}, {value}, {start_time}, {start_time_iso}, {end_time}. Standard format "
                "specs apply, e.g. '{value:+d}' and '{index:03d}'. {value} requires "
                "--expression."
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

    Supports duration-based splitting (every N seconds/minutes/hours),
    timestamp-based splitting (at specific points), or both combined.

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
        Delete the source file after every split output is validated
        (header + summary). URL inputs and any source whose path equals one
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
                output_template=output_template,
                compression=compression,
                chunk_size=chunk_size,
                overwrite_policy=overwrite_policy,
            ),
        )
    except Exception:
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
        return finalize_delete_source(sources=[file], outputs=outputs)

    return 0
